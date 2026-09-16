/**
 * router.c - MoE Top-K routing, batch union, lookahead prefetching, and prefix cache.
 */

#include "router.h"
#include <math.h>
#include <float.h>
#include <string.h>

void vitna_route_topk(
    const float* logits,
    size_t num_experts,
    size_t top_k,
    vitna_routing_choice_t* out
) {
    if (!logits || num_experts == 0 || top_k == 0 || !out) return;
    if (top_k > num_experts) top_k = num_experts;

    /* Initialize output */
    for (size_t k = 0; k < top_k; k++) {
        out[k].expert_idx = 0;
        out[k].weight = -FLT_MAX;
    }

    /* Track top-K elements */
    for (size_t e = 0; e < num_experts; e++) {
        float score = logits[e];
        if (score > out[top_k - 1].weight) {
            /* Insertion sort into top-K */
            size_t pos = top_k - 1;
            while (pos > 0 && score > out[pos - 1].weight) {
                out[pos] = out[pos - 1];
                pos--;
            }
            out[pos].expert_idx = (uint32_t)e;
            out[pos].weight = score;
        }
    }

    /* Softmax over the top-K selections */
    float max_val = out[0].weight;
    float sum_exp = 0.0f;
    for (size_t k = 0; k < top_k; k++) {
        out[k].weight = expf(out[k].weight - max_val);
        sum_exp += out[k].weight;
    }

    /* Normalize */
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (size_t k = 0; k < top_k; k++) {
        out[k].weight *= inv_sum;
    }
}

void vitna_route_batch_union(
    const vitna_routing_choice_t* choices,
    size_t total_choices,
    uint32_t* out_expert_ids,
    size_t* out_count
) {
    if (!choices || total_choices == 0 || !out_expert_ids || !out_count) {
        if (out_count) *out_count = 0;
        return;
    }

    size_t unique = 0;
    for (size_t i = 0; i < total_choices; i++) {
        uint32_t candidate = choices[i].expert_idx;
        bool seen = false;
        for (size_t u = 0; u < unique; u++) {
            if (out_expert_ids[u] == candidate) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            out_expert_ids[unique++] = candidate;
        }
    }
    *out_count = unique;
}

float vitna_router_entropy(const float* logits, size_t num_experts) {
    if (!logits || num_experts == 0) return 0.0f;

    float max_val = logits[0];
    for (size_t i = 1; i < num_experts; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < num_experts; i++) {
        sum += expf(logits[i] - max_val);
    }
    if (sum <= 0.0f) return 0.0f;

    float inv_sum = 1.0f / sum;
    float entropy = 0.0f;
    for (size_t i = 0; i < num_experts; i++) {
        float p = expf(logits[i] - max_val) * inv_sum;
        if (p > 1e-7f) {
            entropy -= p * (logf(p) / 0.69314718f); /* log2 */
        }
    }
    return entropy;
}

vitna_routing_strategy_t vitna_route_adaptive(
    const float* logits,
    size_t num_experts,
    size_t standard_top_k,
    float dense_entropy_threshold,
    vitna_routing_choice_t* out_choices,
    size_t* out_count
) {
    if (!out_count) return VITNA_ROUTING_DENSE_SHORTCUT;

    float entropy = vitna_router_entropy(logits, num_experts);

    /* Very low entropy: simple token (punctuation, format, stopword) */
    if (entropy < dense_entropy_threshold) {
        *out_count = 0;
        return VITNA_ROUTING_DENSE_SHORTCUT;
    }

    /* Intermediate entropy: dominant single expert is sufficient */
    if (entropy < dense_entropy_threshold * 1.5f && standard_top_k > 1) {
        vitna_route_topk(logits, num_experts, 1, out_choices);
        *out_count = 1;
        return VITNA_ROUTING_SPARSE_TOP1;
    }

    /* Standard high-entropy MoE activation */
    vitna_route_topk(logits, num_experts, standard_top_k, out_choices);
    *out_count = standard_top_k;
    return VITNA_ROUTING_STANDARD_TOPK;
}

void vitna_router_predict_downstream(
    const vitna_router_lookahead_t* lookahead,
    size_t current_layer,
    const vitna_routing_choice_t* current_choices,
    size_t num_choices,
    uint32_t* out_predicted_experts,
    size_t* out_count
) {
    if (!out_predicted_experts || !out_count) return;
    *out_count = 0;
    if (!lookahead || !current_choices || num_choices == 0) return;

    /* Semantic correlation heuristic: experts in early layers strongly correlate
       with clustered domain specialists in downstream layers (L+2 to L+4) */
    size_t predicted = 0;
    for (size_t i = 0; i < num_choices; i++) {
        if (current_choices[i].weight < (lookahead->correlation_threshold > 0 ? lookahead->correlation_threshold : 0.15f)) {
            continue;
        }

        uint32_t base_expert = current_choices[i].expert_idx;
        /* Downstream affinity projection: specialist cluster offsets */
        uint32_t affinity_1 = (base_expert + (uint32_t)current_layer * 2) % (uint32_t)lookahead->num_experts;
        uint32_t affinity_2 = (base_expert + 1) % (uint32_t)lookahead->num_experts;

        uint32_t candidates[2] = { affinity_1, affinity_2 };
        for (int c = 0; c < 2; c++) {
            bool exists = false;
            for (size_t p = 0; p < predicted; p++) {
                if (out_predicted_experts[p] == candidates[c]) {
                    exists = true;
                    break;
                }
            }
            if (!exists && predicted < 32) {
                out_predicted_experts[predicted++] = candidates[c];
            }
        }
    }

    *out_count = predicted;
}

void vitna_prefix_cache_init(vitna_prefix_cache_t* cache) {
    if (!cache) return;
    memset(cache, 0, sizeof(vitna_prefix_cache_t));
}

bool vitna_prefix_cache_lookup(
    const vitna_prefix_cache_t* cache,
    uint64_t prefix_hash,
    uint32_t* out_expert_ids,
    size_t* out_count
) {
    if (!cache || !out_expert_ids || !out_count) return false;
    *out_count = 0;

    for (size_t i = 0; i < cache->count; i++) {
        if (cache->entries[i].prefix_hash == prefix_hash) {
            size_t copy_count = cache->entries[i].expert_count;
            if (copy_count > VITNA_PREFIX_MAX_EXPERTS) copy_count = VITNA_PREFIX_MAX_EXPERTS;
            memcpy(out_expert_ids, cache->entries[i].expert_ids, copy_count * sizeof(uint32_t));
            *out_count = copy_count;
            return true;
        }
    }
    return false;
}

void vitna_prefix_cache_record(
    vitna_prefix_cache_t* cache,
    uint64_t prefix_hash,
    const uint32_t* expert_ids,
    size_t count
) {
    if (!cache || !expert_ids || count == 0) return;

    /* Check if already present */
    for (size_t i = 0; i < cache->count; i++) {
        if (cache->entries[i].prefix_hash == prefix_hash) {
            cache->entries[i].hit_count++;
            return;
        }
    }

    /* Insert new entry */
    size_t slot;
    if (cache->count < VITNA_PREFIX_CACHE_CAPACITY) {
        slot = cache->count++;
    } else {
        /* Simple FIFO or lowest hit eviction */
        size_t min_idx = 0;
        uint64_t min_hits = cache->entries[0].hit_count;
        for (size_t i = 1; i < cache->count; i++) {
            if (cache->entries[i].hit_count < min_hits) {
                min_hits = cache->entries[i].hit_count;
                min_idx = i;
            }
        }
        slot = min_idx;
    }

    cache->entries[slot].prefix_hash = prefix_hash;
    size_t store_count = count > VITNA_PREFIX_MAX_EXPERTS ? VITNA_PREFIX_MAX_EXPERTS : count;
    memcpy(cache->entries[slot].expert_ids, expert_ids, store_count * sizeof(uint32_t));
    cache->entries[slot].expert_count = store_count;
    cache->entries[slot].hit_count = 1;
}
