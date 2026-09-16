/**
 * kv_cache.c - Implementation of paged block KV cache and RoPE.
 *
 * Rules:
 * - No em-dashes anywhere in comments or code
 * - Zero external runtime dependencies
 * - High-speed block indexing and cache lookup
 */

#include "kv_cache.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool vitna_kv_cache_init(
    vitna_paged_kv_cache_t* cache,
    size_t num_layers,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_total_tokens
) {
    if (!cache || num_layers == 0 || num_kv_heads == 0 || head_dim == 0) {
        return false;
    }

    cache->num_layers = num_layers;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;

    size_t blocks_needed = (max_total_tokens + VITNA_KV_BLOCK_SIZE - 1) / VITNA_KV_BLOCK_SIZE;
    cache->total_blocks = blocks_needed;
    cache->free_blocks = blocks_needed;

    cache->block_pool = (vitna_kv_block_t*)calloc(blocks_needed, sizeof(vitna_kv_block_t));
    if (!cache->block_pool) {
        return false;
    }

    size_t block_floats = VITNA_KV_BLOCK_SIZE * num_kv_heads * head_dim;
    for (size_t i = 0; i < blocks_needed; i++) {
        cache->block_pool[i].block_id = (uint32_t)i;
        cache->block_pool[i].num_tokens = 0;
        cache->block_pool[i].key_data = (float*)malloc(block_floats * sizeof(float));
        cache->block_pool[i].value_data = (float*)malloc(block_floats * sizeof(float));
        if (!cache->block_pool[i].key_data || !cache->block_pool[i].value_data) {
            vitna_kv_cache_free(cache);
            return false;
        }
    }

    cache->layer_block_tables = (uint32_t**)calloc(num_layers, sizeof(uint32_t*));
    cache->layer_seq_lens = (size_t*)calloc(num_layers, sizeof(size_t));
    if (!cache->layer_block_tables || !cache->layer_seq_lens) {
        vitna_kv_cache_free(cache);
        return false;
    }

    size_t max_blocks_per_layer = (max_total_tokens / VITNA_KV_BLOCK_SIZE) + 4;
    for (size_t l = 0; l < num_layers; l++) {
        cache->layer_block_tables[l] = (uint32_t*)malloc(max_blocks_per_layer * sizeof(uint32_t));
        if (!cache->layer_block_tables[l]) {
            vitna_kv_cache_free(cache);
            return false;
        }
        for (size_t b = 0; b < max_blocks_per_layer; b++) {
            cache->layer_block_tables[l][b] = UINT32_MAX;
        }
    }

    return true;
}

void vitna_kv_cache_free(vitna_paged_kv_cache_t* cache) {
    if (!cache) return;

    if (cache->block_pool) {
        for (size_t i = 0; i < cache->total_blocks; i++) {
            free(cache->block_pool[i].key_data);
            free(cache->block_pool[i].value_data);
        }
        free(cache->block_pool);
        cache->block_pool = NULL;
    }

    if (cache->layer_block_tables) {
        for (size_t l = 0; l < cache->num_layers; l++) {
            free(cache->layer_block_tables[l]);
        }
        free(cache->layer_block_tables);
        cache->layer_block_tables = NULL;
    }

    free(cache->layer_seq_lens);
    cache->layer_seq_lens = NULL;
}

vitna_kv_block_t* vitna_kv_cache_get_or_alloc(
    vitna_paged_kv_cache_t* cache,
    size_t layer_idx,
    size_t token_pos
) {
    if (!cache || layer_idx >= cache->num_layers) return NULL;

    size_t block_idx = token_pos / VITNA_KV_BLOCK_SIZE;
    uint32_t physical_block_id = cache->layer_block_tables[layer_idx][block_idx];

    if (physical_block_id == UINT32_MAX) {
        /* Find first free physical block */
        for (size_t i = 0; i < cache->total_blocks; i++) {
            if (cache->block_pool[i].num_tokens == 0) {
                physical_block_id = (uint32_t)i;
                cache->layer_block_tables[layer_idx][block_idx] = physical_block_id;
                cache->free_blocks--;
                break;
            }
        }
        /* If pool exhausted, trigger LRU eviction of unpinned blocks */
        if (physical_block_id == UINT32_MAX) {
            if (vitna_kv_cache_evict_lru(cache, 1)) {
                for (size_t i = 0; i < cache->total_blocks; i++) {
                    if (cache->block_pool[i].num_tokens == 0) {
                        physical_block_id = (uint32_t)i;
                        cache->layer_block_tables[layer_idx][block_idx] = physical_block_id;
                        cache->free_blocks--;
                        break;
                    }
                }
            }
        }
    }

    if (physical_block_id == UINT32_MAX || physical_block_id >= cache->total_blocks) {
        return NULL; /* Out of memory blocks */
    }

    vitna_kv_block_t* block = &cache->block_pool[physical_block_id];
    size_t token_in_block = token_pos % VITNA_KV_BLOCK_SIZE;
    if (token_in_block + 1 > block->num_tokens) {
        block->num_tokens = (uint16_t)(token_in_block + 1);
    }
    return block;
}

void vitna_rope_apply(
    float* vec,
    size_t dim,
    size_t pos,
    float base_freq
) {
    if (!vec || dim < 2) return;

    if (base_freq <= 0.0f) {
        base_freq = 10000.0f;
    }

    for (size_t i = 0; i < dim; i += 2) {
        float theta = (float)pos / powf(base_freq, (float)i / (float)dim);
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float v0 = vec[i];
        float v1 = vec[i + 1];

        vec[i]     = v0 * cos_theta - v1 * sin_theta;
        vec[i + 1] = v0 * sin_theta + v1 * cos_theta;
    }
}

bool vitna_kv_cache_pin_prefix(
    vitna_paged_kv_cache_t* cache,
    uint64_t prefix_hash,
    size_t num_tokens
) {
    if (!cache || num_tokens == 0 || cache->num_pinned_prefixes >= VITNA_MAX_PINNED_PREFIXES) {
        return false;
    }

    size_t blocks_needed = (num_tokens + VITNA_KV_BLOCK_SIZE - 1) / VITNA_KV_BLOCK_SIZE;
    size_t slot = cache->num_pinned_prefixes;

    vitna_pinned_kv_prefix_t* entry = &cache->pinned_prefixes[slot];
    entry->prefix_hash = prefix_hash;
    entry->num_tokens = num_tokens;
    entry->is_active = true;

    entry->pinned_block_tables = (uint32_t**)malloc(cache->num_layers * sizeof(uint32_t*));
    if (!entry->pinned_block_tables) return false;

    for (size_t l = 0; l < cache->num_layers; l++) {
        entry->pinned_block_tables[l] = (uint32_t*)malloc(blocks_needed * sizeof(uint32_t));
        if (!entry->pinned_block_tables[l]) return false;
        for (size_t b = 0; b < blocks_needed; b++) {
            entry->pinned_block_tables[l][b] = cache->layer_block_tables[l][b];
        }
    }

    cache->num_pinned_prefixes++;
    return true;
}

bool vitna_kv_cache_match_prefix(
    vitna_paged_kv_cache_t* cache,
    uint64_t prefix_hash,
    size_t* out_matched_tokens
) {
    if (!cache || !out_matched_tokens) return false;
    *out_matched_tokens = 0;

    for (size_t i = 0; i < cache->num_pinned_prefixes; i++) {
        vitna_pinned_kv_prefix_t* entry = &cache->pinned_prefixes[i];
        if (entry->is_active && entry->prefix_hash == prefix_hash) {
            size_t blocks = (entry->num_tokens + VITNA_KV_BLOCK_SIZE - 1) / VITNA_KV_BLOCK_SIZE;
            for (size_t l = 0; l < cache->num_layers; l++) {
                for (size_t b = 0; b < blocks; b++) {
                    cache->layer_block_tables[l][b] = entry->pinned_block_tables[l][b];
                }
                cache->layer_seq_lens[l] = entry->num_tokens;
            }
            *out_matched_tokens = entry->num_tokens;
            return true;
        }
    }

    return false;
}

bool vitna_kv_cache_evict_lru(vitna_paged_kv_cache_t* cache, size_t needed_blocks) {
    if (!cache || needed_blocks == 0) return false;

    /* Build a fast mask of pinned physical blocks */
    bool* is_pinned = (bool*)calloc(cache->total_blocks, sizeof(bool));
    if (!is_pinned) return false;

    for (size_t p = 0; p < cache->num_pinned_prefixes; p++) {
        if (!cache->pinned_prefixes[p].is_active) continue;
        size_t blocks = (cache->pinned_prefixes[p].num_tokens + VITNA_KV_BLOCK_SIZE - 1) / VITNA_KV_BLOCK_SIZE;
        for (size_t l = 0; l < cache->num_layers; l++) {
            for (size_t b = 0; b < blocks; b++) {
                uint32_t phys_id = cache->pinned_prefixes[p].pinned_block_tables[l][b];
                if (phys_id < cache->total_blocks) {
                    is_pinned[phys_id] = true;
                }
            }
        }
    }

    size_t evicted = 0;
    for (size_t i = 0; i < cache->total_blocks && evicted < needed_blocks; i++) {
        if (!is_pinned[i] && cache->block_pool[i].num_tokens > 0) {
            /* Evict this block: reset tokens and unbind from active layer tables */
            cache->block_pool[i].num_tokens = 0;
            for (size_t l = 0; l < cache->num_layers; l++) {
                size_t max_blocks_per_layer = (cache->total_blocks / cache->num_layers) + 4;
                for (size_t b = 0; b < max_blocks_per_layer; b++) {
                    if (cache->layer_block_tables[l][b] == (uint32_t)i) {
                        cache->layer_block_tables[l][b] = UINT32_MAX;
                    }
                }
            }
            cache->free_blocks++;
            evicted++;
        }
    }

    free(is_pinned);
    return evicted >= needed_blocks;
}

void vitna_chunked_prefill_init(
    vitna_chunked_prefill_t* prefill,
    size_t total_prompt_tokens,
    size_t chunk_size
) {
    if (!prefill) return;
    prefill->total_prompt_tokens = total_prompt_tokens;
    prefill->processed_tokens = 0;
    prefill->chunk_size = chunk_size > 0 ? chunk_size : VITNA_PREFILL_CHUNK_SIZE;
}

size_t vitna_chunked_prefill_step(
    vitna_chunked_prefill_t* prefill,
    vitna_paged_kv_cache_t* cache
) {
    if (!prefill || prefill->processed_tokens >= prefill->total_prompt_tokens) {
        return 0;
    }

    size_t remaining = prefill->total_prompt_tokens - prefill->processed_tokens;
    size_t slice = remaining < prefill->chunk_size ? remaining : prefill->chunk_size;

    if (cache) {
        for (size_t l = 0; l < cache->num_layers; l++) {
            for (size_t t = 0; t < slice; t++) {
                vitna_kv_cache_get_or_alloc(cache, l, prefill->processed_tokens + t);
            }
            cache->layer_seq_lens[l] = prefill->processed_tokens + slice;
        }
    }

    prefill->processed_tokens += slice;
    return slice;
}
