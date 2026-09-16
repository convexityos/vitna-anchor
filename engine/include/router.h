/**
 * router.h - Mixture-of-Experts routing, Top-K selection, and batch union.
 */

#ifndef VITNA_ROUTER_H
#define VITNA_ROUTER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t expert_idx;
    float weight;
} vitna_routing_choice_t;

/**
 * Top-K Softmax routing for a single token.
 *
 * @param logits Array of raw routing scores, length num_experts
 * @param num_experts Total number of routed experts in this layer
 * @param top_k Number of experts to activate (e.g. 2 or 8)
 * @param out Array of length top_k to receive the selected experts and normalized weights
 */
void vitna_route_topk(
    const float* logits,
    size_t num_experts,
    size_t top_k,
    vitna_routing_choice_t* out
);

/**
 * Compute the deduplicated batch union of expert IDs across multiple tokens.
 *
 * @param choices Array of (batch_size * top_k) routing choices
 * @param total_choices Count of choices
 * @param out_expert_ids Output buffer to hold unique expert IDs
 * @param out_count Pointer to receive count of unique experts
 */
void vitna_route_batch_union(
    const vitna_routing_choice_t* choices,
    size_t total_choices,
    uint32_t* out_expert_ids,
    size_t* out_count
);

typedef enum {
    VITNA_ROUTING_DENSE_SHORTCUT = 0, /* 0 experts: dense anchor execution only, 0 disk I/O */
    VITNA_ROUTING_SPARSE_TOP1    = 1, /* 1 expert activated instead of Top-K */
    VITNA_ROUTING_STANDARD_TOPK  = 2  /* Standard Top-K MoE specialist experts */
} vitna_routing_strategy_t;

/**
 * Calculate routing distribution entropy across experts.
 */
float vitna_router_entropy(const float* logits, size_t num_experts);

/**
 * Adaptive MoE router that skips specialist disk I/O for simple / predictable tokens.
 */
vitna_routing_strategy_t vitna_route_adaptive(
    const float* logits,
    size_t num_experts,
    size_t standard_top_k,
    float dense_entropy_threshold,
    vitna_routing_choice_t* out_choices,
    size_t* out_count
);

/**
 * Semantic Lookahead & Downstream Expert Predictor.
 * Predicts expert affinity across downstream layers (L+2 to L+8) from early-layer routing activations.
 */
typedef struct {
    size_t num_experts;
    size_t num_layers;
    float correlation_threshold;
} vitna_router_lookahead_t;

/**
 * Predict downstream expert activations given current layer routing choices.
 */
void vitna_router_predict_downstream(
    const vitna_router_lookahead_t* lookahead,
    size_t current_layer,
    const vitna_routing_choice_t* current_choices,
    size_t num_choices,
    uint32_t* out_predicted_experts,
    size_t* out_count
);

/**
 * Prompt Prefix Affinity Cache Entry.
 * Pre-pins anchor experts for known recurring system prompts and schemas.
 */
#define VITNA_PREFIX_MAX_EXPERTS 32
#define VITNA_PREFIX_CACHE_CAPACITY 64

typedef struct {
    uint64_t prefix_hash;
    uint32_t expert_ids[VITNA_PREFIX_MAX_EXPERTS];
    size_t expert_count;
    uint64_t hit_count;
} vitna_prefix_entry_t;

typedef struct {
    vitna_prefix_entry_t entries[VITNA_PREFIX_CACHE_CAPACITY];
    size_t count;
} vitna_prefix_cache_t;

void vitna_prefix_cache_init(vitna_prefix_cache_t* cache);
bool vitna_prefix_cache_lookup(const vitna_prefix_cache_t* cache, uint64_t prefix_hash, uint32_t* out_expert_ids, size_t* out_count);
void vitna_prefix_cache_record(vitna_prefix_cache_t* cache, uint64_t prefix_hash, const uint32_t* expert_ids, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_ROUTER_H */
