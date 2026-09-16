/**
 * kv_cache.h - Paged block KV cache and RoPE attention positioning.
 *
 * Rules:
 * - Pure C11 zero-dependency implementation
 * - No em-dashes in comments or code
 * - Paged fixed-block allocation avoiding fragmentation
 */

#ifndef VITNA_KV_CACHE_H
#define VITNA_KV_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VITNA_KV_BLOCK_SIZE 16

/**
 * Paged memory block holding key and value states for a block of tokens.
 */
typedef struct {
    uint32_t block_id;
    uint16_t num_tokens;
    float* key_data;    /* [VITNA_KV_BLOCK_SIZE, num_kv_heads, head_dim] */
    float* value_data;  /* [VITNA_KV_BLOCK_SIZE, num_kv_heads, head_dim] */
} vitna_kv_block_t;

#define VITNA_MAX_PINNED_PREFIXES 16

typedef struct {
    uint64_t prefix_hash;
    size_t num_tokens;
    uint32_t** pinned_block_tables; /* [num_layers][num_blocks] */
    bool is_active;
} vitna_pinned_kv_prefix_t;

/**
 * Logical sequence KV cache descriptor pointing to physical block tables.
 */
typedef struct {
    size_t num_layers;
    size_t num_kv_heads;
    size_t head_dim;
    size_t total_blocks;
    size_t free_blocks;
    vitna_kv_block_t* block_pool;
    uint32_t** layer_block_tables; /* [num_layers][max_blocks_per_seq] */
    size_t* layer_seq_lens;        /* [num_layers] */
    vitna_pinned_kv_prefix_t pinned_prefixes[VITNA_MAX_PINNED_PREFIXES];
    size_t num_pinned_prefixes;
} vitna_paged_kv_cache_t;

/**
 * Initialize paged block KV cache.
 */
bool vitna_kv_cache_init(
    vitna_paged_kv_cache_t* cache,
    size_t num_layers,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_total_tokens
);

/**
 * Free all allocated blocks in paged KV cache.
 */
void vitna_kv_cache_free(vitna_paged_kv_cache_t* cache);

/**
 * Allocate or fetch block for sequence appending at given layer and position.
 */
vitna_kv_block_t* vitna_kv_cache_get_or_alloc(
    vitna_paged_kv_cache_t* cache,
    size_t layer_idx,
    size_t token_pos
);

/**
 * Pin current sequence KV states as a reusable prefix associated with prefix_hash.
 */
bool vitna_kv_cache_pin_prefix(
    vitna_paged_kv_cache_t* cache,
    uint64_t prefix_hash,
    size_t num_tokens
);

/**
 * Match an incoming prompt prefix hash and restore pre-warmed KV cache blocks.
 * Returns true and matched token count if found, skipping prefill attention compute.
 */
bool vitna_kv_cache_match_prefix(
    vitna_paged_kv_cache_t* cache,
    uint64_t prefix_hash,
    size_t* out_matched_tokens
);

/**
 * Apply Rotary Position Embedding (RoPE) to query or key vectors.
 */
void vitna_rope_apply(
    float* vec,
    size_t dim,
    size_t pos,
    float base_freq
);

#define VITNA_PREFILL_CHUNK_SIZE 256

/**
 * Continuous chunked prefill state tracker.
 */
typedef struct {
    size_t total_prompt_tokens;
    size_t processed_tokens;
    size_t chunk_size;
} vitna_chunked_prefill_t;

/**
 * Initialize chunked prefill state.
 */
void vitna_chunked_prefill_init(
    vitna_chunked_prefill_t* prefill,
    size_t total_prompt_tokens,
    size_t chunk_size
);

/**
 * Step through next chunk of prefill.
 * Returns number of tokens processed in this step, or 0 when prefill is complete.
 */
size_t vitna_chunked_prefill_step(
    vitna_chunked_prefill_t* prefill,
    vitna_paged_kv_cache_t* cache
);

/**
 * Dynamic LRU page eviction when block pool is exhausted.
 * Evicts oldest unpinned blocks to free capacity for active generation.
 */
bool vitna_kv_cache_evict_lru(
    vitna_paged_kv_cache_t* cache,
    size_t needed_blocks
);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_KV_CACHE_H */
