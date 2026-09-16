/**
 * generate.h - End-to-end autoregressive token generation pipeline.
 *
 * Rules:
 * - Pure C11 zero external dependencies
 * - No em-dashes anywhere in comments or code
 * - Coordinates prefetching, GEMV execution, and cryptographic attestation
 */

#ifndef VITNA_GENERATE_H
#define VITNA_GENERATE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "expert_store.h"
#include "router.h"
#include "kv_cache.h"
#include "crypto.h"
#include "radix_kv.h"
#include "grammar.h"
#include "speculative.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*vitna_token_callback_fn)(
    uint32_t token_id,
    const char* token_str,
    bool is_final,
    void* user_data
);

typedef struct {
    size_t max_new_tokens;
    float temperature;
    size_t top_k;
    vitna_expert_store_t* expert_store;
    vitna_router_lookahead_t* router_lookahead;
    vitna_paged_kv_cache_t* kv_cache;
    vitna_radix_tree_t* radix_tree;
    vitna_grammar_matcher_t* grammar;
    vitna_speculative_drafter_t* drafter;
    vitna_token_callback_fn on_token;
    void* user_data;
} vitna_generate_config_t;

typedef struct {
    size_t tokens_generated;
    double elapsed_ms;
    double toks_per_sec;
    size_t cache_hits;
    size_t direct_io_reads;
    size_t dense_shortcuts_taken;
    size_t prefix_tokens_matched;
    size_t radix_prefix_matched;
    size_t speculative_accepted;
    float speculative_speedup;
    size_t grammar_tokens_masked;
    char trajectory_hash_hex[65];
    bool airgap_verified;
} vitna_generate_stats_t;

/**
 * Run autoregressive token generation loop.
 */
bool vitna_generate_run(
    const char* prompt,
    const vitna_generate_config_t* config,
    vitna_generate_stats_t* stats
);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_GENERATE_H */
