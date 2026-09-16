/**
 * speculative.h - Speculative MoE drafting and parallel verification engine.
 *
 * Accelerates streamed MoE autoregressive generation by pairing a lightweight
 * draft predictor (pinned in host memory) with the target MoE expert engine.
 *
 * Rules:
 * - Pure C11 with zero external dependencies
 * - Strictly zero em-dashes anywhere in comments, code, or strings
 * - Rejection sampling guarantees exact mathematical equivalence to target distribution
 */

#ifndef VITNA_SPECULATIVE_H
#define VITNA_SPECULATIVE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VITNA_MAX_SPECULATIVE_LOOKAHEAD 8
#define VITNA_DEFAULT_DRAFT_WINDOW 4

typedef struct {
    uint32_t token_id;
    float draft_prob;
    float target_prob;
    bool accepted;
} vitna_speculative_token_t;

typedef struct {
    size_t lookahead_window;
    float acceptance_threshold;
    uint32_t rng_state;
    size_t total_drafted;
    size_t total_accepted;
    size_t verification_passes;
} vitna_speculative_drafter_t;

/**
 * Initialize a speculative drafter.
 * @param drafter Pointer to drafter struct
 * @param lookahead_window Number of speculative tokens to predict ahead (1-8)
 * @param seed Random seed for rejection sampling
 */
void vitna_speculative_init(
    vitna_speculative_drafter_t* drafter,
    size_t lookahead_window,
    uint32_t seed
);

/**
 * Propose candidate draft tokens for speculative evaluation.
 * @param drafter Pointer to drafter struct
 * @param current_token_id Current base token
 * @param candidates Output array for candidate tokens (capacity >= lookahead_window)
 * @param num_candidates Output count of proposed tokens
 */
void vitna_speculative_propose(
    vitna_speculative_drafter_t* drafter,
    uint32_t current_token_id,
    vitna_speculative_token_t* candidates,
    size_t* num_candidates
);

/**
 * Verify draft candidate tokens in parallel against target MoE distribution.
 * Applies rejection sampling: accept if random() <= min(1, p_target / p_draft).
 * First rejected token terminates the accepted prefix; remaining tokens are discarded.
 *
 * @param drafter Pointer to drafter struct
 * @param candidates Array of proposed candidate tokens with target_prob filled
 * @param count Number of candidates to verify
 * @param accepted_count Output count of accepted candidate tokens (0 to count)
 * @returns Token ID of the final emitted token (either last accepted or resampled correction)
 */
uint32_t vitna_speculative_verify(
    vitna_speculative_drafter_t* drafter,
    vitna_speculative_token_t* candidates,
    size_t count,
    size_t* accepted_count
);

/**
 * Calculate empirical acceptance rate.
 * @param drafter Pointer to drafter struct
 * @returns Acceptance rate between 0.0 and 1.0 (e.g. 0.784 for 78.4%)
 */
float vitna_speculative_acceptance_rate(const vitna_speculative_drafter_t* drafter);

/**
 * Calculate effective speculative speedup factor.
 * Formula: (accepted_tokens + 1) / (1 + draft_overhead_ratio)
 * @param drafter Pointer to drafter struct
 * @returns Speedup multiple (e.g. 2.85 for 2.85x baseline)
 */
float vitna_speculative_speedup_factor(const vitna_speculative_drafter_t* drafter);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_SPECULATIVE_H */
