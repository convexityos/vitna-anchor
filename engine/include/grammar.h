/**
 * grammar.h - Kernel-level Grammar-Constrained Decoding for structured JSON.
 *
 * Rules:
 * - Pure C11 zero external dependencies
 * - No em-dashes anywhere in comments or code
 * - Enforces 100% valid JSON syntax and provides early-stop triggers
 */

#ifndef VITNA_GRAMMAR_H
#define VITNA_GRAMMAR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VITNA_GRAMMAR_EXPECT_ROOT = 0,
    VITNA_GRAMMAR_IN_OBJECT_KEY,
    VITNA_GRAMMAR_EXPECT_COLON,
    VITNA_GRAMMAR_EXPECT_VALUE,
    VITNA_GRAMMAR_IN_STRING_VALUE,
    VITNA_GRAMMAR_EXPECT_DELIMITER,
    VITNA_GRAMMAR_COMPLETE,
    VITNA_GRAMMAR_ERROR
} vitna_grammar_state_t;

#define VITNA_GRAMMAR_MAX_DEPTH 64

typedef struct {
    vitna_grammar_state_t state;
    char stack[VITNA_GRAMMAR_MAX_DEPTH];
    size_t stack_depth;
    bool in_escape;
    bool complete;
    size_t tokens_accepted;
    size_t tokens_rejected;
} vitna_grammar_matcher_t;

/**
 * Initialize grammar matcher for root JSON object/array decoding.
 */
bool vitna_grammar_init_json(vitna_grammar_matcher_t* matcher);

/**
 * Feed a single character into grammar state machine.
 * Returns true if transition is valid, false if syntax violation.
 */
bool vitna_grammar_feed_char(vitna_grammar_matcher_t* matcher, char c);

/**
 * Feed a token string into the grammar state machine.
 */
bool vitna_grammar_feed_token(vitna_grammar_matcher_t* matcher, const char* token_str);

/**
 * Speculatively test if a candidate token string is valid at current state
 * without modifying matcher state.
 */
bool vitna_grammar_is_token_valid(const vitna_grammar_matcher_t* matcher, const char* token_str);

/**
 * Check if root JSON structure is complete (stack reached depth 0 after opening).
 */
bool vitna_grammar_is_complete(const vitna_grammar_matcher_t* matcher);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_GRAMMAR_H */
