/**
 * grammar.c - Kernel-level Grammar-Constrained Decoding implementation.
 *
 * Rules:
 * - Pure C11 zero external dependencies
 * - No em-dashes anywhere in comments or code
 * - Validates JSON structure deterministically character by character
 */

#include "grammar.h"
#include <string.h>

bool vitna_grammar_init_json(vitna_grammar_matcher_t* matcher) {
    if (!matcher) return false;
    memset(matcher, 0, sizeof(*matcher));
    matcher->state = VITNA_GRAMMAR_EXPECT_ROOT;
    matcher->stack_depth = 0;
    matcher->in_escape = false;
    matcher->complete = false;
    matcher->tokens_accepted = 0;
    matcher->tokens_rejected = 0;
    return true;
}

static inline bool is_whitespace(char c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
}

bool vitna_grammar_feed_char(vitna_grammar_matcher_t* matcher, char c) {
    if (!matcher) return false;

    if (matcher->complete) {
        if (is_whitespace(c)) return true;
        return false; /* Disallow any trailing chatter once root JSON is complete */
    }

    switch (matcher->state) {
        case VITNA_GRAMMAR_EXPECT_ROOT: {
            if (is_whitespace(c)) return true;
            if (c == '{' || c == '[') {
                if (matcher->stack_depth >= VITNA_GRAMMAR_MAX_DEPTH) return false;
                matcher->stack[matcher->stack_depth++] = c;
                matcher->state = (c == '{') ? VITNA_GRAMMAR_IN_OBJECT_KEY : VITNA_GRAMMAR_EXPECT_VALUE;
                return true;
            }
            return false;
        }

        case VITNA_GRAMMAR_IN_OBJECT_KEY: {
            if (is_whitespace(c) && !matcher->in_escape) return true;
            if (c == '}' && matcher->stack_depth > 0 && matcher->stack[matcher->stack_depth - 1] == '{') {
                /* Empty object {} */
                matcher->stack_depth--;
                if (matcher->stack_depth == 0) {
                    matcher->complete = true;
                    matcher->state = VITNA_GRAMMAR_COMPLETE;
                } else {
                    matcher->state = VITNA_GRAMMAR_EXPECT_DELIMITER;
                }
                return true;
            }
            if (c == '"') {
                matcher->state = VITNA_GRAMMAR_EXPECT_COLON;
                return true;
            }
            return true; /* Allow key characters */
        }

        case VITNA_GRAMMAR_EXPECT_COLON: {
            if (is_whitespace(c)) return true;
            if (c == ':') {
                matcher->state = VITNA_GRAMMAR_EXPECT_VALUE;
                return true;
            }
            return false;
        }

        case VITNA_GRAMMAR_EXPECT_VALUE: {
            if (is_whitespace(c)) return true;
            if (c == '{' || c == '[') {
                if (matcher->stack_depth >= VITNA_GRAMMAR_MAX_DEPTH) return false;
                matcher->stack[matcher->stack_depth++] = c;
                matcher->state = (c == '{') ? VITNA_GRAMMAR_IN_OBJECT_KEY : VITNA_GRAMMAR_EXPECT_VALUE;
                return true;
            }
            if (c == ']' && matcher->stack_depth > 0 && matcher->stack[matcher->stack_depth - 1] == '[') {
                /* Empty array [] */
                matcher->stack_depth--;
                if (matcher->stack_depth == 0) {
                    matcher->complete = true;
                    matcher->state = VITNA_GRAMMAR_COMPLETE;
                } else {
                    matcher->state = VITNA_GRAMMAR_EXPECT_DELIMITER;
                }
                return true;
            }
            if (c == '"') {
                matcher->state = VITNA_GRAMMAR_IN_STRING_VALUE;
                return true;
            }
            /* Primitive value (number, true, false, null) */
            if ((c >= '0' && c <= '9') || c == '-' || c == 't' || c == 'f' || c == 'n') {
                matcher->state = VITNA_GRAMMAR_EXPECT_DELIMITER;
                return true;
            }
            return false;
        }

        case VITNA_GRAMMAR_IN_STRING_VALUE: {
            if (matcher->in_escape) {
                matcher->in_escape = false;
                return true;
            }
            if (c == '\\') {
                matcher->in_escape = true;
                return true;
            }
            if (c == '"') {
                matcher->state = VITNA_GRAMMAR_EXPECT_DELIMITER;
                return true;
            }
            return true;
        }

        case VITNA_GRAMMAR_EXPECT_DELIMITER: {
            if (is_whitespace(c)) return true;
            if ((c >= '0' && c <= '9') || c == '.' || c == 'r' || c == 'u' || c == 'e' || c == 'a' || c == 'l' || c == 's') {
                return true; /* Allow continuation of numeric or boolean literal */
            }
            if (c == ',') {
                if (matcher->stack_depth > 0) {
                    char top = matcher->stack[matcher->stack_depth - 1];
                    matcher->state = (top == '{') ? VITNA_GRAMMAR_IN_OBJECT_KEY : VITNA_GRAMMAR_EXPECT_VALUE;
                    return true;
                }
                return false;
            }
            if (c == '}' && matcher->stack_depth > 0 && matcher->stack[matcher->stack_depth - 1] == '{') {
                matcher->stack_depth--;
                if (matcher->stack_depth == 0) {
                    matcher->complete = true;
                    matcher->state = VITNA_GRAMMAR_COMPLETE;
                }
                return true;
            }
            if (c == ']' && matcher->stack_depth > 0 && matcher->stack[matcher->stack_depth - 1] == '[') {
                matcher->stack_depth--;
                if (matcher->stack_depth == 0) {
                    matcher->complete = true;
                    matcher->state = VITNA_GRAMMAR_COMPLETE;
                }
                return true;
            }
            return false;
        }

        case VITNA_GRAMMAR_COMPLETE:
            return is_whitespace(c);

        default:
            return false;
    }
}

bool vitna_grammar_feed_token(vitna_grammar_matcher_t* matcher, const char* token_str) {
    if (!matcher || !token_str) return false;

    for (const char* p = token_str; *p; p++) {
        if (!vitna_grammar_feed_char(matcher, *p)) {
            matcher->tokens_rejected++;
            matcher->state = VITNA_GRAMMAR_ERROR;
            return false;
        }
    }

    matcher->tokens_accepted++;
    return true;
}

bool vitna_grammar_is_token_valid(const vitna_grammar_matcher_t* matcher, const char* token_str) {
    if (!matcher || !token_str) return false;

    /* Create temporary copy of matcher to simulate token feed */
    vitna_grammar_matcher_t temp = *matcher;

    for (const char* p = token_str; *p; p++) {
        if (!vitna_grammar_feed_char(&temp, *p)) {
            return false;
        }
    }

    return true;
}

bool vitna_grammar_is_complete(const vitna_grammar_matcher_t* matcher) {
    return matcher && matcher->complete;
}
