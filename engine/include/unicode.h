/**
 * unicode.h - The character classes the tokenizer's split pattern needs.
 */

#ifndef VITNA_UNICODE_H
#define VITNA_UNICODE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \p{L}: general category Lu, Ll, Lt, Lm or Lo. */
bool vitna_uni_is_letter(uint32_t cp);

/** \p{N}: general category Nd, Nl or No. */
bool vitna_uni_is_number(uint32_t cp);

/** \s: the White_Space property. */
bool vitna_uni_is_space(uint32_t cp);

/** The Unicode version the letter and number tables come from. */
const char* vitna_uni_version(void);

/**
 * Decode one UTF-8 sequence at s[0..len). Returns its length in bytes (1 to
 * 4) and the code point in *cp. A byte that does not start a valid sequence
 * is returned alone, with *cp = 0x110000 + that byte, which no class contains.
 */
size_t vitna_utf8_decode(const unsigned char* s, size_t len, uint32_t* cp);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_UNICODE_H */
