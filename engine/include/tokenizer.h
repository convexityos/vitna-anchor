/**
 * tokenizer.h - Byte-level BPE, read from a Hugging Face tokenizer.json.
 *
 * Encoding follows the tokenizers library step by step. Added (special)
 * tokens are matched first, leftmost and longest. The text between them is
 * split by the pre-tokenizer: optionally each numeric character on its own
 * (Digits), then the GPT-2 pattern
 *
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 *
 * Each piece's bytes become byte-level symbols, a byte the vocabulary lacks
 * is dropped (there is no unknown token), and merges are applied lowest rank
 * first, the leftmost pair first among equal ranks.
 *
 * Only what the tokenizer.json in hand needs is supported: no normalizer, a
 * ByteLevel pre-tokenizer alone or after Digits, no post-processor (so no
 * BOS is added), a ByteLevel decoder, and a BPE model without dropout,
 * unknown token or byte fallback. Anything else is refused on load, not
 * approximated.
 */

#ifndef VITNA_TOKENIZER_H
#define VITNA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vitna_tokenizer vitna_tokenizer_t;

/** A growable list of token ids. */
typedef struct {
    int32_t* ids;
    size_t count;
    size_t cap;
} vitna_token_list_t;

bool vitna_token_list_push(vitna_token_list_t* list, int32_t id);
void vitna_token_list_free(vitna_token_list_t* list);

/** Load tokenizer.json. Returns NULL, with a reason in err, if it cannot be read or is unsupported. */
vitna_tokenizer_t* vitna_tokenizer_load(const char* path, char* err, size_t err_len);

void vitna_tokenizer_free(vitna_tokenizer_t* tok);

/** One more than the largest token id. */
size_t vitna_tokenizer_vocab_size(const vitna_tokenizer_t* tok);

/** Encode len bytes of UTF-8 text, appending ids to out. False only if memory runs out. */
bool vitna_tokenizer_encode(const vitna_tokenizer_t* tok, const char* text, size_t len, vitna_token_list_t* out);

/** The bytes a token stands for, or NULL for an id outside the vocabulary. */
const unsigned char* vitna_tokenizer_token_bytes(const vitna_tokenizer_t* tok, int32_t id, size_t* len);

/** Whether id is an added token marked special. */
bool vitna_tokenizer_is_special(const vitna_tokenizer_t* tok, int32_t id);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_TOKENIZER_H */
