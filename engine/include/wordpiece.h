/**
 * wordpiece.h - A BERT tokenizer, read from a Hugging Face tokenizer.json.
 *
 * Encoding follows the tokenizers library step by step. Added tokens are
 * matched first, as written, leftmost and longest. The text between them is
 * normalized as BertNormalizer does: characters that are other (Cc, Cf, Co,
 * Cs), NUL and U+FFFD dropped, each whitespace character made a space, a
 * space put either side of each CJK ideograph, then NFD with the nonspacing
 * marks (Mn) dropped, then each character lowercased on its own. Then it is
 * split as BertPreTokenizer splits it, at whitespace, which goes, and at
 * punctuation (ASCII punctuation and P*), which stays as a word of its own.
 * Each word becomes the longest vocabulary entries that spell it, left to
 * right, every one after the first with the continuing prefix ("##"); a word
 * no entries spell, or longer than max_input_chars_per_word characters, is
 * the unknown token. The post-processor's tokens go either side ([CLS] and
 * [SEP]).
 *
 * The library takes its character data from crates of three Unicode
 * versions, and so does this: categories from 8.0, NFD from 9.0, lowercase
 * from 17.0 (engine/tools/gen_wordpiece_unicode.py says how they were found).
 *
 * Only what BERT's tokenizer.json files need is supported: BertNormalizer,
 * BertPreTokenizer, a WordPiece model, a TemplateProcessing or BertProcessing
 * post-processor for a single text, and added tokens matched as written. No
 * truncation or padding is applied; a caller checks lengths itself. Anything
 * else is refused on load, not approximated.
 */

#ifndef VITNA_WORDPIECE_H
#define VITNA_WORDPIECE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "json.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vitna_wordpiece vitna_wordpiece_t;

/** Whether a parsed tokenizer.json's model is WordPiece. */
bool vitna_wordpiece_wanted(const vitna_json_value_t* root);

/** Build the tokenizer a parsed tokenizer.json describes. Returns NULL, with a reason in err, if it is unsupported. */
vitna_wordpiece_t* vitna_wordpiece_from_json(const vitna_json_value_t* root, char* err, size_t err_len);

void vitna_wordpiece_free(vitna_wordpiece_t* t);

/** One more than the largest token id. */
size_t vitna_wordpiece_vocab_size(const vitna_wordpiece_t* t);

/** Encode len bytes of UTF-8 text, the post-processor's tokens included, appending ids to out. False only if memory runs out. */
bool vitna_wordpiece_encode(const vitna_wordpiece_t* t, const char* text, size_t len, vitna_token_list_t* out);

/** The text the normalizer makes of len bytes, as a new buffer of *out_len bytes the caller frees, or NULL if memory runs out. */
unsigned char* vitna_wordpiece_normalize(const vitna_wordpiece_t* t, const char* text, size_t len, size_t* out_len);

/** The text of a token, or NULL for an id outside the vocabulary. */
const unsigned char* vitna_wordpiece_token_bytes(const vitna_wordpiece_t* t, int32_t id, size_t* len);

/** Whether id is an added token marked special. */
bool vitna_wordpiece_is_special(const vitna_wordpiece_t* t, int32_t id);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_WORDPIECE_H */
