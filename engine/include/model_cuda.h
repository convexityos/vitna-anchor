/**
 * model_cuda.h - The forward pass of model.c on an NVIDIA GPU, in float32.
 *
 * Compiled only into an engine built with the CUDA path (VITNA_CUDA), and
 * called only by model.c. See model_cuda.cu.
 */

#ifndef VITNA_MODEL_CUDA_H
#define VITNA_MODEL_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Whether the CUDA runtime finds a device. Returns false, with the reason in err, if not. */
bool vitna_cuda_probe(char* err, size_t err_len);

/**
 * Upload m's weights to the first device, with the rotary cos and sin of
 * every position its cache holds (m->ctx rows of head_dim / 2), and allocate
 * the key-value cache, m->seqs sequences of m->ctx positions, and scratch
 * there. m stays as loaded; its weights are read once, here. Returns NULL,
 * with the reason in err, on failure.
 */
struct vitna_cuda_model* vitna_cuda_create(const vitna_llama_t* m, const float* cos_tab, const float* sin_tab, char* err, size_t err_len);

/**
 * Run one token at position pos of sequence seq, writing its keys and values
 * into that sequence's cache at pos. Copies vocab logits back to logits
 * unless it is NULL. The caller has checked token, seq and pos. Returns
 * false, with the reason in err, if the device reports an error.
 */
bool vitna_cuda_step(struct vitna_cuda_model* g, size_t seq, int32_t token, size_t pos, float* logits, char* err, size_t err_len);

/**
 * The fewest tokens worth giving vitna_cuda_steps: fewer run faster a step at
 * a time. SIZE_MAX for a model whose prompts cannot run together.
 */
size_t vitna_cuda_prompt_min(const struct vitna_cuda_model* g);

/**
 * The fewest tokens a piece of a prompt may have for vitna_cuda_steps to run
 * it exactly as the same tokens run in one call: pieces of fewer take other
 * kernels, whose sums round differently. 1 for a model whose prompts run a
 * step at a time.
 */
size_t vitna_cuda_prompt_piece_min(const struct vitna_cuda_model* g);

/**
 * The most rows vitna_cuda_rows takes at once for this model, 0 for a model
 * that takes none.
 */
size_t vitna_cuda_exact_max(const struct vitna_cuda_model* g);

/** A row of a pass: a token, its position, and the sequence whose cache it reads and writes. */
typedef struct {
    int32_t token;
    int32_t pos;
    int32_t seq;
} vitna_cuda_row_t;

/**
 * Run count rows as count calls to vitna_cuda_step would, in one pass that
 * reads each weight once, and copy row i's logits to logits[i] where that is
 * not NULL: the same values as those steps, bit for bit. Rows of one
 * sequence hold consecutive positions, in order; rows of different
 * sequences share nothing. count is 1 to vitna_cuda_exact_max. The caller
 * has checked the tokens, sequences and positions. Returns false, with the
 * reason in err, if the device reports an error.
 */
bool vitna_cuda_rows(struct vitna_cuda_model* g, const vitna_cuda_row_t* rows, size_t count, float* const* logits, char* err,
                     size_t err_len);

/**
 * Run count tokens of sequence seq at positions pos to pos + count - 1
 * together, writing their keys and values into that sequence's cache. If
 * logits is not NULL, it receives vocab logits for each of the last rows
 * positions, rows x vocab. The caller has checked the tokens and the room in
 * the cache. Returns false, with the reason in err, if the device reports an
 * error, or if the model's prompts cannot run together
 * (vitna_cuda_prompt_min).
 */
bool vitna_cuda_steps(struct vitna_cuda_model* g, size_t seq, const int32_t* tokens, size_t count, size_t pos, float* logits, size_t rows,
                      char* err, size_t err_len);

/** The device, for a person to read: its name and compute capability. */
const char* vitna_cuda_device_name(const struct vitna_cuda_model* g);

void vitna_cuda_free(struct vitna_cuda_model* g);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_MODEL_CUDA_H */
