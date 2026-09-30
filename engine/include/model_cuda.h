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
 * the key-value cache and scratch there. m stays as loaded; its weights are
 * read once, here. Returns NULL, with the reason in err, on failure.
 */
struct vitna_cuda_model* vitna_cuda_create(const vitna_llama_t* m, const float* cos_tab, const float* sin_tab, char* err, size_t err_len);

/**
 * Run one token at position pos, writing its keys and values into the
 * device's cache at pos. Copies vocab logits back to logits unless it is
 * NULL. The caller has checked token and pos. Returns false, with the reason
 * in err, if the device reports an error.
 */
bool vitna_cuda_step(struct vitna_cuda_model* g, int32_t token, size_t pos, float* logits, char* err, size_t err_len);

/**
 * The fewest tokens worth giving vitna_cuda_steps: fewer run faster a step at
 * a time. SIZE_MAX for a model whose prompts cannot run together.
 */
size_t vitna_cuda_prompt_min(const struct vitna_cuda_model* g);

/**
 * The most tokens vitna_cuda_steps_exact takes at once for this model, 0 for
 * a model that takes none.
 */
size_t vitna_cuda_exact_max(const struct vitna_cuda_model* g);

/**
 * Run count tokens at positions pos to pos + count - 1 as count calls to
 * vitna_cuda_step would, writing every one's logits, count x vocab, in one
 * pass that reads each weight once: the same values as those steps, bit for
 * bit. count is 1 to vitna_cuda_exact_max. The caller has checked the tokens
 * and the room in the cache. Returns false, with the reason in err, if the
 * device reports an error.
 */
bool vitna_cuda_steps_exact(struct vitna_cuda_model* g, const int32_t* tokens, size_t count, size_t pos, float* logits, char* err,
                            size_t err_len);

/**
 * Run count tokens at positions pos to pos + count - 1 together, writing
 * their keys and values into the device's cache. If logits is not NULL, it
 * receives vocab logits for each of the last rows positions, rows x vocab.
 * The caller has checked the tokens and the room in the cache. Returns
 * false, with the reason in err, if the device reports an error, or if the
 * model's prompts cannot run together (vitna_cuda_prompt_min).
 */
bool vitna_cuda_steps(struct vitna_cuda_model* g, const int32_t* tokens, size_t count, size_t pos, float* logits, size_t rows,
                      char* err, size_t err_len);

/** The device, for a person to read: its name and compute capability. */
const char* vitna_cuda_device_name(const struct vitna_cuda_model* g);

void vitna_cuda_free(struct vitna_cuda_model* g);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_MODEL_CUDA_H */
