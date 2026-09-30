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

/** The device, for a person to read: its name and compute capability. */
const char* vitna_cuda_device_name(const struct vitna_cuda_model* g);

void vitna_cuda_free(struct vitna_cuda_model* g);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_MODEL_CUDA_H */
