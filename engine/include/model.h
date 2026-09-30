/**
 * model.h - A dense Llama-architecture model in float32, on the CPU or, in an
 * engine built with the CUDA path, on an NVIDIA GPU.
 *
 * Reads config.json and a single model.safetensors from a model directory,
 * and runs one token at a time: embedding, then per layer RMSNorm,
 * grouped-query attention with half-split rotary embeddings over a
 * key-value cache, a residual add, RMSNorm, a SwiGLU MLP and a residual add;
 * then a final RMSNorm and the output projection, tied to the embedding
 * when the config says so. This is the computation of Hugging Face's
 * LlamaForCausalLM with eager attention.
 *
 * What the config asks for and this does not do (rope scaling, attention or
 * MLP biases, another activation, sharded checkpoints) is refused on load.
 *
 * A model loads on the CPU. vitna_llama_use_cuda moves its forward pass to
 * the GPU, where model_cuda.cu runs the same computation; nothing falls back
 * from one device to the other.
 */

#ifndef VITNA_MODEL_H
#define VITNA_MODEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t n_layers;
    size_t hidden;
    size_t intermediate;
    size_t n_heads;
    size_t n_kv_heads;
    size_t head_dim;
    size_t vocab;
    size_t max_positions;
    float rms_eps;
    float rope_theta;
    bool tied_embeddings;
} vitna_llama_config_t;

typedef struct {
    const void* data;
    vitna_dtype_t dtype;
    size_t rows;
    size_t cols;
} vitna_matrix_t;

typedef struct {
    float* attn_norm;
    float* mlp_norm;
    vitna_matrix_t q, k, v, o, gate, up, down;
} vitna_llama_layer_t;

/* The model's state on a GPU, in model_cuda.cu. */
struct vitna_cuda_model;

typedef struct {
    vitna_llama_config_t cfg;
    vitna_safetensors_t st;
    vitna_matrix_t embed;
    vitna_matrix_t lm_head;
    float* final_norm;
    vitna_llama_layer_t* layers;
    float* inv_freq;          /* head_dim / 2 rotary frequencies */

    size_t ctx;               /* positions the key-value cache holds */
    size_t n_past;            /* positions filled */
    float* k_cache;           /* [n_layers][ctx][n_kv_heads * head_dim] */
    float* v_cache;

    /* scratch */
    float *x, *xn, *q, *k, *v, *att, *proj, *gate, *up, *scores, *cos_t, *sin_t;

    /* Set by vitna_llama_use_cuda: the forward pass then runs on the GPU,
     * with its own key-value cache there, and k_cache and v_cache go unused. */
    struct vitna_cuda_model* cuda;

    /* For tests, set by vitna_llama_fail_step_once: while fail_armed, the
     * step at position fail_at fails. */
    bool fail_armed;
    size_t fail_at;
} vitna_llama_t;

/**
 * Load the model in dir, with a key-value cache of ctx positions (0 for the
 * lesser of the model's maximum and 4096). Returns false, with a reason in
 * err, if the model cannot be read or asks for something unsupported.
 */
bool vitna_llama_load(vitna_llama_t* m, const char* dir, size_t ctx, char* err, size_t err_len);

void vitna_llama_free(vitna_llama_t* m);

/** Forget every position, so the next token is at position 0. */
void vitna_llama_reset(vitna_llama_t* m);

/**
 * Keep the first n positions and forget the rest, so the next token is at
 * position n. A position's keys and values depend only on the tokens up to
 * it, so what is kept is exactly what feeding those n tokens again would
 * compute. n larger than the positions filled changes nothing.
 */
void vitna_llama_truncate(vitna_llama_t* m, size_t n);

/**
 * Run one token at the next position (m->n_past), adding its keys and values
 * to the cache. Writes vocab logits to logits unless it is NULL, which skips
 * the output projection. Returns false if the cache is full or the token is
 * out of range, or, on a GPU, if the device reports an error, which is then
 * printed to stderr, or when a test asked for this step to fail
 * (vitna_llama_fail_step_once). A step that returns false leaves m->n_past
 * where it was.
 */
bool vitna_llama_step(vitna_llama_t* m, int32_t token, float* logits);

/**
 * For tests: make the next step at position pos fail, once, as a step on the
 * GPU does when the device reports an error, so that what its callers do
 * then can be tested on a CPU. That step returns false before computing
 * anything, on either device, and says so on stderr. A later step at pos runs
 * as usual.
 */
void vitna_llama_fail_step_once(vitna_llama_t* m, size_t pos);

/** True when this engine was built with the CUDA path (VITNA_CUDA). */
bool vitna_llama_cuda_built(void);

/**
 * Check that the GPU can be used before anything is loaded: this engine has
 * the CUDA path, and the CUDA runtime finds a device. Returns false, with the
 * reason in err, when either is missing.
 */
bool vitna_llama_cuda_probe(char* err, size_t err_len);

/**
 * Move a loaded model's forward pass to the first CUDA device: upload every
 * weight once, in its stored dtype, and hold the key-value cache on the
 * device. Call it once, straight after vitna_llama_load, before any step.
 * Returns false, with the reason in err, if this engine was built without
 * CUDA or the device cannot take the model. The model is then as it was, on
 * the CPU; a caller that asked for the GPU should stop, not run it there.
 */
bool vitna_llama_use_cuda(vitna_llama_t* m, char* err, size_t err_len);

/**
 * Where the forward pass runs, as the server's start-up line says it:
 * "matvec path avx2+fma" on the CPU, or "on CUDA device 0, <name> (sm_86)".
 * Writes to buf and returns it.
 */
const char* vitna_llama_device(const vitna_llama_t* m, char* buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_MODEL_H */
