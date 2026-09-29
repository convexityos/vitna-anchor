/**
 * model.h - A dense Llama-architecture model on the CPU, in float32.
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
 * Run one token at the next position (m->n_past), adding its keys and values
 * to the cache. Writes vocab logits to logits unless it is NULL, which skips
 * the output projection. Returns false if the cache is full or the token is
 * out of range.
 */
bool vitna_llama_step(vitna_llama_t* m, int32_t token, float* logits);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_MODEL_H */
