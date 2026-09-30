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

    size_t ctx;               /* positions the key-value cache holds for each sequence */
    size_t seqs;              /* sequences it holds, each run on its own (see vitna_llama_step_rows) */
    size_t* past;             /* [seqs]: each sequence's positions filled */
    float* k_cache;           /* [seqs][n_layers][ctx][n_kv_heads * head_dim] */
    float* v_cache;

    /* scratch */
    float *x, *xn, *q, *k, *v, *att, *proj, *gate, *up, *scores, *cos_t, *sin_t;

    /* Set by vitna_llama_use_cuda: the forward pass then runs on the GPU,
     * with its own key-value cache there, and k_cache and v_cache go unused. */
    struct vitna_cuda_model* cuda;

    /* For tests, set by vitna_llama_fail_step_once: while fail_armed, the
     * step at position fail_at, in any sequence, fails. */
    bool fail_armed;
    size_t fail_at;
} vitna_llama_t;

/**
 * Load the model in dir, with a key-value cache of seqs sequences (at least
 * 1), each of ctx positions (0 for the lesser of the model's maximum and
 * 4096). Returns false, with a reason in err, if the model cannot be read or
 * asks for something unsupported.
 */
bool vitna_llama_load(vitna_llama_t* m, const char* dir, size_t ctx, size_t seqs, char* err, size_t err_len);

void vitna_llama_free(vitna_llama_t* m);

/** Forget every position of every sequence, so each one's next token is at position 0. */
void vitna_llama_reset(vitna_llama_t* m);

/**
 * Keep the first n positions of sequence seq and forget the rest, so its next
 * token is at position n. A position's keys and values depend only on the
 * tokens up to it, so what is kept is exactly what feeding those n tokens
 * again would compute. n larger than the positions filled changes nothing.
 */
void vitna_llama_truncate(vitna_llama_t* m, size_t seq, size_t n);

/**
 * Run one token at sequence seq's next position (m->past[seq]), adding its
 * keys and values to that sequence's cache. Writes vocab logits to logits
 * unless it is NULL, which skips the output projection. Returns false if the
 * sequence's cache is full or the token is out of range, or, on a GPU, if
 * the device reports an error, which is then printed to stderr, or when a
 * test asked for this step to fail (vitna_llama_fail_step_once). A step that
 * returns false leaves m->past[seq] where it was.
 */
bool vitna_llama_step(vitna_llama_t* m, size_t seq, int32_t token, float* logits);

/**
 * Run count tokens of sequence seq at its positions m->past[seq] onwards, as
 * count calls to vitna_llama_step would, and return how many ran: count,
 * unless one could not, in which case the tokens before it ran and stay in
 * the cache, it did not, and m->past[seq] is left after the last that ran. A
 * token fails as a step does: out of range, past the cache, failed by a
 * test, or, on a GPU, a device error, which also leaves out every token of
 * the batch that had not yet been confirmed.
 *
 * If logits is not NULL and every token ran, it receives vocab logits for
 * each of the last rows positions (rows at most count), rows x vocab, in
 * position order: 1 for the next token, count for every position.
 *
 * On the CPU this is those steps one at a time. On a GPU the tokens run
 * together, as matrix-matrix products that read each weight once for many
 * tokens, and causal attention among them; the results are the same up to
 * float32 rounding.
 */
size_t vitna_llama_steps(vitna_llama_t* m, size_t seq, const int32_t* tokens, size_t count, float* logits, size_t rows);

/**
 * Run count tokens of sequence seq at its positions m->past[seq] onwards as
 * count calls to vitna_llama_step would, writing every one's logits, count x
 * vocab, and return how many ran, with vitna_llama_steps's rule for a token
 * that fails. Every value is the one those steps would give, bit for bit, so
 * a caller can check several drafted tokens in one call and reply exactly as
 * it would have one token at a time. It is vitna_llama_step_rows with every
 * row in seq.
 */
size_t vitna_llama_steps_exact(vitna_llama_t* m, size_t seq, const int32_t* tokens, size_t count, float* logits);

/** A token to run in sequence seq, at that sequence's next position. */
typedef struct {
    size_t seq;
    int32_t token;
} vitna_llama_row_t;

/**
 * Run n rows, each a token at its sequence's next position, as n calls to
 * vitna_llama_step would in order: rows of one sequence take its positions
 * one after another, and sequences share nothing. Writes every row's logits,
 * n x vocab, and sets ran[i] to whether row i ran. Every value is the one
 * those steps would give, bit for bit, however the rows mix sequences, so
 * several requests, each with drafted tokens or not, can run together and
 * each reply exactly as it would alone, a token at a time.
 *
 * A row fails as a step does, and a sequence stops at its first row that
 * fails: that row and its sequence's rows after it do not run. Rows of other
 * sequences run as they would without it. A device error on a GPU fails every
 * row of the pass it was in.
 *
 * On the CPU the rows run a step at a time. On a GPU up to
 * vitna_llama_exact_max rows run in each pass, which reads each weight once
 * for all of them. Returns how many rows ran.
 */
size_t vitna_llama_step_rows(vitna_llama_t* m, const vitna_llama_row_t* rows, size_t n, float* logits, bool* ran);

/** The most rows vitna_llama_step_rows runs in one pass: 1 on the CPU. */
size_t vitna_llama_exact_max(const vitna_llama_t* m);

/**
 * The fewest tokens each piece of a prompt must have for vitna_llama_steps
 * to run the pieces, one call after another, exactly as it runs them in one
 * call: the same keys, values and logits, bit for bit, however the prompt is
 * split. 1 on the CPU, where a prompt runs a step at a time. On a GPU a
 * piece of fewer tokens runs through other kernels, whose sums round
 * differently.
 */
size_t vitna_llama_prompt_piece_min(const vitna_llama_t* m);

/**
 * For tests: make the next step at position pos fail, once, in whichever
 * sequence reaches it first, as a step on the GPU does when the device
 * reports an error, so that what its callers do then can be tested on a CPU.
 * That step returns false before computing anything, on either device, and
 * says so on stderr. A later step at pos runs as usual.
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
