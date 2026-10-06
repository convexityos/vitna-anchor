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

/** A row of a pass: a token, its position, and the sequence whose cache it reads and writes. */
typedef struct {
    int32_t token;
    int32_t pos;
    int32_t seq;
} vitna_cuda_row_t;

/** Whether the CUDA runtime finds a device. Returns false, with the reason in err, if not. */
bool vitna_cuda_probe(char* err, size_t err_len);

/**
 * Upload m's weights to the first device, with the rotary cos and sin of
 * every position its cache holds (m->ctx rows of head_dim / 2), and allocate
 * the key-value cache, m->seqs sequences of m->ctx positions, and scratch
 * there. m stays as loaded; its weights are read once, here. Returns NULL,
 * with the reason in err, on failure.
 *
 * A mixture of experts' experts are not uploaded. The device keeps a cache
 * of them instead, expert_cache_bytes (or, for 0, what the device has free
 * once the rest is in place, less 512 MiB), and copies each into it when a
 * layer wants it: from the expert stream's slots if m->stream is set
 * (vitna_llama_stream_experts), from the mapped checkpoint otherwise. The
 * cache must hold twice the experts a token goes through. m must outlive
 * the returned model, and its expert stream too.
 */
struct vitna_cuda_model* vitna_cuda_create(const vitna_llama_t* m, const float* cos_tab, const float* sin_tab, size_t expert_cache_bytes,
                                           char* err, size_t err_len);

/*
 * A mixture of experts runs a token a layer at a time, its prompts too:
 * vitna_cuda_moe_route for layer 0, which also embeds the token, then
 * vitna_cuda_moe_experts for it, then each later layer the same way, and
 * vitna_cuda_moe_head when the token's logits are wanted. model.c routes in
 * between, from the router's logits, as it does on the CPU. Several rows run
 * the same way together (vitna_cuda_moe_rows_*, below). vitna_cuda_step,
 * vitna_cuda_steps and vitna_cuda_rows refuse such a model.
 */

/**
 * Run the token at position pos of sequence seq through layer up to its
 * router: for layer 0 its embedding first; then the attention, its output
 * added to the residual, and the router's logits over every expert, copied
 * to logits. Unless next is NULL, the next layer's norm and router score
 * the same residual into next, a guess at the experts that layer will
 * want. Returns false, with the reason in err, on a device error.
 */
bool vitna_cuda_moe_route(struct vitna_cuda_model* g, size_t seq, int32_t token, size_t pos, size_t layer, float* logits, float* next,
                          char* err, size_t err_len);

/**
 * Run the k experts ids of layer, distinct, their outputs scaled by weights
 * and added in the order given, and add the sum to the residual. Experts the
 * device's cache lacks are copied into it first. Unless guess is NULL, it
 * names k experts of the next layer, the likeliest first: the first half
 * start being copied behind these, as far as the cache takes them without
 * giving up these, and, when the experts are streamed, those the expert
 * stream lacks start being read from the drive. Returns false, with the
 * reason in err, if an expert cannot be read or the device reports an error.
 */
bool vitna_cuda_moe_experts(struct vitna_cuda_model* g, size_t layer, const int32_t* ids, const float* weights, size_t k,
                            const int32_t* guess, char* err, size_t err_len);

/** The final RMSNorm and the output projection of the token the layers ran, its vocab logits copied to logits. */
bool vitna_cuda_moe_head(struct vitna_cuda_model* g, float* logits, char* err, size_t err_len);

/** What the device's expert cache has done so far, in a sentence, for --timing; empty for a dense model. Returns buf. */
const char* vitna_cuda_moe_report(const struct vitna_cuda_model* g, char* buf, size_t len);

/**
 * From now on, a step of a mixture of experts shares the experts the device
 * lacks between the CPU, on threads threads, and copies to the device, while
 * the device runs those it holds (gate A8): the CPU takes as many as let a
 * layer finish soonest, by what an expert has cost each lately, the least
 * used first, and none when copying them all is sooner. The CPU computes
 * in the GPU's arithmetic (warp.h) and the outputs are added in the
 * device's order, so every value, the logits among them, is the bits it
 * would have been, whichever experts ran where.
 * Rows (vitna_cuda_moe_rows_*) copy in what they lack, as before. False,
 * with the reason in err, for a dense model or if the memory or the threads
 * cannot be had.
 */
bool vitna_cuda_cpu_experts(struct vitna_cuda_model* g, size_t threads, char* err, size_t err_len);

/*
 * Several rows of a mixture of experts at once, each a token at its position
 * in its sequence (vitna_cuda_row_t), a layer at a time as one token runs
 * above: vitna_cuda_moe_rows_route for layer 0, which also takes the rows
 * and embeds them, vitna_cuda_moe_rows_experts for it, and so on for every
 * layer, then vitna_cuda_moe_rows_head. Rows of one sequence hold
 * consecutive positions, in order; each row's arithmetic is the one its step
 * would do, in its step's order, so every value is the step's, bit for bit.
 * An expert runs once for all the rows routed to it, and is copied to the
 * device once a layer for them, where steps would copy it once a token.
 */

/** The most rows a pass takes: 0 for a dense model. */
size_t vitna_cuda_moe_rows_max(const struct vitna_cuda_model* g);

/**
 * Run the n rows through layer up to its router, as vitna_cuda_moe_route
 * runs one, and copy each row's router logits to logits, n x n_experts, and
 * unless next is NULL the next layer's guess to next. rows is read for layer
 * 0 alone; later layers take the same n rows. Returns false, with the reason
 * in err, on a device error.
 */
bool vitna_cuda_moe_rows_route(struct vitna_cuda_model* g, const vitna_cuda_row_t* rows, size_t n, size_t layer, float* logits, float* next,
                               char* err, size_t err_len);

/**
 * Run each row's k experts, ids n x k, distinct within a row and in the
 * order its outputs are added, weighted by weights n x k, as
 * vitna_cuda_moe_experts runs one row's, and add each row's sum to it.
 * Unless guess is NULL, it holds n x k experts of the next layer, each row's
 * likeliest first, which start being copied as for one row. Returns false,
 * with the reason in err, if an expert cannot be read or the device reports
 * an error.
 */
bool vitna_cuda_moe_rows_experts(struct vitna_cuda_model* g, size_t layer, size_t n, const int32_t* ids, const float* weights, size_t k,
                                 const int32_t* guess, char* err, size_t err_len);

/** The head of the pass's rows: row i's vocab logits copied to logits[i] where that is not NULL. */
bool vitna_cuda_moe_rows_head(struct vitna_cuda_model* g, size_t n, float* const* logits, char* err, size_t err_len);

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
 * that takes none; for a mixture of experts, the most a pass of
 * vitna_cuda_moe_rows_* takes.
 */
size_t vitna_cuda_exact_max(const struct vitna_cuda_model* g);

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

/** Wait until everything queued on the model's stream has run. */
void vitna_cuda_wait(struct vitna_cuda_model* g);

/**
 * After one of the calls above has failed: whether the device can run
 * anything more in this process. CUDA calls some errors sticky (an illegal
 * address, a kernel that faulted or ran too long, and others): any further
 * work returns them again, and only a new process can use the device. So this
 * clears the error the failed call left, asks for further work, a wait on the
 * model's stream, and returns true, with that work's error in err, when it
 * fails too.
 */
bool vitna_cuda_lost(struct vitna_cuda_model* g, char* err, size_t err_len);

void vitna_cuda_free(struct vitna_cuda_model* g);

/**
 * Widen the matrix m to float32 on the GPU, into out (rows x cols), for
 * checking what the kernels compute with: three times, through each way a
 * kernel reads weights (a chunk of eight, as the projections read them; four,
 * as a prompt's matrix product does; one, as the embedding does), and false,
 * with the reason in err, unless all three give the same bits. Copies m to
 * the device and back; device memory is kept between calls, for the largest
 * matrix seen so far.
 */
bool vitna_cuda_widen(const vitna_matrix_t* m, float* out, char* err, size_t err_len);

/**
 * Run one expert on the GPU as a step runs a mixture's experts, for
 * checking the CPU's copy of that arithmetic (warp.h): x through RMSNorm
 * with norm_w into xs, then the expert's gate and up projections and
 * activation into act (gate->rows floats), then its down projection into y
 * (gate->cols floats), by the kernels a step uses. Copies the expert to the
 * device and back.
 */
bool vitna_cuda_expert_check(const vitna_matrix_t* gate, const vitna_matrix_t* up, const vitna_matrix_t* down, const float* norm_w,
                             const float* x, float eps, float* xs, float* act, float* y, char* err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_MODEL_CUDA_H */
