/**
 * warp.h - Products summed in the GPU's order, on the CPU.
 *
 * model_cuda.cu's row_dot multiplies a row of weights by a vector a warp at
 * a time: lane l of 32 takes chunks l, l + 32, l + 64, ... of eight weights,
 * fusing each weight's product into its running sum in order, and the warp
 * then adds its lanes in warp_sum's tree, lane l with lane l + 16, then
 * with l + 8, l + 4, l + 2, l + 1. The functions here add the same products
 * in the same order, so a row's sum is the same float the GPU's is, bit for
 * bit, and an expert computed here gives the GPU's output (gate A8). Weights
 * are widened to float32 first, as the GPU widens them (quant.h).
 */

#ifndef VITNA_WARP_H
#define VITNA_WARP_H

#include <stdbool.h>
#include <stddef.h>
#include "pool.h"
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * y[r] = row r of w times x, for w of rows x cols in dtype, each sum in
 * row_dot's order. cols is a multiple of 8. AVX2 with FMA when the CPU has
 * them and cols is a multiple of 256, else the scalar loop. False, with
 * nothing computed, if its scratch cannot be had.
 */
bool vitna_warp_matvec(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y);

/** The scalar loop alone, a lane at a time: the reference the AVX2 path is tested against. */
bool vitna_warp_matvec_scalar(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y);

/** Which path vitna_warp_matvec takes for cols wide rows: "avx2+fma" or "scalar". */
const char* vitna_warp_path(size_t cols);

/**
 * An expert's output for the normalized input xs (hidden floats): its down
 * projection of silu(gate xs) * (up xs), into y (hidden floats). That is
 * what model_cuda.cu's experts_in_kernel and the row_dot of its experts'
 * down projection compute for one expert, before the expert's weight: every
 * sum in row_dot's order and the activation in exact.h's arithmetic. gate
 * and up are intermediate x hidden in gu_dtype, down hidden x intermediate
 * in down_dtype. act is scratch for 2 * intermediate floats, and holds the
 * activation in its first intermediate on return. False, with nothing
 * computed, if its scratch cannot be had.
 */
bool vitna_warp_expert(const void* gate, const void* up, const void* down, vitna_dtype_t gu_dtype, vitna_dtype_t down_dtype,
                       size_t hidden, size_t intermediate, const float* xs, float* y, float* act);

/** An expert for vitna_warp_experts: its matrices, and where its output goes (hidden floats). */
typedef struct {
    const void* gate;
    const void* up;
    const void* down;
    float* y;
} vitna_warp_expert_t;

/** The scratch a thread of vitna_warp_experts needs, in floats. */
size_t vitna_warp_scratch_floats(size_t hidden, size_t intermediate);

/**
 * The outputs of n experts of one layer for xs, each as vitna_warp_expert
 * computes it, bit for bit, with their rows shared among pool's threads.
 * act is n * 2 * intermediate floats; scratch is vitna_warp_scratch_floats
 * for each of the pool's threads, so nothing is allocated here.
 */
void vitna_warp_experts(vitna_pool_t* pool, const vitna_warp_expert_t* experts, size_t n, vitna_dtype_t gu_dtype, vitna_dtype_t down_dtype,
                        size_t hidden, size_t intermediate, const float* xs, float* act, float* scratch);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_WARP_H */
