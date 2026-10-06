/**
 * ops.h - Float32 operations for the forward pass.
 *
 * Weights are read in their stored dtype (F32, BF16 or F16) and widened to
 * float32 as they are used, so a BF16 checkpoint is computed exactly as if
 * it had been converted to float32 first. Arithmetic is float32 throughout.
 */

#ifndef VITNA_OPS_H
#define VITNA_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

float vitna_bf16_to_f32(uint16_t h);
float vitna_f16_to_f32(uint16_t h);

/** Widen n values of dtype to float32. dtype is F32, BF16 or F16, or a block
    format (quant.h), for which n is whole blocks starting at a block. */
void vitna_to_f32(const void* src, vitna_dtype_t dtype, float* dst, size_t n);

/**
 * y = W x, for W of rows x cols in row-major order, stored as dtype. Uses
 * NEON or AVX2 with FMA when the CPU has them, else the scalar loop. A block
 * format's rows are widened a block at a time, then multiplied as float32
 * rows are (AVX2), or by the scalar loop (elsewhere).
 */
void vitna_matvec(const void* w, vitna_dtype_t dtype, const float* x, float* y, size_t rows, size_t cols);

/** The scalar loop alone, one row at a time: the reference the SIMD paths are tested against. */
void vitna_matvec_scalar(const void* w, vitna_dtype_t dtype, const float* x, float* y, size_t rows, size_t cols);

/** Which matvec path runs on this CPU: "avx2+fma", "neon" or "scalar". */
const char* vitna_matvec_path(void);

/**
 * Rotary position embedding in the half-split form Hugging Face's Llama uses
 * (rotate_half): element j pairs with element j + head_dim / 2. cos_t and
 * sin_t hold head_dim / 2 values for the position.
 */
void vitna_rope_half(float* v, size_t head_dim, const float* cos_t, const float* sin_t);

/** out[i] = silu(gate[i]) * up[i], where silu(g) = g / (1 + exp(-g)). */
void vitna_silu_mul(const float* gate, const float* up, float* out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_OPS_H */
