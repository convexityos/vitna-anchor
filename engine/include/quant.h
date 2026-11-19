/**
 * quant.h - ggml's block formats (Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) widened to float32.
 *
 * A GGUF file holds a quantized tensor row by row, each row whole blocks: a
 * block is a scale or two in float16 and small integers, and each weight is
 * an integer times a scale. The engine computes in float32, so a row is
 * widened block by block into floats and then multiplied as any other.
 *
 * The widening is the arithmetic of gguf-py's dequantize, which is
 * llama.cpp's own: in float32, each product rounded on its own, never fused
 * with the subtraction after it. So a tensor widened here is the tensor the
 * reference was recorded on, bit for bit, and the CUDA path widens to the
 * same bits (model_cuda.cu), whichever device runs a row.
 *
 * Q8_0, 34 bytes for 32 weights: d (float16), then 32 int8 q.
 *     w = q * d
 * Q2_K, 84 bytes for 256 weights in 16 groups of 16: 16 bytes holding a 4-bit
 * scale (low nibble) and 4-bit min (high nibble) per group, 64 bytes of 2-bit
 * q in two 128-weight halves (4 shifts of 2 bits over 32 bytes each), then d
 * and dmin (float16).
 *     w = (d * scale) * q - (dmin * min)
 * Q3_K, 110 bytes for 256 weights in 16 groups of 16: 32 bytes of high bit
 * hmask, 64 bytes of low 2-bit q in two 128-weight halves, 12 bytes packing
 * sixteen 6-bit scales (bias -32), then d (float16).
 *     w = (d * (scale - 32)) * (q - 4)
 * Q4_K, 144 bytes for 256 weights in 8 groups of 32: d and dmin (float16),
 * 12 bytes packing a 6-bit scale and a 6-bit min for each group, then 128
 * bytes of 4-bit q. Bytes 32j to 32j + 31 hold group 2j in their low halves
 * and group 2j + 1 in their high halves.
 *     w = (d * scale) * q - (dmin * min)
 * Q5_K, 176 bytes for 256 weights in 8 groups of 32: d and dmin (float16),
 * 12 bytes packing 6-bit scale and min as in Q4_K, 32 bytes of high bits qh
 * (bit g of byte l for group g, weight l), then 128 bytes of low 4-bit q.
 *     w = (d * scale) * q - (dmin * min)
 * Q6_K, 210 bytes for 256 weights in 16 groups of 16: 128 bytes of the low
 * four bits of each q, 64 bytes of the high two, 16 int8 scales, then d
 * (float16). Each half of 128 weights takes 64 low bytes and 32 high ones.
 *     w = (d * scale) * (q - 32)
 */

#ifndef VITNA_QUANT_H
#define VITNA_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Widen n weights of a block format, starting at a block boundary, to
    float32. n is whole blocks. */
void vitna_dequant(const void* src, vitna_dtype_t dtype, float* dst, size_t n);

/** One block of each format. */
void vitna_dequant_q8_0(const uint8_t* block, float out[32]);
void vitna_dequant_q2_k(const uint8_t* block, float out[256]);
void vitna_dequant_q3_k(const uint8_t* block, float out[256]);
void vitna_dequant_q4_k(const uint8_t* block, float out[256]);
void vitna_dequant_q5_k(const uint8_t* block, float out[256]);
void vitna_dequant_q6_k(const uint8_t* block, float out[256]);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_QUANT_H */
