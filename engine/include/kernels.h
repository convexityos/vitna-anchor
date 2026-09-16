/**
 * kernels.h - High-performance inference kernels (quantization, matmul, activations).
 */

#ifndef VITNA_KERNELS_H
#define VITNA_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VITNA_QUANT_INT8 = 8,
    VITNA_QUANT_INT4 = 4,
    VITNA_QUANT_INT3 = 3,
    VITNA_QUANT_INT2 = 2
} vitna_quant_type_t;

/**
 * Root Mean Square Normalization (RMSNorm).
 * y = (x / sqrt(mean(x^2) + eps)) * weight
 */
void vitna_rmsnorm(
    const float* x,
    const float* weight,
    float* y,
    size_t dim,
    float eps
);

/**
 * SwiGLU activation function.
 * gate_up has shape [2 * hidden_dim]:
 * gate = gate_up[0..hidden_dim-1]
 * up   = gate_up[hidden_dim..2*hidden_dim-1]
 * out  = (gate / (1.0 + exp(-gate))) * up
 */
void vitna_swiglu(
    const float* gate_up,
    float* out,
    size_t hidden_dim
);

/**
 * Dequantize symmetric int4 packed weights into float.
 * Each byte holds two 4-bit values: low nibble [0..3], high nibble [4..7].
 * out[i] = ((int8_t)nibble - 8) * scale
 */
void vitna_dequant_int4(
    const uint8_t* packed_weights,
    const float* scales,
    size_t group_size,
    float* out,
    size_t count
);

/**
 * Quantized Matrix-Vector multiplication (int4 weights x float input vector).
 * W has shape [rows, cols], packed as [rows, cols / 2] bytes.
 * y = W * x
 */
void vitna_gemv_int4(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

/**
 * High-precision Matrix-Vector multiplication for hot anchor experts (int8).
 * W has shape [rows, cols], stored as [rows, cols] int8 bytes.
 */
void vitna_gemv_int8(
    const int8_t* w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

/**
 * Ultra-compact 3-bit Matrix-Vector multiplication for cold domain experts.
 * Packs 8 weights into 3 bytes (24 bits total), symmetric [-4, 3].
 */
void vitna_gemv_int3(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

/**
 * Ultra-compact 2-bit Matrix-Vector multiplication for cold long-tail experts.
 * Packs 4 weights per byte, symmetric [-2, 1].
 */
void vitna_gemv_int2(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
);

/**
 * Hardware SIMD capabilities detection.
 */
typedef struct {
    bool has_avx2;
    bool has_avx512;
    bool has_neon;
    bool has_fma;
} vitna_simd_capabilities_t;

vitna_simd_capabilities_t vitna_detect_simd_capabilities(void);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_KERNELS_H */
