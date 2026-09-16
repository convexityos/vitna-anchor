/**
 * kernels.c - Optimized CPU inference kernels implementation.
 */

#include "kernels.h"
#include <math.h>
#include <string.h>

void vitna_rmsnorm(
    const float* x,
    const float* weight,
    float* y,
    size_t dim,
    float eps
) {
    if (!x || !weight || !y || dim == 0) return;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }

    float inv_rms = 1.0f / sqrtf((sum_sq / (float)dim) + eps);

    for (size_t i = 0; i < dim; i++) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

void vitna_swiglu(
    const float* gate_up,
    float* out,
    size_t hidden_dim
) {
    if (!gate_up || !out || hidden_dim == 0) return;

    const float* gate = gate_up;
    const float* up = gate_up + hidden_dim;

    for (size_t i = 0; i < hidden_dim; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

void vitna_dequant_int4(
    const uint8_t* packed_weights,
    const float* scales,
    size_t group_size,
    float* out,
    size_t count
) {
    if (!packed_weights || !scales || !out || count == 0) return;
    if (group_size == 0) group_size = 128;

    size_t num_bytes = count / 2;
    for (size_t b = 0; b < num_bytes; b++) {
        uint8_t byte = packed_weights[b];
        size_t idx0 = b * 2;
        size_t idx1 = idx0 + 1;

        float scale0 = scales[idx0 / group_size];
        float scale1 = scales[idx1 / group_size];

        int8_t v0 = (int8_t)(byte & 0x0F) - 8;
        int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;

        out[idx0] = (float)v0 * scale0;
        out[idx1] = (float)v1 * scale1;
    }
}

void vitna_gemv_int4(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!packed_w || !scales || !x || !y || rows == 0 || cols == 0) return;
    if (group_size == 0) group_size = 128;

    size_t bytes_per_row = cols / 2;
    size_t scales_per_row = cols / group_size;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_w = packed_w + (r * bytes_per_row);
        const float* row_scales = scales + (r * scales_per_row);

        float sum = 0.0f;
        for (size_t b = 0; b < bytes_per_row; b++) {
            uint8_t byte = row_w[b];
            size_t c0 = b * 2;
            size_t c1 = c0 + 1;

            float scale0 = row_scales[c0 / group_size];
            float scale1 = row_scales[c1 / group_size];

            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;

            sum += ((float)v0 * scale0) * x[c0];
            sum += ((float)v1 * scale1) * x[c1];
        }
        y[r] = sum;
    }
}

void vitna_gemv_int8(
    const int8_t* w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!w || !scales || !x || !y || rows == 0 || cols == 0) return;
    if (group_size == 0) group_size = 128;

    size_t scales_per_row = cols / group_size;

    for (size_t r = 0; r < rows; r++) {
        const int8_t* row_w = w + (r * cols);
        const float* row_scales = scales + (r * scales_per_row);

        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float scale = row_scales[c / group_size];
            sum += ((float)row_w[c] * scale) * x[c];
        }
        y[r] = sum;
    }
}

void vitna_gemv_int3(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!packed_w || !scales || !x || !y || rows == 0 || cols == 0) return;
    if (group_size == 0) group_size = 128;

    /* 8 columns per 3 bytes */
    size_t chunks = cols / 8;
    size_t bytes_per_row = chunks * 3;
    size_t scales_per_row = cols / group_size;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_w = packed_w + (r * bytes_per_row);
        const float* row_scales = scales + (r * scales_per_row);

        float sum = 0.0f;
        for (size_t chunk = 0; chunk < chunks; chunk++) {
            size_t b_idx = chunk * 3;
            uint8_t b0 = row_w[b_idx];
            uint8_t b1 = row_w[b_idx + 1];
            uint8_t b2 = row_w[b_idx + 2];

            int8_t w[8];
            w[0] = (int8_t)(b0 & 0x07) - 4;
            w[1] = (int8_t)((b0 >> 3) & 0x07) - 4;
            w[2] = (int8_t)(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)) - 4;
            w[3] = (int8_t)((b1 >> 1) & 0x07) - 4;
            w[4] = (int8_t)((b1 >> 4) & 0x07) - 4;
            w[5] = (int8_t)(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)) - 4;
            w[6] = (int8_t)((b2 >> 2) & 0x07) - 4;
            w[7] = (int8_t)((b2 >> 5) & 0x07) - 4;

            size_t c_base = chunk * 8;
            for (size_t k = 0; k < 8; k++) {
                size_t c = c_base + k;
                float scale = row_scales[c / group_size];
                sum += ((float)w[k] * scale) * x[c];
            }
        }
        y[r] = sum;
    }
}

void vitna_gemv_int2(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    if (!packed_w || !scales || !x || !y || rows == 0 || cols == 0) return;
    if (group_size == 0) group_size = 128;

    /* 4 columns per byte */
    size_t bytes_per_row = cols / 4;
    size_t scales_per_row = cols / group_size;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_w = packed_w + (r * bytes_per_row);
        const float* row_scales = scales + (r * scales_per_row);

        float sum = 0.0f;
        for (size_t b = 0; b < bytes_per_row; b++) {
            uint8_t byte = row_w[b];
            size_t c_base = b * 4;

            int8_t w[4];
            w[0] = (int8_t)(byte & 0x03) - 2;
            w[1] = (int8_t)((byte >> 2) & 0x03) - 2;
            w[2] = (int8_t)((byte >> 4) & 0x03) - 2;
            w[3] = (int8_t)((byte >> 6) & 0x03) - 2;

            for (size_t k = 0; k < 4; k++) {
                size_t c = c_base + k;
                float scale = row_scales[c / group_size];
                sum += ((float)w[k] * scale) * x[c];
            }
        }
        y[r] = sum;
    }
}
