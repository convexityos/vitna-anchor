/**
 * kernels.c - Optimized CPU inference kernels implementation with SIMD acceleration.
 */

#include "kernels.h"
#include <math.h>
#include <string.h>

#if defined(_MSC_VER)
  #include <intrin.h>
  #include <immintrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <cpuid.h>
  #include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
  #include <arm_neon.h>
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
  #define VITNA_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
  #define VITNA_TARGET_AVX2
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
VITNA_TARGET_AVX2
static inline float vitna_reduce_sum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(s, s);
    s = _mm_add_ps(s, shuf);
    shuf = _mm_shuffle_ps(s, s, 1);
    s = _mm_add_ss(s, shuf);
    return _mm_cvtss_f32(s);
}
#endif

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
static inline float vitna_reduce_sum_neon(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
#endif
}
#endif

/**
 * Hardware SIMD capabilities detection.
 */
vitna_simd_capabilities_t vitna_detect_simd_capabilities(void) {
    vitna_simd_capabilities_t caps = {false, false, false, false};
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    caps.has_fma = (cpuInfo[2] & (1 << 12)) != 0;

    __cpuidex(cpuInfo, 7, 0);
    caps.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    caps.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && (defined(__GNUC__) || defined(__clang__))
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps.has_fma = (ecx & (1 << 12)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps.has_avx2 = (ebx & (1 << 5)) != 0;
        caps.has_avx512 = (ebx & (1 << 16)) != 0;
    }
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    caps.has_neon = true;
    caps.has_fma = true;
#endif
    return caps;
}

static bool s_simd_probed = false;
static vitna_simd_capabilities_t s_simd_caps;

static inline vitna_simd_capabilities_t vitna_get_simd_caps(void) {
    if (!s_simd_probed) {
        s_simd_caps = vitna_detect_simd_capabilities();
        s_simd_probed = true;
    }
    return s_simd_caps;
}

/**
 * Root Mean Square Normalization (RMSNorm) - Scalar fallback.
 */
static void vitna_rmsnorm_scalar(
    const float* x,
    const float* weight,
    float* y,
    size_t dim,
    float eps
) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }

    float inv_rms = 1.0f / sqrtf((sum_sq / (float)dim) + eps);

    for (size_t i = 0; i < dim; i++) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
VITNA_TARGET_AVX2
static void vitna_rmsnorm_avx2(
    const float* x,
    const float* weight,
    float* y,
    size_t dim,
    float eps
) {
    size_t i = 0;
    __m256 sum256_0 = _mm256_setzero_ps();
    __m256 sum256_1 = _mm256_setzero_ps();

    for (; i + 16 <= dim; i += 16) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        sum256_0 = _mm256_fmadd_ps(v0, v0, sum256_0);
        sum256_1 = _mm256_fmadd_ps(v1, v1, sum256_1);
    }

    sum256_0 = _mm256_add_ps(sum256_0, sum256_1);

    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum256_0 = _mm256_fmadd_ps(v, v, sum256_0);
    }

    float sum_sq = vitna_reduce_sum_m256(sum256_0);

    for (; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }

    float inv_rms = 1.0f / sqrtf((sum_sq / (float)dim) + eps);
    __m256 vinv = _mm256_set1_ps(inv_rms);

    i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vy = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
        _mm256_storeu_ps(y + i, vy);
    }

    for (; i < dim; i++) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}
#endif

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
static void vitna_rmsnorm_neon(
    const float* x,
    const float* weight,
    float* y,
    size_t dim,
    float eps
) {
    size_t i = 0;
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    for (; i + 8 <= dim; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        sum0 = vfmaq_f32(sum0, v0, v0);
        sum1 = vfmaq_f32(sum1, v1, v1);
    }

    float sum_sq = vitna_reduce_sum_neon(vaddq_f32(sum0, sum1));

    for (; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }

    float inv_rms = 1.0f / sqrtf((sum_sq / (float)dim) + eps);
    float32x4_t vinv = vdupq_n_f32(inv_rms);

    i = 0;
    for (; i + 8 <= dim; i += 8) {
        float32x4_t vx0 = vld1q_f32(x + i);
        float32x4_t vx1 = vld1q_f32(x + i + 4);
        float32x4_t vw0 = vld1q_f32(weight + i);
        float32x4_t vw1 = vld1q_f32(weight + i + 4);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(vx0, vinv), vw0));
        vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(vx1, vinv), vw1));
    }

    for (; i < dim; i++) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}
#endif

void vitna_rmsnorm(
    const float* x,
    const float* weight,
    float* y,
    size_t dim,
    float eps
) {
    if (!x || !weight || !y || dim == 0) return;

    vitna_simd_capabilities_t caps = vitna_get_simd_caps();
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (caps.has_avx2 && caps.has_fma) {
        vitna_rmsnorm_avx2(x, weight, y, dim, eps);
        return;
    }
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    if (caps.has_neon) {
        vitna_rmsnorm_neon(x, weight, y, dim, eps);
        return;
    }
#endif

    vitna_rmsnorm_scalar(x, weight, y, dim, eps);
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

static void vitna_gemv_int4_scalar(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
VITNA_TARGET_AVX2
static void vitna_gemv_int4_avx2(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    size_t bytes_per_row = cols / 2;
    size_t scales_per_row = cols / group_size;
    size_t bytes_per_group = group_size / 2;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_w = packed_w + (r * bytes_per_row);
        const float* row_scales = scales + (r * scales_per_row);
        float row_sum = 0.0f;

        for (size_t g = 0; g < scales_per_row; g++) {
            float scale = row_scales[g];
            const uint8_t* gw = row_w + (g * bytes_per_group);
            const float* gx = x + (g * group_size);

            __m256 vacc = _mm256_setzero_ps();
            size_t c = 0;

            for (; c + 8 <= group_size; c += 8) {
                size_t b_offset = c / 2;
                float w_buf[8];
                for (int j = 0; j < 4; j++) {
                    uint8_t b = gw[b_offset + j];
                    w_buf[j * 2]     = (float)((int8_t)(b & 0x0F) - 8);
                    w_buf[j * 2 + 1] = (float)((int8_t)((b >> 4) & 0x0F) - 8);
                }
                __m256 vw = _mm256_loadu_ps(w_buf);
                __m256 vx = _mm256_loadu_ps(gx + c);
                vacc = _mm256_fmadd_ps(vw, vx, vacc);
            }

            float group_acc = vitna_reduce_sum_m256(vacc);

            for (; c < group_size; c += 2) {
                size_t b_idx = c / 2;
                uint8_t byte = gw[b_idx];
                int8_t v0 = (int8_t)(byte & 0x0F) - 8;
                int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
                group_acc += (float)v0 * gx[c];
                if (c + 1 < group_size) {
                    group_acc += (float)v1 * gx[c + 1];
                }
            }

            row_sum += group_acc * scale;
        }
        y[r] = row_sum;
    }
}
#endif

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
static void vitna_gemv_int4_neon(
    const uint8_t* packed_w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    size_t bytes_per_row = cols / 2;
    size_t scales_per_row = cols / group_size;
    size_t bytes_per_group = group_size / 2;

    for (size_t r = 0; r < rows; r++) {
        const uint8_t* row_w = packed_w + (r * bytes_per_row);
        const float* row_scales = scales + (r * scales_per_row);
        float row_sum = 0.0f;

        for (size_t g = 0; g < scales_per_row; g++) {
            float scale = row_scales[g];
            const uint8_t* gw = row_w + (g * bytes_per_group);
            const float* gx = x + (g * group_size);

            float32x4_t vacc = vdupq_n_f32(0.0f);
            size_t c = 0;

            for (; c + 4 <= group_size; c += 4) {
                size_t b_offset = c / 2;
                uint8_t b0 = gw[b_offset];
                uint8_t b1 = gw[b_offset + 1];
                float w_buf[4];
                w_buf[0] = (float)((int8_t)(b0 & 0x0F) - 8);
                w_buf[1] = (float)((int8_t)((b0 >> 4) & 0x0F) - 8);
                w_buf[2] = (float)((int8_t)(b1 & 0x0F) - 8);
                w_buf[3] = (float)((int8_t)((b1 >> 4) & 0x0F) - 8);

                float32x4_t vw = vld1q_f32(w_buf);
                float32x4_t vx = vld1q_f32(gx + c);
                vacc = vfmaq_f32(vacc, vw, vx);
            }

            float group_acc = vitna_reduce_sum_neon(vacc);

            for (; c < group_size; c += 2) {
                size_t b_idx = c / 2;
                uint8_t byte = gw[b_idx];
                int8_t v0 = (int8_t)(byte & 0x0F) - 8;
                int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
                group_acc += (float)v0 * gx[c];
                if (c + 1 < group_size) {
                    group_acc += (float)v1 * gx[c + 1];
                }
            }

            row_sum += group_acc * scale;
        }
        y[r] = row_sum;
    }
}
#endif

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

    if ((cols % group_size == 0) && (group_size % 8 == 0)) {
        vitna_simd_capabilities_t caps = vitna_get_simd_caps();
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        if (caps.has_avx2 && caps.has_fma) {
            vitna_gemv_int4_avx2(packed_w, scales, group_size, x, y, rows, cols);
            return;
        }
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
        if (caps.has_neon) {
            vitna_gemv_int4_neon(packed_w, scales, group_size, x, y, rows, cols);
            return;
        }
#endif
    }

    vitna_gemv_int4_scalar(packed_w, scales, group_size, x, y, rows, cols);
}

static void vitna_gemv_int8_scalar(
    const int8_t* w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
VITNA_TARGET_AVX2
static void vitna_gemv_int8_avx2(
    const int8_t* w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    size_t scales_per_row = cols / group_size;

    for (size_t r = 0; r < rows; r++) {
        const int8_t* row_w = w + (r * cols);
        const float* row_scales = scales + (r * scales_per_row);
        float row_sum = 0.0f;

        for (size_t g = 0; g < scales_per_row; g++) {
            float scale = row_scales[g];
            const int8_t* gw = row_w + (g * group_size);
            const float* gx = x + (g * group_size);

            __m256 vacc = _mm256_setzero_ps();
            size_t c = 0;

            for (; c + 8 <= group_size; c += 8) {
                __m128i v_i8 = _mm_loadl_epi64((const __m128i*)(gw + c));
                __m256i v_i32 = _mm256_cvtepi8_epi32(v_i8);
                __m256 v_f32 = _mm256_cvtepi32_ps(v_i32);
                __m256 vx = _mm256_loadu_ps(gx + c);
                vacc = _mm256_fmadd_ps(v_f32, vx, vacc);
            }

            float group_acc = vitna_reduce_sum_m256(vacc);

            for (; c < group_size; c++) {
                group_acc += (float)gw[c] * gx[c];
            }

            row_sum += group_acc * scale;
        }
        y[r] = row_sum;
    }
}
#endif

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
static void vitna_gemv_int8_neon(
    const int8_t* w,
    const float* scales,
    size_t group_size,
    const float* x,
    float* y,
    size_t rows,
    size_t cols
) {
    size_t scales_per_row = cols / group_size;

    for (size_t r = 0; r < rows; r++) {
        const int8_t* row_w = w + (r * cols);
        const float* row_scales = scales + (r * scales_per_row);
        float row_sum = 0.0f;

        for (size_t g = 0; g < scales_per_row; g++) {
            float scale = row_scales[g];
            const int8_t* gw = row_w + (g * group_size);
            const float* gx = x + (g * group_size);

            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);
            size_t c = 0;

            for (; c + 8 <= group_size; c += 8) {
                int8x8_t v_i8 = vld1_s8(gw + c);
                int16x8_t v_i16 = vmovl_s8(v_i8);
                int32x4_t v_i32_lo = vmovl_s16(vget_low_s16(v_i16));
                int32x4_t v_i32_hi = vmovl_s16(vget_high_s16(v_i16));
                float32x4_t v_f0 = vcvtq_f32_s32(v_i32_lo);
                float32x4_t v_f1 = vcvtq_f32_s32(v_i32_hi);
                float32x4_t vx0 = vld1q_f32(gx + c);
                float32x4_t vx1 = vld1q_f32(gx + c + 4);
                vacc0 = vfmaq_f32(vacc0, v_f0, vx0);
                vacc1 = vfmaq_f32(vacc1, v_f1, vx1);
            }

            float group_acc = vitna_reduce_sum_neon(vaddq_f32(vacc0, vacc1));

            for (; c < group_size; c++) {
                group_acc += (float)gw[c] * gx[c];
            }

            row_sum += group_acc * scale;
        }
        y[r] = row_sum;
    }
}
#endif

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

    if ((cols % group_size == 0) && (group_size % 8 == 0)) {
        vitna_simd_capabilities_t caps = vitna_get_simd_caps();
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        if (caps.has_avx2 && caps.has_fma) {
            vitna_gemv_int8_avx2(w, scales, group_size, x, y, rows, cols);
            return;
        }
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
        if (caps.has_neon) {
            vitna_gemv_int8_neon(w, scales, group_size, x, y, rows, cols);
            return;
        }
#endif
    }

    vitna_gemv_int8_scalar(w, scales, group_size, x, y, rows, cols);
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
