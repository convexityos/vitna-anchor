/**
 * ops.c - Float32 operations for the forward pass.
 */

#include "ops.h"
#include "kernels.h"
#include <math.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
  #define VITNA_OPS_X86 1
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define VITNA_OPS_NEON 1
  #include <arm_neon.h>
#endif

#if defined(VITNA_OPS_X86) && (defined(__GNUC__) || defined(__clang__))
  #define VITNA_OPS_AVX2 __attribute__((target("avx2,fma")))
#else
  #define VITNA_OPS_AVX2
#endif

float vitna_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

float vitna_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Subnormal: normalize the mantissa. */
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

void vitna_to_f32(const void* src, vitna_dtype_t dtype, float* dst, size_t n) {
    if (dtype == VITNA_DTYPE_F32) {
        memcpy(dst, src, n * sizeof(float));
    } else if (dtype == VITNA_DTYPE_BF16) {
        const uint16_t* s = (const uint16_t*)src;
        for (size_t i = 0; i < n; i++) dst[i] = vitna_bf16_to_f32(s[i]);
    } else if (dtype == VITNA_DTYPE_F16) {
        const uint16_t* s = (const uint16_t*)src;
        for (size_t i = 0; i < n; i++) dst[i] = vitna_f16_to_f32(s[i]);
    }
}

static float load_w(const void* w, vitna_dtype_t dtype, size_t i) {
    if (dtype == VITNA_DTYPE_F32) return ((const float*)w)[i];
    if (dtype == VITNA_DTYPE_BF16) return vitna_bf16_to_f32(((const uint16_t*)w)[i]);
    return vitna_f16_to_f32(((const uint16_t*)w)[i]);
}

void vitna_matvec_scalar(const void* w, vitna_dtype_t dtype, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        size_t base = r * cols;
        for (size_t c = 0; c < cols; c++) sum += load_w(w, dtype, base + c) * x[c];
        y[r] = sum;
    }
}

#if defined(VITNA_OPS_NEON)
static void matvec_bf16_neon(const uint16_t* w, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        const uint16_t* row = w + r * cols;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        size_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            uint16x8_t lo = vld1q_u16(row + c);
            uint16x8_t hi = vld1q_u16(row + c + 8);
            float32x4_t w0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(lo)), 16));
            float32x4_t w1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(lo)), 16));
            float32x4_t w2 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(hi)), 16));
            float32x4_t w3 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(hi)), 16));
            a0 = vfmaq_f32(a0, w0, vld1q_f32(x + c));
            a1 = vfmaq_f32(a1, w1, vld1q_f32(x + c + 4));
            a2 = vfmaq_f32(a2, w2, vld1q_f32(x + c + 8));
            a3 = vfmaq_f32(a3, w3, vld1q_f32(x + c + 12));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; c < cols; c++) sum += vitna_bf16_to_f32(row[c]) * x[c];
        y[r] = sum;
    }
}

static void matvec_f32_neon(const float* w, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        const float* row = w + r * cols;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        size_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(row + c), vld1q_f32(x + c));
            a1 = vfmaq_f32(a1, vld1q_f32(row + c + 4), vld1q_f32(x + c + 4));
            a2 = vfmaq_f32(a2, vld1q_f32(row + c + 8), vld1q_f32(x + c + 8));
            a3 = vfmaq_f32(a3, vld1q_f32(row + c + 12), vld1q_f32(x + c + 12));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; c < cols; c++) sum += row[c] * x[c];
        y[r] = sum;
    }
}
#endif

#if defined(VITNA_OPS_X86)
VITNA_OPS_AVX2
static float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

VITNA_OPS_AVX2
static void matvec_bf16_avx2(const uint16_t* w, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        const uint16_t* row = w + r * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        size_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m128i lo = _mm_loadu_si128((const __m128i*)(row + c));
            __m128i hi = _mm_loadu_si128((const __m128i*)(row + c + 8));
            __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16));
            __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16));
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + c), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + c + 8), a1);
        }
        float sum = hsum256(_mm256_add_ps(a0, a1));
        for (; c < cols; c++) sum += vitna_bf16_to_f32(row[c]) * x[c];
        y[r] = sum;
    }
}

VITNA_OPS_AVX2
static void matvec_f32_avx2(const float* w, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        const float* row = w + r * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        size_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(x + c), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8), _mm256_loadu_ps(x + c + 8), a1);
        }
        float sum = hsum256(_mm256_add_ps(a0, a1));
        for (; c < cols; c++) sum += row[c] * x[c];
        y[r] = sum;
    }
}
#endif

static int simd_path(void) {
    static int path = -1; /* 0 scalar, 1 avx2+fma, 2 neon */
    if (path < 0) {
        vitna_simd_capabilities_t caps = vitna_detect_simd_capabilities();
#if defined(VITNA_OPS_X86)
        path = (caps.has_avx2 && caps.has_fma) ? 1 : 0;
#elif defined(VITNA_OPS_NEON)
        path = caps.has_neon ? 2 : 0;
#else
        (void)caps;
        path = 0;
#endif
    }
    return path;
}

const char* vitna_matvec_path(void) {
    switch (simd_path()) {
        case 1: return "avx2+fma";
        case 2: return "neon";
        default: return "scalar";
    }
}

void vitna_matvec(const void* w, vitna_dtype_t dtype, const float* x, float* y, size_t rows, size_t cols) {
    int path = simd_path();
#if defined(VITNA_OPS_X86)
    if (path == 1) {
        if (dtype == VITNA_DTYPE_BF16) { matvec_bf16_avx2((const uint16_t*)w, x, y, rows, cols); return; }
        if (dtype == VITNA_DTYPE_F32) { matvec_f32_avx2((const float*)w, x, y, rows, cols); return; }
    }
#elif defined(VITNA_OPS_NEON)
    if (path == 2) {
        if (dtype == VITNA_DTYPE_BF16) { matvec_bf16_neon((const uint16_t*)w, x, y, rows, cols); return; }
        if (dtype == VITNA_DTYPE_F32) { matvec_f32_neon((const float*)w, x, y, rows, cols); return; }
    }
#endif
    (void)path;
    vitna_matvec_scalar(w, dtype, x, y, rows, cols);
}

void vitna_rope_half(float* v, size_t head_dim, const float* cos_t, const float* sin_t) {
    size_t half = head_dim / 2;
    for (size_t j = 0; j < half; j++) {
        float a = v[j];
        float b = v[j + half];
        v[j] = a * cos_t[j] - b * sin_t[j];
        v[j + half] = b * cos_t[j] + a * sin_t[j];
    }
}

void vitna_silu_mul(const float* gate, const float* up, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
