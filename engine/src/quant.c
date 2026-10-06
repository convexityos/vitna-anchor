/**
 * quant.c - ggml's block formats widened to float32 (quant.h).
 *
 * Every product is rounded before anything is added to it or taken from it,
 * as numpy rounds it in gguf-py. A compiler may otherwise fuse (d * q) - m
 * into one multiply-add, which rounds once and gives other bits: clang does
 * within an expression by default on ARM, and GCC across statements. The
 * pragmas below forbid it for this file, and each product is also its own
 * statement, which no contraction the C standard allows can reach across.
 */

#if defined(__clang__)
  #pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
  #pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
  #pragma fp_contract(off)
#endif

#include "quant.h"
#include "kernels.h"
#include "ops.h"
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
  #define VITNA_QUANT_X86 1
  #include <immintrin.h>
#endif

#if defined(VITNA_QUANT_X86) && (defined(__GNUC__) || defined(__clang__))
  #define VITNA_QUANT_AVX2 __attribute__((target("avx2")))
#else
  #define VITNA_QUANT_AVX2
#endif

static uint16_t u16le(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

void vitna_dequant_q8_0(const uint8_t* block, float out[32]) {
    const float d = vitna_f16_to_f32(u16le(block));
    const int8_t* q = (const int8_t*)(block + 2);
    for (int i = 0; i < 32; i++) out[i] = (float)q[i] * d;
}

/* Group j's 6-bit scale and min from the 12 packed bytes, as ggml's
   get_scale_min_k4: groups 0 to 3 in the low six bits of bytes j and j + 4,
   groups 4 to 7 in the halves of byte j + 4 with two high bits borrowed
   from bytes j - 4 and j. */
static void scale_min_k4(int j, const uint8_t* s, uint8_t* scale, uint8_t* min) {
    if (j < 4) {
        *scale = s[j] & 63;
        *min = s[j + 4] & 63;
    } else {
        *scale = (uint8_t)((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        *min = (uint8_t)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

void vitna_dequant_q4_k(const uint8_t* block, float out[256]) {
    const float d = vitna_f16_to_f32(u16le(block));
    const float dmin = vitna_f16_to_f32(u16le(block + 2));
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mn;
        scale_min_k4(g, scales, &sc, &mn);
        const float d1 = d * (float)sc;
        const float m1 = dmin * (float)mn;
        const uint8_t* q = qs + 32 * (g / 2);
        const int shift = (g & 1) ? 4 : 0;
        for (int l = 0; l < 32; l++) {
            const float p = d1 * (float)((q[l] >> shift) & 0x0F);
            out[32 * g + l] = p - m1;
        }
    }
}

void vitna_dequant_q6_k(const uint8_t* block, float out[256]) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const int8_t* scales = (const int8_t*)(block + 192);
    const float d = vitna_f16_to_f32(u16le(block + 208));
    /* Two halves of 128 weights; in each, group r of 32 (0 to 3) takes low
       bits from byte l or l + 32 of the half's 64, shifted by 0 or 4, and
       high bits from byte l of its 32, shifted by 2r. */
    for (int h = 0; h < 2; h++) {
        const uint8_t* lo = ql + 64 * h;
        const uint8_t* hi = qh + 32 * h;
        for (int r = 0; r < 4; r++) {
            for (int l = 0; l < 32; l++) {
                const int low = (lo[l + 32 * (r & 1)] >> (4 * (r >> 1))) & 0x0F;
                const int high = (hi[l] >> (2 * r)) & 0x03;
                const int q = (int)(int8_t)(uint8_t)(low | (high << 4)) - 32;
                const int i = 128 * h + 32 * r + l;
                const float ds = d * (float)scales[i / 16];
                out[i] = ds * (float)q;
            }
        }
    }
}

#if defined(VITNA_QUANT_X86)
/* The same blocks eight weights at a time: each product _mm256_mul_ps and
 * each difference _mm256_sub_ps, never fused, so every weight is the bits
 * the scalar functions above give. */

/* Bytes p[0..7], as eight floats: signed for Q8_0's integers. */
VITNA_QUANT_AVX2 static __m256 s8x8(const uint8_t* p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}

VITNA_QUANT_AVX2 static __m256 u8x8(__m128i bytes) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
}

VITNA_QUANT_AVX2 static void q8_0_avx2(const uint8_t* block, float out[32]) {
    const __m256 d = _mm256_set1_ps(vitna_f16_to_f32(u16le(block)));
    for (int i = 0; i < 32; i += 8) _mm256_storeu_ps(out + i, _mm256_mul_ps(s8x8(block + 2 + i), d));
}

VITNA_QUANT_AVX2 static void q4_k_avx2(const uint8_t* block, float out[256]) {
    const float d = vitna_f16_to_f32(u16le(block));
    const float dmin = vitna_f16_to_f32(u16le(block + 2));
    const __m128i low4 = _mm_set1_epi8(0x0F);
    for (int g = 0; g < 8; g++) {
        uint8_t sc, mn;
        scale_min_k4(g, block + 4, &sc, &mn);
        const __m256 d1 = _mm256_set1_ps(d * (float)sc);
        const __m256 m1 = _mm256_set1_ps(dmin * (float)mn);
        const uint8_t* q = block + 16 + 32 * (g / 2);
        for (int l = 0; l < 32; l += 16) {
            __m128i v = _mm_loadu_si128((const __m128i*)(q + l));
            if (g & 1) v = _mm_srli_epi16(v, 4);
            v = _mm_and_si128(v, low4);
            /* bytes l to l + 15 of the group, as unsigned 4-bit integers */
            const __m256 lo = u8x8(v), hi = u8x8(_mm_srli_si128(v, 8));
            _mm256_storeu_ps(out + 32 * g + l, _mm256_sub_ps(_mm256_mul_ps(d1, lo), m1));
            _mm256_storeu_ps(out + 32 * g + l + 8, _mm256_sub_ps(_mm256_mul_ps(d1, hi), m1));
        }
    }
}

VITNA_QUANT_AVX2 static void q6_k_avx2(const uint8_t* block, float out[256]) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const int8_t* scales = (const int8_t*)(block + 192);
    const float d = vitna_f16_to_f32(u16le(block + 208));
    const __m128i low4 = _mm_set1_epi8(0x0F), low2 = _mm_set1_epi8(0x03);
    const __m256i bias = _mm256_set1_epi32(32);
    for (int h = 0; h < 2; h++) {
        for (int r = 0; r < 4; r++) {
            for (int l = 0; l < 32; l += 16) {
                __m128i lo = _mm_loadu_si128((const __m128i*)(ql + 64 * h + l + 32 * (r & 1)));
                lo = _mm_and_si128(_mm_srl_epi16(lo, _mm_cvtsi32_si128(4 * (r >> 1))), low4);
                __m128i hi = _mm_loadu_si128((const __m128i*)(qh + 32 * h + l));
                hi = _mm_and_si128(_mm_srl_epi16(hi, _mm_cvtsi32_si128(2 * r)), low2);
                const __m128i q = _mm_or_si128(lo, _mm_slli_epi16(hi, 4)); /* 0 to 63, a byte each */
                const int i = 128 * h + 32 * r + l;
                const __m256 ds = _mm256_set1_ps(d * (float)scales[i / 16]);
                const __m256 a = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(q), bias));
                const __m256 b = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(q, 8)), bias));
                _mm256_storeu_ps(out + i, _mm256_mul_ps(ds, a));
                _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(ds, b));
            }
        }
    }
}

static int quant_avx2(void) {
    static int avx2 = -1;
    if (avx2 < 0) avx2 = vitna_detect_simd_capabilities().has_avx2;
    return avx2;
}
#endif

void vitna_dequant(const void* src, vitna_dtype_t dtype, float* dst, size_t n) {
    const uint8_t* b = (const uint8_t*)src;
#if defined(VITNA_QUANT_X86)
    if (quant_avx2()) {
        if (dtype == VITNA_DTYPE_Q8_0) {
            for (size_t i = 0; i < n; i += 32, b += 34) q8_0_avx2(b, dst + i);
        } else if (dtype == VITNA_DTYPE_Q4_K) {
            for (size_t i = 0; i < n; i += 256, b += 144) q4_k_avx2(b, dst + i);
        } else if (dtype == VITNA_DTYPE_Q6_K) {
            for (size_t i = 0; i < n; i += 256, b += 210) q6_k_avx2(b, dst + i);
        }
        return;
    }
#endif
    if (dtype == VITNA_DTYPE_Q8_0) {
        for (size_t i = 0; i < n; i += 32, b += 34) vitna_dequant_q8_0(b, dst + i);
    } else if (dtype == VITNA_DTYPE_Q4_K) {
        for (size_t i = 0; i < n; i += 256, b += 144) vitna_dequant_q4_k(b, dst + i);
    } else if (dtype == VITNA_DTYPE_Q6_K) {
        for (size_t i = 0; i < n; i += 256, b += 210) vitna_dequant_q6_k(b, dst + i);
    }
}
