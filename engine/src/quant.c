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
#include "ops.h"
#include <string.h>

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

void vitna_dequant(const void* src, vitna_dtype_t dtype, float* dst, size_t n) {
    const uint8_t* b = (const uint8_t*)src;
    if (dtype == VITNA_DTYPE_Q8_0) {
        for (size_t i = 0; i < n; i += 32, b += 34) vitna_dequant_q8_0(b, dst + i);
    } else if (dtype == VITNA_DTYPE_Q4_K) {
        for (size_t i = 0; i < n; i += 256, b += 144) vitna_dequant_q4_k(b, dst + i);
    } else if (dtype == VITNA_DTYPE_Q6_K) {
        for (size_t i = 0; i < n; i += 256, b += 210) vitna_dequant_q6_k(b, dst + i);
    }
}
