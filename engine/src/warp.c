/**
 * warp.c - Products summed in the GPU's order, on the CPU (warp.h).
 *
 * The AVX2 path keeps the 32 lanes' sums in four registers of eight. For
 * each group of 32 chunks (256 weights, chunk l of the group being lane l's
 * next), it loads eight lanes' chunks as the rows of an 8 x 8 block,
 * transposes the block so that register e holds element e of each lane's
 * chunk, and fuses the eight elements into the lanes' sums in order, against
 * x transposed the same way once a call. So each lane adds its products in
 * the order a GPU lane does; the lanes are then added in warp_sum's tree.
 */

#include "warp.h"
#include "exact.h"
#include "kernels.h"
#include "ops.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
  #define VITNA_WARP_X86 1
  #include <immintrin.h>
#endif

#if defined(VITNA_WARP_X86) && (defined(__GNUC__) || defined(__clang__))
  #define VITNA_WARP_AVX2 __attribute__((target("avx2,fma")))
#else
  #define VITNA_WARP_AVX2
#endif

/* warp_sum's tree over 32 lanes' sums, as lane 0 sees it: at the level with
 * offset o, lane l adds lane l + o's partial to its own. */
static float tree32(float* s) {
    for (int o = 16; o > 0; o >>= 1) {
        for (int l = 0; l < o; l++) s[l] = s[l] + s[l + o];
    }
    return s[0];
}

void vitna_warp_matvec_scalar(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y) {
    const size_t rb = (size_t)vitna_row_bytes(dtype, cols);
    float* row = (float*)malloc(cols * sizeof(float));
    if (!row) return;
    for (size_t r = 0; r < rows; r++) {
        vitna_to_f32((const char*)w + r * rb, dtype, row, cols);
        float s[32] = { 0 };
        for (size_t c = 0; c < cols / 8; c++) {
            float* lane = &s[c % 32];
            for (size_t e = 0; e < 8; e++) *lane = fmaf(row[8 * c + e], x[8 * c + e], *lane);
        }
        y[r] = tree32(s);
    }
    free(row);
}

#if defined(VITNA_WARP_X86)
/* The 8 x 8 block m transposed in place: m[e] then holds element e of each of the eight rows. */
VITNA_WARP_AVX2
static void transpose8(__m256 m[8]) {
    const __m256 t0 = _mm256_unpacklo_ps(m[0], m[1]), t1 = _mm256_unpackhi_ps(m[0], m[1]);
    const __m256 t2 = _mm256_unpacklo_ps(m[2], m[3]), t3 = _mm256_unpackhi_ps(m[2], m[3]);
    const __m256 t4 = _mm256_unpacklo_ps(m[4], m[5]), t5 = _mm256_unpackhi_ps(m[4], m[5]);
    const __m256 t6 = _mm256_unpacklo_ps(m[6], m[7]), t7 = _mm256_unpackhi_ps(m[6], m[7]);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)), u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)), u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0)), u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0)), u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    m[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
    m[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    m[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
    m[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    m[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
    m[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    m[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
    m[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

VITNA_WARP_AVX2
static void matvec_avx2(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y) {
    const size_t rb = (size_t)vitna_row_bytes(dtype, cols);
    const size_t groups = cols / 256;
    float* row = (float*)malloc(cols * sizeof(float));
    float* xt = (float*)malloc(cols * sizeof(float));
    if (!row || !xt) {
        free(row);
        free(xt);
        return;
    }
    /* xt[(j * 8 + e) * 32 + l] = element e of lane l's chunk in group j. */
    for (size_t j = 0; j < groups; j++) {
        for (size_t l = 0; l < 32; l++) {
            for (size_t e = 0; e < 8; e++) xt[(j * 8 + e) * 32 + l] = x[(32 * j + l) * 8 + e];
        }
    }
    for (size_t r = 0; r < rows; r++) {
        vitna_to_f32((const char*)w + r * rb, dtype, row, cols);
        __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (size_t j = 0; j < groups; j++) {
            const float* wg = row + 256 * j;
            const float* xg = xt + j * 256;
            for (int g = 0; g < 4; g++) {
                __m256 m[8];
                for (int i = 0; i < 8; i++) m[i] = _mm256_loadu_ps(wg + 8 * (8 * g + i));
                transpose8(m);
                for (int e = 0; e < 8; e++) acc[g] = _mm256_fmadd_ps(m[e], _mm256_loadu_ps(xg + e * 32 + 8 * g), acc[g]);
            }
        }
        float s[32];
        for (int g = 0; g < 4; g++) _mm256_storeu_ps(s + 8 * g, acc[g]);
        y[r] = tree32(s);
    }
    free(row);
    free(xt);
}
#endif

static bool use_avx2(size_t cols) {
#if defined(VITNA_WARP_X86)
    static int avx2 = -1;
    if (avx2 < 0) {
        const vitna_simd_capabilities_t caps = vitna_detect_simd_capabilities();
        avx2 = caps.has_avx2 && caps.has_fma;
    }
    return avx2 && cols % 256 == 0;
#else
    (void)cols;
    return false;
#endif
}

const char* vitna_warp_path(size_t cols) {
    return use_avx2(cols) ? "avx2+fma" : "scalar";
}

void vitna_warp_matvec(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y) {
#if defined(VITNA_WARP_X86)
    if (use_avx2(cols)) {
        matvec_avx2(w, dtype, rows, cols, x, y);
        return;
    }
#endif
    vitna_warp_matvec_scalar(w, dtype, rows, cols, x, y);
}

void vitna_warp_expert(const void* gate, const void* up, const void* down, vitna_dtype_t gu_dtype, vitna_dtype_t down_dtype,
                       size_t hidden, size_t intermediate, const float* xs, float* y, float* act) {
    float* g = act;
    float* u = act + intermediate;
    vitna_warp_matvec(gate, gu_dtype, intermediate, hidden, xs, g);
    vitna_warp_matvec(up, gu_dtype, intermediate, hidden, xs, u);
    vitna_silu_mul_exact(g, u, g, intermediate);
    vitna_warp_matvec(down, down_dtype, hidden, intermediate, g, y);
}
