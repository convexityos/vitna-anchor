/**
 * exact.c - exp and SiLU in arithmetic every device does the same way (exact.h).
 *
 * Each operation is its own statement, and the pragmas forbid a compiler to
 * fuse a product into a later sum, so what runs is what is written: the same
 * sequence model_cuda.cu's vitna_exp_dev writes with __fmaf_rn, __fmul_rn,
 * __fadd_rn and __fdiv_rn.
 */

#if defined(__clang__)
  #pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
  #pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
  #pragma fp_contract(off)
#endif

#include "exact.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

/* 2^e for e in [-126, 127], from its bits. */
static float pow2(int e) {
    const uint32_t bits = (uint32_t)(e + 127) << 23;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

float vitna_exp_exact(float x) {
    if (x != x) return x;
    if (x > 88.72283935546875f) return HUGE_VALF;
    if (x < -87.3365478515625f) return 0.0f;
    /* n, the nearest integer to x / ln 2; r = x - n ln 2, with ln 2 in two parts. */
    const float t = x * 1.44269502162933349609375f;
    const float n = rintf(t);
    float r = fmaf(n, -0.693359375f, x);
    r = fmaf(n, 2.12194440e-4f, r);
    /* e^r on [-ln 2 / 2, ln 2 / 2]: 1 + r + r^2 p(r). */
    const float z = r * r;
    float p = 1.9875691500e-4f;
    p = fmaf(p, r, 1.3981999507e-3f);
    p = fmaf(p, r, 8.3334519073e-3f);
    p = fmaf(p, r, 4.1665795894e-2f);
    p = fmaf(p, r, 1.6666665459e-1f);
    p = fmaf(p, r, 5.0000001201e-1f);
    p = fmaf(p, z, r);
    p = p + 1.0f;
    /* Times 2^n, n in [-126, 128], in two steps of at most 2^64 each, so
     * neither factor leaves the normal range; only the last can round, and
     * only into the subnormals. */
    const int ni = (int)n;
    const int h = ni / 2;
    const float a = p * pow2(h);
    return a * pow2(ni - h);
}

void vitna_silu_mul_exact(const float* g, const float* u, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float e = vitna_exp_exact(-g[i]);
        const float d = 1.0f + e;
        const float s = g[i] / d;
        out[i] = s * u[i];
    }
}
