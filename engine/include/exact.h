/**
 * exact.h - exp and SiLU in arithmetic every device does the same way.
 *
 * CUDA's expf finishes with ex2.approx, the GPU's hardware approximation of
 * 2^x, whose bits the CPU cannot reproduce. The experts of a mixture run on
 * either device (gate A8: the CPU computes the experts the GPU's cache
 * lacks), and their outputs must be the same bits wherever they run, so
 * their activation uses this exp instead, on both: Cephes' expf, its range
 * reduced by Cody and Waite's two-part ln 2 and its polynomial in fused
 * multiply-adds, then scaled by a power of two in two exact steps. Every
 * operation is a correctly rounded IEEE one (fmaf, a product, a sum, a
 * quotient, a rounding to an integer), so the CPU and model_cuda.cu, which
 * repeats it operation for operation, agree bit for bit. It is within about
 * two units in the last place of the true exp, as libm's and CUDA's are.
 *
 * The dense path and the CPU-only path keep the C library's expf.
 */

#ifndef VITNA_EXACT_H
#define VITNA_EXACT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** e^x: +inf above 88.7228, 0 below -87.3365, NaN for NaN. */
float vitna_exp_exact(float x);

/** out[i] = (g[i] / (1 + vitna_exp_exact(-g[i]))) * u[i]: SiLU of the gate
    times the up projection, as the GPU's experts compute it. out may be g. */
void vitna_silu_mul_exact(const float* g, const float* u, float* out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_EXACT_H */
