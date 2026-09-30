/**
 * model_cuda.cu - The forward pass of model.c on an NVIDIA GPU, in float32.
 *
 * The computation is model.c's vitna_llama_step: the token's embedding row,
 * then per layer RMSNorm, the query, key and value projections, the
 * half-split rotary embedding, attention over the key-value cache, the output
 * projection added to the residual, RMSNorm, the gate and up projections,
 * SiLU(gate) * up and the down projection added to the residual; then the
 * final RMSNorm and the output projection.
 *
 * Kernels are fused where one of those steps feeds the next, so a layer is
 * five kernels:
 *   - RMSNorm into shared memory, then the query, key and value projections,
 *     with the rotary embedding applied as the query and key rows are written;
 *   - attention, its positions split into slices across blocks, the slices
 *     combined by the last block to finish;
 *   - the output projection, added to the residual;
 *   - RMSNorm into shared memory, then the gate and up projections, with
 *     SiLU(gate) * up formed as their rows are;
 *   - the down projection, added to the residual.
 * The final RMSNorm and the output projection are one more. Fusing changes
 * where intermediate vectors live and the order in which a row's products are
 * summed, and nothing else: each step computes what model.c computes.
 *
 * Weights are uploaded once, in their stored dtype, and widened to float32 as
 * they are read, as ops.c does, sixteen bytes at a time. That needs every
 * matrix in one dtype (F32, BF16 or F16) and widths that are multiples of
 * eight; a model that is not so is refused on load, with the reason. The
 * key-value cache is float32 on the device, in model.c's layout, [layer]
 * [position][key-value head * head_dim], and a token's keys and values are
 * written straight into their slot. The rotary cos and sin of every position
 * come from model.c, computed on the host the way the CPU path computes them.
 *
 * Arithmetic is float32 on CUDA cores. Nothing here uses tensor cores, so
 * TF32 never applies, and the build does not pass --use_fast_math, so
 * division, square root and expf keep their accurate forms. nvcc's default
 * fused multiply-add stays on; the CPU path's AVX2 matvec uses FMA too.
 *
 * A token is a replay of CUDA graphs, not a launch of every kernel. After the
 * upload, the kernels for the embedding and every layer are captured into one
 * graph, and the head into a second. The token and its position reach the
 * kernels through a small struct on the device, which one kernel writes
 * before each replay, so the same graphs serve every position. Everything
 * runs in order on the model's stream. A step that returns no logits does not
 * wait for the device; one that does waits for the copy back.
 *
 * A prompt runs up to PREFILL_MAX tokens at a time instead (vitna_cuda_steps), each
 * layer taking all of them together. The projections become matrix-matrix
 * products, which read each weight once for the whole chunk rather than once
 * a token, and RMSNorm, the rotary embedding, attention and SiLU(gate) * up
 * are kernels of their own. A token's attention reads the cache up to its own
 * position, which holds the chunk's earlier tokens by then, so it is causal as
 * a step is. This is the same computation again, its sums in other orders.
 * The next token's logits come from the head a step uses; when every
 * position's are asked for, a matrix-matrix product computes them.
 */

#define NOMINMAX
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "model_cuda.h"

/* Threads per block for every kernel but set_step_kernel: eight warps. */
#define THREADS 256
#define WARPS (THREADS / 32)

/* Attention splits each key-value head's positions into at most this many
 * slices, of at least ATTENTION_MIN_SLICE positions each. */
#define ATTENTION_SPLITS 32
#define ATTENTION_MIN_SLICE 32

/* A prompt runs in chunks of g->prefill tokens: as many as PREFILL_BYTES of
 * scratch holds, at most PREFILL_MAX and the cache, at least PREFILL_MIN.
 * Larger chunks give each kernel more blocks; past 2048 tokens SmolLM2-135M
 * gained nothing measurable. The logits of every position, when they are
 * asked for, come LOGIT_ROWS rows at a time. A warp of a
 * prompt's attention takes up to PROMPT_ROWS query rows, so a model whose
 * key-value heads each serve more query heads than that runs its prompts a
 * token at a time. */
#define PREFILL_MAX 2048
#define PREFILL_MIN 256
#define PREFILL_BYTES ((size_t)64 << 20)
#define LOGIT_ROWS 32
#define PROMPT_ROWS 8

/* The fewest tokens that run together; fewer are faster a step at a time. */
#define PROMPT_MIN 2

/* The matrix-matrix product takes a tile of GEMM_BM tokens by GEMM_BN
 * outputs at a time, over the inner dimension GEMM_BK at a time. */
#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 16

/* Up to FEW_TOKENS tokens take gemm_few_kernel instead: a block FEW_TB tokens
 * by WARPS * FEW_MR outputs, a warp FEW_MR outputs. */
#define FEW_TOKENS 32
#define FEW_TB 16
#define FEW_MR 4

/* Minus infinity, from its bits: MSVC's INFINITY macro overflows a constant to get it. */
#define NEG_INF __uint_as_float(0xff800000u)

typedef struct {
    const void* w;
    int rows, cols;
} dmat_t;

typedef struct {
    dmat_t q, k, v, o, gate, up, down;
    const float* attn_norm;
    const float* mlp_norm;
} dlayer_t;

/* The token being run and its position, as the graphs read them. */
typedef struct {
    int token;
    int pos;
} step_t;

struct vitna_cuda_model {
    int device;
    char name[320];
    int max_blocks;         /* a grid's cap: eight blocks per multiprocessor */

    int n_layers, hidden, intermediate, n_heads, n_kv_heads, head_dim, vocab, ctx;
    float eps, scale;
    vitna_dtype_t dtype;    /* of every weight matrix */

    void* arena;            /* one allocation, carved into everything below */
    dlayer_t* layers;       /* host array of device pointers */
    dmat_t embed, lm_head;
    const float* final_norm;
    const float* cos_t;     /* [ctx][head_dim / 2] */
    const float* sin_t;
    float* k_cache;         /* [n_layers][ctx][n_kv_heads * head_dim] */
    float* v_cache;
    float *x, *q, *att, *act, *logits;
    float *part_m, *part_l, *part_o; /* attention's slices: [n_heads][splits], and [.][.][head_dim] */
    unsigned int* done;              /* per key-value head, the slices finished in this layer */
    step_t* step;

    /* A prompt's chunk: [prefill][hidden] for p_x and p_xn, [prefill][n_heads
     * * head_dim] for p_q and p_att, [prefill][intermediate] for p_gate and
     * p_up, and [LOGIT_ROWS][vocab] for p_logits; p_tokens holds the whole
     * prompt, [ctx] at most. */
    float *p_x, *p_xn, *p_q, *p_att, *p_gate, *p_up, *p_logits;
    int32_t* p_tokens;
    size_t prefill;         /* the tokens in a prompt's chunk */
    bool prompt;            /* whether a prompt runs many tokens at once (prompt_batches) */

    cudaStream_t stream;
    cudaGraphExec_t body;   /* the embedding and every layer */
    cudaGraphExec_t head;   /* the final RMSNorm and the output projection */
    float* host_logits;     /* pinned, for the copy back */
};

static bool fail(char* err, size_t err_len, const char* fmt, ...) {
    if (err && err_len > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, err_len, fmt, ap);
        va_end(ap);
    }
    return false;
}

static bool fail_cuda(char* err, size_t err_len, const char* what, cudaError_t e) {
    return fail(err, err_len, "%s: %s (%s)", what, cudaGetErrorString(e), cudaGetErrorName(e));
}

/* --- Device code --- */

/* Every half is exactly a float32; the conversion keeps subnormals, as vitna_f16_to_f32 does. */
__device__ __forceinline__ float half_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

/* bfloat16 is the high half of a float32, as in vitna_bf16_to_f32. In a
 * 32-bit word the lower-addressed of two 16-bit values is the low half. */
__device__ __forceinline__ float bf16_low(unsigned int u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_high(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }

/* Element i of a weight matrix, widened to float32 exactly. */
template <vitna_dtype_t DT>
__device__ __forceinline__ float widen(const void* w, size_t i);

template <>
__device__ __forceinline__ float widen<VITNA_DTYPE_F32>(const void* w, size_t i) {
    return static_cast<const float*>(w)[i];
}

template <>
__device__ __forceinline__ float widen<VITNA_DTYPE_BF16>(const void* w, size_t i) {
    return bf16_low(static_cast<const unsigned short*>(w)[i]);
}

template <>
__device__ __forceinline__ float widen<VITNA_DTYPE_F16>(const void* w, size_t i) {
    return half_to_float(static_cast<const unsigned short*>(w)[i]);
}

/* Weights 8c to 8c + 7 of a row, widened to float32 exactly. */
template <vitna_dtype_t DT>
__device__ __forceinline__ void load8(const void* row, int c, float w[8]);

template <>
__device__ __forceinline__ void load8<VITNA_DTYPE_F32>(const void* row, int c, float w[8]) {
    const float4* p = static_cast<const float4*>(row) + 2 * c;
    const float4 a = p[0], b = p[1];
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
    w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
}

template <>
__device__ __forceinline__ void load8<VITNA_DTYPE_BF16>(const void* row, int c, float w[8]) {
    const uint4 p = static_cast<const uint4*>(row)[c];
    w[0] = bf16_low(p.x); w[1] = bf16_high(p.x);
    w[2] = bf16_low(p.y); w[3] = bf16_high(p.y);
    w[4] = bf16_low(p.z); w[5] = bf16_high(p.z);
    w[6] = bf16_low(p.w); w[7] = bf16_high(p.w);
}

template <>
__device__ __forceinline__ void load8<VITNA_DTYPE_F16>(const void* row, int c, float w[8]) {
    const uint4 p = static_cast<const uint4*>(row)[c];
    w[0] = half_to_float((unsigned short)(p.x & 0xffffu)); w[1] = half_to_float((unsigned short)(p.x >> 16));
    w[2] = half_to_float((unsigned short)(p.y & 0xffffu)); w[3] = half_to_float((unsigned short)(p.y >> 16));
    w[4] = half_to_float((unsigned short)(p.z & 0xffffu)); w[5] = half_to_float((unsigned short)(p.z >> 16));
    w[6] = half_to_float((unsigned short)(p.w & 0xffffu)); w[7] = half_to_float((unsigned short)(p.w >> 16));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* The sum over a block of THREADS threads, returned to every thread. red holds WARPS floats. */
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads(); /* red may still be read from the reduction before */
    if (lane == 0) red[warp] = v;
    __syncthreads();
    return warp_sum(lane < WARPS ? red[lane] : 0.0f);
}

__device__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    return warp_max(lane < WARPS ? red[lane] : NEG_INF);
}

/* Row r of a matrix of the given width, times x, summed across the warp:
 * lane l takes chunks l, l + 32, ... of eight weights. Every lane gets the
 * sum. x may be in shared or global memory; it is 16-byte aligned. */
template <vitna_dtype_t DT>
__device__ __forceinline__ float row_dot(const void* w, int r, int cols, const float* x, int lane) {
    const size_t bytes = DT == VITNA_DTYPE_F32 ? 4 : 2;
    const void* row = static_cast<const char*>(w) + (size_t)r * cols * bytes;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float s = 0.0f;
    for (int c = lane; c < cols / 8; c += 32) {
        float v[8];
        load8<DT>(row, c, v);
        const float4 a = x4[2 * c];
        const float4 b = x4[2 * c + 1];
        s = fmaf(v[0], a.x, s);
        s = fmaf(v[1], a.y, s);
        s = fmaf(v[2], a.z, s);
        s = fmaf(v[3], a.w, s);
        s = fmaf(v[4], b.x, s);
        s = fmaf(v[5], b.y, s);
        s = fmaf(v[6], b.z, s);
        s = fmaf(v[7], b.w, s);
    }
    return warp_sum(s);
}

/* xs = x / sqrt(mean(x^2) + eps) * w, as kernels.c's vitna_rmsnorm, written
 * to shared memory by the whole block. Every block computes the same values. */
__device__ void norm_to_shared(const float* __restrict__ x, const float* __restrict__ w, float* xs, int n, float eps, float* red) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += THREADS) ss = fmaf(x[i], x[i], ss);
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += THREADS) xs[i] = x[i] * inv * w[i];
    __syncthreads();
}

/* The one kernel a step launches outside the graphs: it says which token to
 * run and where. Its arguments are copied at launch, so the host can move on. */
__global__ void set_step_kernel(step_t* step, int token, int pos) {
    step->token = token;
    step->pos = pos;
}

template <vitna_dtype_t DT>
__global__ void embed_kernel(const void* __restrict__ table, const step_t* __restrict__ step, int hidden, float* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[i] = widen<DT>(table, (size_t)step->token * hidden + i);
}

/* RMSNorm, then the query, key and value projections of one layer. The
 * half-split rotary embedding of ops.c's vitna_rope_half pairs element j of a
 * head with element j + head_dim / 2, so a warp computes those two rows
 * together and writes them rotated. Keys and values go straight to the
 * position's slot in the cache; value rows are taken two at a time, unrotated. */
template <vitna_dtype_t DT>
__global__ void attn_in_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                               const void* __restrict__ wq, const void* __restrict__ wk, const void* __restrict__ wv,
                               float* __restrict__ q, float* __restrict__ kc, float* __restrict__ vc,
                               const float* __restrict__ cos_all, const float* __restrict__ sin_all,
                               const step_t* __restrict__ step, int hidden, int n_heads, int n_kv_heads, int head_dim) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    norm_to_shared(x, norm_w, xs, hidden, eps, red);

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = head_dim / 2;
    const int kv_dim = n_kv_heads * head_dim;
    const size_t pos = (size_t)step->pos;
    const float* cos_t = cos_all + pos * half;
    const float* sin_t = sin_all + pos * half;
    float* k = kc + pos * kv_dim;
    float* v = vc + pos * kv_dim;
    const int units_q = n_heads * half, units_qk = units_q + n_kv_heads * half;
    const int units = units_qk + kv_dim / 2;
    for (int u = blockIdx.x * WARPS + warp; u < units; u += gridDim.x * WARPS) {
        if (u < units_qk) {
            const bool is_q = u < units_q;
            const int uu = is_q ? u : u - units_q;
            const int j = uu % half;
            const int ra = (uu / half) * head_dim + j, rb = ra + half;
            const void* w = is_q ? wq : wk;
            const float a = row_dot<DT>(w, ra, hidden, xs, lane);
            const float b = row_dot<DT>(w, rb, hidden, xs, lane);
            if (lane == 0) {
                float* out = is_q ? q : k;
                out[ra] = a * cos_t[j] - b * sin_t[j];
                out[rb] = b * cos_t[j] + a * sin_t[j];
            }
        } else {
            const int ra = 2 * (u - units_qk);
            const float a = row_dot<DT>(wv, ra, hidden, xs, lane);
            const float b = row_dot<DT>(wv, ra + 1, hidden, xs, lane);
            if (lane == 0) {
                v[ra] = a;
                v[ra + 1] = b;
            }
        }
    }
}

/* Attention over positions 0 to pos, split across blocks by position
 * (flash-decoding). Block (split, kvh) takes key-value head kvh with the
 * query heads that read it (repeat_kv: query head h reads key-value head
 * h / group), and one slice of the positions: their scores, a softmax over
 * the slice alone, and the slice's weighted sum of values, unnormalized.
 * Only as many slices as the positions need at ATTENTION_MIN_SLICE each are
 * used, and the other blocks return at once. The last block of a key-value
 * head to finish combines the slices, in split order, into the softmax over
 * every position: a slice's sums are scaled by exp(its max - the overall
 * max). That is model.c's softmax, summed in another order. Dynamic shared
 * memory holds the group's queries, then the slice's scores: group *
 * (head_dim + per) floats. */
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
                                 float* __restrict__ att, float* __restrict__ part_m, float* __restrict__ part_l,
                                 float* __restrict__ part_o, unsigned int* __restrict__ done, const step_t* __restrict__ step,
                                 int head_dim, int kv_dim, int group, float scale) {
    extern __shared__ float4 shared4[];
    __shared__ int last;
    const int split = blockIdx.x, kvh = blockIdx.y;
    const int n = step->pos + 1;
    /* As many slices as n needs at ATTENTION_MIN_SLICE positions each, up
     * to the grid's width; the blocks past them have nothing to do. */
    const int splits = min((int)gridDim.x, (n + ATTENTION_MIN_SLICE - 1) / ATTENTION_MIN_SLICE);
    if (split >= splits) return;
    const int per = (n + splits - 1) / splits;
    const int t0 = split * per;
    const int len = max(0, min(n, t0 + per) - t0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int h0 = kvh * group; /* the first query head that reads this key-value head */
    const int gd = group * head_dim;
    const float* kh = kc + (size_t)kvh * head_dim;
    const float* vh = vc + (size_t)kvh * head_dim;
    float* qs = reinterpret_cast<float*>(shared4);
    float* s = qs + gd; /* s[h * per + t] */

    for (int i = threadIdx.x; i < gd; i += THREADS) qs[i] = q[(size_t)h0 * head_dim + i];
    __syncthreads();

    /* Scores, a warp per query head, its lanes across the positions: lane l
     * takes positions l, l + 32, ..., each a whole dot product of its own,
     * read sixteen bytes at a time. Then, in the same warp, the slice's
     * softmax left unnormalized: its max, exp(score - max) and their sum. */
    for (int h = warp; h < group; h += WARPS) {
        const float4* q4 = reinterpret_cast<const float4*>(qs + h * head_dim);
        float* sh = s + h * per;
        float mx = NEG_INF;
        for (int t = lane; t < len; t += 32) {
            const float4* k4 = reinterpret_cast<const float4*>(kh + (size_t)(t0 + t) * kv_dim);
            float d = 0.0f;
#pragma unroll 8
            for (int i = 0; i < head_dim / 4; i++) {
                const float4 kv = k4[i];
                const float4 qv = q4[i];
                d = fmaf(qv.x, kv.x, d);
                d = fmaf(qv.y, kv.y, d);
                d = fmaf(qv.z, kv.z, d);
                d = fmaf(qv.w, kv.w, d);
            }
            d *= scale;
            sh[t] = d;
            mx = fmaxf(mx, d);
        }
        mx = warp_max(mx);
        float sum = 0.0f;
        for (int t = lane; t < len; t += 32) {
            const float e = expf(sh[t] - mx);
            sh[t] = e;
            sum += e;
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            part_m[(size_t)(h0 + h) * ATTENTION_SPLITS + split] = mx;  /* minus infinity for an empty slice */
            part_l[(size_t)(h0 + h) * ATTENTION_SPLITS + split] = sum; /* and 0 */
        }
    }
    __syncthreads();

    /* The slice's sum of exp-weighted values, thread (h, i) for dimension i
     * of head h, its loads issued sixteen positions at a time. */
    for (int hi = threadIdx.x; hi < gd; hi += THREADS) {
        const int h = hi / head_dim, i = hi % head_dim;
        const float* p = s + h * per;
        const float* vcol = vh + (size_t)t0 * kv_dim + i;
        float acc = 0.0f;
        for (int tb = 0; tb < len; tb += 16) {
            float v[16];
#pragma unroll
            for (int j = 0; j < 16; j++) {
                if (tb + j < len) v[j] = vcol[(size_t)(tb + j) * kv_dim];
            }
#pragma unroll
            for (int j = 0; j < 16; j++) {
                if (tb + j < len) acc = fmaf(p[tb + j], v[j], acc);
            }
        }
        part_o[((size_t)(h0 + h) * ATTENTION_SPLITS + split) * head_dim + i] = acc;
    }

    /* Every block publishes its slice; the last of this key-value head's blocks to do so combines. */
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&done[kvh], 1u) == (unsigned int)splits - 1;
    __syncthreads();
    if (!last) return;
    for (int hi = threadIdx.x; hi < gd; hi += THREADS) {
        const size_t h = (size_t)h0 + hi / head_dim;
        const int i = hi % head_dim;
        /* Other blocks wrote these: read them from L2, past this SM's L1. */
        const size_t row = h * ATTENTION_SPLITS;
        float mx = NEG_INF;
        for (int k = 0; k < splits; k++) mx = fmaxf(mx, __ldcg(&part_m[row + k]));
        float sum = 0.0f, o = 0.0f;
        for (int k = 0; k < splits; k++) {
            const float w = expf(__ldcg(&part_m[row + k]) - mx); /* 0 for an empty slice */
            sum = fmaf(__ldcg(&part_l[row + k]), w, sum);
            o = fmaf(__ldcg(&part_o[(row + k) * head_dim + i]), w, o);
        }
        att[h * head_dim + i] = o / sum;
    }
    if (threadIdx.x == 0) done[kvh] = 0; /* ready for the next layer */
}

/* y += W x, a warp per row: the output and down projections, added to the residual. */
template <vitna_dtype_t DT>
__global__ void matvec_add_kernel(const void* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * WARPS + warp; r < rows; r += gridDim.x * WARPS) {
        const float s = row_dot<DT>(w, r, cols, x, lane);
        if (lane == 0) y[r] = y[r] + s;
    }
}

/* RMSNorm, then the gate and up projections of one layer. A warp computes
 * row r of both and writes silu(gate) * up, with silu(g) = g / (1 + exp(-g)),
 * as ops.c's vitna_silu_mul. */
template <vitna_dtype_t DT>
__global__ void mlp_in_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                              const void* __restrict__ wg, const void* __restrict__ wu, float* __restrict__ act,
                              int hidden, int intermediate) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    norm_to_shared(x, norm_w, xs, hidden, eps, red);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * WARPS + warp; r < intermediate; r += gridDim.x * WARPS) {
        const float g = row_dot<DT>(wg, r, hidden, xs, lane);
        const float u = row_dot<DT>(wu, r, hidden, xs, lane);
        if (lane == 0) act[r] = (g / (1.0f + expf(-g))) * u;
    }
}

/* The final RMSNorm and the output projection: logits = W rmsnorm(x). */
template <vitna_dtype_t DT>
__global__ void head_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                            const void* __restrict__ w, float* __restrict__ logits, int hidden, int vocab) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    norm_to_shared(x, norm_w, xs, hidden, eps, red);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * WARPS + warp; r < vocab; r += gridDim.x * WARPS) {
        const float s = row_dot<DT>(w, r, hidden, xs, lane);
        if (lane == 0) logits[r] = s;
    }
}

/* --- A prompt, many tokens at once --- */

/* Row t of x is token t's embedding row, widened to float32. */
template <vitna_dtype_t DT>
__global__ void embed_rows_kernel(const void* __restrict__ table, const int32_t* __restrict__ tokens, int hidden, float* __restrict__ x) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[(size_t)t * hidden + i] = widen<DT>(table, (size_t)tokens[t] * hidden + i);
}

/* Row t of y is x's row t through RMSNorm, as kernels.c's vitna_rmsnorm: a block a row. */
__global__ void rmsnorm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n, float eps) {
    __shared__ float red[WARPS];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = y + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += THREADS) ss = fmaf(xr[i], xr[i], ss);
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += THREADS) yr[i] = xr[i] * inv * w[i];
}

/* Weights i to i + 3 of a matrix, widened to float32 exactly; i is a multiple of 4. */
template <vitna_dtype_t DT>
__device__ __forceinline__ void load4(const void* w, size_t i, float out[4]);

template <>
__device__ __forceinline__ void load4<VITNA_DTYPE_F32>(const void* w, size_t i, float out[4]) {
    const float4 p = static_cast<const float4*>(w)[i / 4];
    out[0] = p.x; out[1] = p.y; out[2] = p.z; out[3] = p.w;
}

template <>
__device__ __forceinline__ void load4<VITNA_DTYPE_BF16>(const void* w, size_t i, float out[4]) {
    const uint2 p = static_cast<const uint2*>(w)[i / 4];
    out[0] = bf16_low(p.x); out[1] = bf16_high(p.x);
    out[2] = bf16_low(p.y); out[3] = bf16_high(p.y);
}

template <>
__device__ __forceinline__ void load4<VITNA_DTYPE_F16>(const void* w, size_t i, float out[4]) {
    const uint2 p = static_cast<const uint2*>(w)[i / 4];
    out[0] = half_to_float((unsigned short)(p.x & 0xffffu)); out[1] = half_to_float((unsigned short)(p.x >> 16));
    out[2] = half_to_float((unsigned short)(p.y & 0xffffu)); out[3] = half_to_float((unsigned short)(p.y >> 16));
}

/* Y = X W^T for T rows of X: y[t][n] is the sum over k of x[t][k] * W[n][k],
 * W's rows being its outputs, as the checkpoint stores it. With add, the sums
 * are added to Y instead: the residual add after a projection. Row t of X is
 * at X + t * ldx and of Y at Y + t * ldy. A block computes a GEMM_BM x
 * GEMM_BN tile of Y, each of its 256 threads 4 x 4 of it, taking GEMM_BK of
 * the inner dimension at a time through shared memory, and loading the next
 * GEMM_BK into registers while it multiplies the last. Every output sums its
 * products in order of k, whatever T is. K is a multiple of GEMM_BK. */
template <vitna_dtype_t DT>
__global__ void gemm_kernel(const float* __restrict__ X, int ldx, const void* __restrict__ W, float* __restrict__ Y, int ldy,
                            int T, int N, int K, int add) {
    __shared__ __align__(16) float xs[GEMM_BK][GEMM_BM + 4];
    __shared__ __align__(16) float ws[GEMM_BK][GEMM_BN + 4];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int t0 = blockIdx.y * GEMM_BM, n0 = blockIdx.x * GEMM_BN;
    const int lr = tid >> 2, lk = (tid & 3) * 4; /* the row and the four k this thread loads */
    const bool xrow = t0 + lr < T, wrow = n0 + lr < N;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    }
    float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float wv[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (xrow) xv = *reinterpret_cast<const float4*>(X + (size_t)(t0 + lr) * ldx + lk);
    if (wrow) load4<DT>(W, (size_t)(n0 + lr) * K + lk, wv);
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        xs[lk][lr] = xv.x;
        xs[lk + 1][lr] = xv.y;
        xs[lk + 2][lr] = xv.z;
        xs[lk + 3][lr] = xv.w;
        ws[lk][lr] = wv[0];
        ws[lk + 1][lr] = wv[1];
        ws[lk + 2][lr] = wv[2];
        ws[lk + 3][lr] = wv[3];
        __syncthreads();
        if (k0 + GEMM_BK < K) {
            if (xrow) xv = *reinterpret_cast<const float4*>(X + (size_t)(t0 + lr) * ldx + k0 + GEMM_BK + lk);
            if (wrow) load4<DT>(W, (size_t)(n0 + lr) * K + k0 + GEMM_BK + lk, wv);
        }
#pragma unroll
        for (int kk = 0; kk < GEMM_BK; kk++) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[kk][tx * 4]);
            const float av[4] = { a.x, a.y, a.z, a.w };
            const float bv[4] = { b.x, b.y, b.z, b.w };
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int t = t0 + ty * 4 + i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n < N) {
                float* y = Y + (size_t)t * ldy + n;
                *y = add ? *y + acc[i][j] : acc[i][j];
            }
        }
    }
}

/* Y = X W^T for a few tokens, each output summed exactly as row_dot sums it
 * for one token: lane l takes chunks l, l + 32, ... of eight weights, and the
 * warp adds its lanes in warp_sum's tree. So each output is, bit for bit,
 * the one a step's projection gives for the same input row. Block (x, y) takes TB tokens from x * TB and warp w
 * the MR outputs from (y * WARPS + w) * MR. X comes through shared memory 32
 * chunks at a time, in two planes of float4 so that the lanes read
 * consecutive addresses, and each lane keeps MR * TB partial sums, reduced at
 * the end 32 at a time by a transposing butterfly: after it, lane l holds sum
 * l of the group, added in warp_sum's order. add = 1 adds into Y. */
template <vitna_dtype_t DT, int MR, int TB>
__global__ void gemm_few_kernel(const float* __restrict__ X, int ldx, const void* __restrict__ W, float* __restrict__ Y, int ldy,
                                int T, int N, int K, int add) {
    __shared__ float4 lo[TB][32], hi[TB][32];
    const size_t bytes = DT == VITNA_DTYPE_F32 ? 4 : 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t0 = blockIdx.x * TB, n0 = (blockIdx.y * WARPS + warp) * MR;
    const int chunks = K / 8;
    float acc[MR * TB];
#pragma unroll
    for (int i = 0; i < MR * TB; i++) acc[i] = 0.0f;
    for (int c0 = 0; c0 < chunks; c0 += 32) {
        __syncthreads(); /* the previous chunks have been read */
        for (int i = threadIdx.x; i < TB * 32; i += THREADS) {
            const int t = i / 32, c = i % 32;
            float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b = a;
            if (t0 + t < T && c0 + c < chunks) {
                const float4* xp = reinterpret_cast<const float4*>(X + (size_t)(t0 + t) * ldx) + 2 * (c0 + c);
                a = xp[0];
                b = xp[1];
            }
            lo[t][c] = a;
            hi[t][c] = b;
        }
        __syncthreads();
        const int c = c0 + lane;
        if (c < chunks) {
#pragma unroll
            for (int r = 0; r < MR; r++) {
                float wv[8];
                const int n = min(n0 + r, N - 1);
                load8<DT>(static_cast<const char*>(W) + (size_t)n * K * bytes, c, wv);
#pragma unroll
                for (int t = 0; t < TB; t++) {
                    const float4 a = lo[t][lane], b = hi[t][lane];
                    float s = acc[r * TB + t];
                    s = fmaf(wv[0], a.x, s);
                    s = fmaf(wv[1], a.y, s);
                    s = fmaf(wv[2], a.z, s);
                    s = fmaf(wv[3], a.w, s);
                    s = fmaf(wv[4], b.x, s);
                    s = fmaf(wv[5], b.y, s);
                    s = fmaf(wv[6], b.z, s);
                    s = fmaf(wv[7], b.w, s);
                    acc[r * TB + t] = s;
                }
            }
        }
    }
#pragma unroll
    for (int g = 0; g < MR * TB; g += 32) {
        float v[32];
#pragma unroll
        for (int i = 0; i < 32; i++) v[i] = g + i < MR * TB ? acc[g + i] : 0.0f;
#pragma unroll
        for (int s = 16, cnt = 32; s > 0; s >>= 1, cnt >>= 1) {
            const bool upper = lane & s;
#pragma unroll
            for (int i = 0; i < cnt / 2; i++) {
                const float send = upper ? v[i] : v[i + cnt / 2];
                const float keep = upper ? v[i + cnt / 2] : v[i];
                v[i] = keep + __shfl_xor_sync(0xffffffffu, send, s);
            }
        }
        const int idx = g + lane;
        if (idx < MR * TB) {
            const int r = idx / TB, t = idx % TB;
            if (t0 + t < T && n0 + r < N) {
                float* y = Y + (size_t)(t0 + t) * ldy + n0 + r;
                *y = add ? *y + v[0] : v[0];
            }
        }
    }
}

/* The half-split rotary embedding for T tokens at positions p0 onwards, as
 * rope in attn_in_kernel: every query head of row t of q, and every key head
 * of the cache's row p0 + t. */
__global__ void rope_rows_kernel(float* __restrict__ q, float* __restrict__ kc, const float* __restrict__ cos_all,
                                 const float* __restrict__ sin_all, int p0, int T, int n_heads, int n_kv_heads, int head_dim) {
    const int half = head_dim / 2;
    const int per_token = (n_heads + n_kv_heads) * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * per_token) return;
    const int t = idx / per_token, i = idx % per_token;
    const int h = i / half, j = i % half;
    const size_t pos = (size_t)p0 + t;
    const float c = cos_all[pos * half + j];
    const float s = sin_all[pos * half + j];
    float* v = h < n_heads ? q + (size_t)t * n_heads * head_dim + (size_t)h * head_dim
                           : kc + pos * n_kv_heads * head_dim + (size_t)(h - n_heads) * head_dim;
    const float a = v[j];
    const float b = v[j + half];
    v[j] = a * c - b * s;
    v[j + half] = b * c + a * s;
}

/* gate[i] = silu(gate[i]) * up[i] over n elements, as ops.c's vitna_silu_mul. */
__global__ void silu_mul_rows_kernel(float* __restrict__ gate, const float* __restrict__ up, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* Causal attention for a prompt's tokens, as flash attention computes it.
 * Block (b, kvh) takes key-value head kvh and WARPS of the chunk's tokens, a
 * warp a token: its rows are the RW = group query heads that read kvh
 * (repeat_kv). Positions go by in tiles of 32, whose keys and values the
 * block stages in shared memory once for every row, the values transposed.
 * In a tile, lane j scores position t0 + j against each row, masked after
 * the row's own position, and each row keeps its softmax as it goes: a
 * running max and sum, with the sum and the weighted values scaled by
 * exp(old max - new max) when the max moves. The tile's weights go to shared
 * memory, and lane l then adds dimensions l, l + 32, ... of each row's
 * weighted values, four positions at a time. That is model.c's softmax,
 * summed in another order; a row's sums run the same way whatever else the
 * block holds, so a token's result does not depend on the chunk it came in.
 * DS is head_dim / 32 rounded up: the dimensions a lane keeps. */
template <int RW, int DS>
__global__ void attention_prompt_kernel(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
                                        float* __restrict__ att, int p0, int T, int head_dim, int kv_dim, int q_dim, float scale) {
    extern __shared__ float4 shared4[];
    const int kvh = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kstride = head_dim + 4; /* padded, so that eight lanes reading eight rows sixteen bytes each hit distinct banks */
    float* ks = reinterpret_cast<float*>(shared4);   /* [32][kstride] */
    float* vt = ks + 32 * kstride;                   /* [head_dim][36]: values, transposed */
    float* pw = vt + head_dim * 36 + warp * RW * 32; /* [RW][32]: this warp's weights for the tile */
    const int tb = blockIdx.x * WARPS;               /* the block's first token */
    const int t = tb + warp;                         /* this warp's token */
    const int n = p0 + min(T, tb + WARPS);           /* the positions the block reads */
    const int c4 = head_dim / 4;
    /* The warp's token's position, or -1 past the chunk's last token, whose rows score nothing. */
    const int pos = t < T ? p0 + t : -1;
    const float* qw = q + (size_t)(t < T ? t : 0) * q_dim + (size_t)kvh * RW * head_dim;

    float m[RW], l[RW], o[RW][DS];
#pragma unroll
    for (int i = 0; i < RW; i++) {
        m[i] = NEG_INF;
        l[i] = 0.0f;
#pragma unroll
        for (int k = 0; k < DS; k++) o[i][k] = 0.0f;
    }

    for (int t0 = 0; t0 < n; t0 += 32) {
        const int len = min(32, n - t0);
        __syncthreads(); /* the previous tile has been read */
        for (int idx = threadIdx.x; idx < 32 * c4; idx += THREADS) {
            const int j = idx / c4, c = idx % c4;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vv = kv; /* zeros past the end: a weight of 0 times them is 0 */
            if (j < len) {
                const size_t src = (size_t)(t0 + j) * kv_dim + (size_t)kvh * head_dim + 4 * c;
                kv = *reinterpret_cast<const float4*>(kc + src);
                vv = *reinterpret_cast<const float4*>(vc + src);
            }
            *reinterpret_cast<float4*>(ks + j * kstride + 4 * c) = kv;
            vt[(4 * c) * 36 + j] = vv.x;
            vt[(4 * c + 1) * 36 + j] = vv.y;
            vt[(4 * c + 2) * 36 + j] = vv.z;
            vt[(4 * c + 3) * 36 + j] = vv.w;
        }
        __syncthreads();

        /* Scores: lane j against position t0 + j, every lane reading the same sixteen bytes of the queries at once. */
        float s[RW];
#pragma unroll
        for (int i = 0; i < RW; i++) s[i] = 0.0f;
        const float4* k4 = reinterpret_cast<const float4*>(ks + lane * kstride);
        for (int c = 0; c < c4; c++) {
            const float4 kv = k4[c];
#pragma unroll
            for (int i = 0; i < RW; i++) {
                const float4 qv = reinterpret_cast<const float4*>(qw + i * head_dim)[c];
                s[i] = fmaf(qv.x, kv.x, s[i]);
                s[i] = fmaf(qv.y, kv.y, s[i]);
                s[i] = fmaf(qv.z, kv.z, s[i]);
                s[i] = fmaf(qv.w, kv.w, s[i]);
            }
        }
        /* The running softmax. A row with nothing valid yet keeps a max of
         * minus infinity, and its scale is then 1, not exp(nan). */
        const bool valid = lane < len && t0 + lane <= pos;
#pragma unroll
        for (int i = 0; i < RW; i++) {
            const float sc = valid ? s[i] * scale : NEG_INF;
            const float mnew = fmaxf(m[i], warp_max(sc));
            const float e = valid ? expf(sc - mnew) : 0.0f;
            const float a = mnew == NEG_INF ? 1.0f : expf(m[i] - mnew); /* 0 at a row's first tile */
            l[i] = l[i] * a + warp_sum(e);
            m[i] = mnew;
#pragma unroll
            for (int k = 0; k < DS; k++) o[i][k] *= a;
            pw[i * 32 + lane] = e;
        }
        __syncwarp();
        /* The weighted values: lane l keeps dimensions l, l + 32, ..., four positions at a time. */
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            float4 v[DS];
#pragma unroll
            for (int k = 0; k < DS; k++) {
                const int d = lane + 32 * k;
                v[k] = d < head_dim ? *reinterpret_cast<const float4*>(vt + d * 36 + j) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
            for (int i = 0; i < RW; i++) {
                const float4 p = *reinterpret_cast<const float4*>(pw + i * 32 + j);
#pragma unroll
                for (int k = 0; k < DS; k++) {
                    o[i][k] = fmaf(p.x, v[k].x, o[i][k]);
                    o[i][k] = fmaf(p.y, v[k].y, o[i][k]);
                    o[i][k] = fmaf(p.z, v[k].z, o[i][k]);
                    o[i][k] = fmaf(p.w, v[k].w, o[i][k]);
                }
            }
        }
        __syncwarp(); /* the weights have been read before the next tile's are written */
    }
    if (pos < 0) return;
    float* out = att + (size_t)t * q_dim + (size_t)kvh * RW * head_dim;
#pragma unroll
    for (int i = 0; i < RW; i++) {
#pragma unroll
        for (int k = 0; k < DS; k++) {
            const int d = lane + 32 * k;
            if (d < head_dim) out[i * head_dim + d] = o[i][k] / l[i];
        }
    }
}

/* --- Launches --- */

/* KERNEL<DT><<<...>>>(...), for the dtype every weight matrix shares. */
#define LAUNCH(dtype, KERNEL, grid, shared, stream, ...)                                          \
    do {                                                                                          \
        switch (dtype) {                                                                          \
            case VITNA_DTYPE_BF16: KERNEL<VITNA_DTYPE_BF16><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
            case VITNA_DTYPE_F16: KERNEL<VITNA_DTYPE_F16><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break;   \
            default: KERNEL<VITNA_DTYPE_F32><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break;                \
        }                                                                                         \
    } while (0)

static unsigned int blocks_for(size_t n, size_t per_block) {
    return (unsigned int)((n + per_block - 1) / per_block);
}

/* Blocks for work done a warp per unit: one unit per warp, up to the cap,
 * past which each warp takes several. */
static unsigned int grid_for(const struct vitna_cuda_model* g, int units) {
    const unsigned int blocks = blocks_for((size_t)units, WARPS);
    return blocks < (unsigned int)g->max_blocks ? blocks : (unsigned int)g->max_blocks;
}

/* Attention's dynamic shared memory: the group's queries, and a slice's
 * scores at the longest. A slice is at most ATTENTION_MIN_SLICE positions
 * until every split is in use, and ctx / ATTENTION_SPLITS after. */
static size_t attention_shared(const struct vitna_cuda_model* g) {
    const size_t group = (size_t)(g->n_heads / g->n_kv_heads);
    size_t per = ((size_t)g->ctx + ATTENTION_SPLITS - 1) / ATTENTION_SPLITS;
    if (per < ATTENTION_MIN_SLICE) per = ATTENTION_MIN_SLICE;
    return group * ((size_t)g->head_dim + per) * sizeof(float);
}

/* The embedding and every layer, in model.c's order. Captured once, into g->body. */
static void enqueue_body(const struct vitna_cuda_model* g) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden, hd = g->head_dim, half = hd / 2;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const int group = g->n_heads / g->n_kv_heads;
    const size_t norm_shared = (size_t)H * sizeof(float);
    const unsigned int attn_in_blocks = grid_for(g, (g->n_heads + g->n_kv_heads) * half + kv_dim / 2);
    const unsigned int mlp_in_blocks = grid_for(g, g->intermediate);
    const unsigned int out_blocks = grid_for(g, H);
    const unsigned int embed_blocks = blocks_for((size_t)H, THREADS);

    LAUNCH(g->dtype, embed_kernel, embed_blocks, 0, s, g->embed.w, g->step, H, g->x);
    for (int l = 0; l < g->n_layers; l++) {
        const dlayer_t* L = &g->layers[l];
        float* kc = g->k_cache + (size_t)l * g->ctx * kv_dim;
        float* vc = g->v_cache + (size_t)l * g->ctx * kv_dim;

        LAUNCH(g->dtype, attn_in_kernel, attn_in_blocks, norm_shared, s, g->x, L->attn_norm, g->eps, L->q.w, L->k.w, L->v.w,
               g->q, kc, vc, g->cos_t, g->sin_t, g->step, H, g->n_heads, g->n_kv_heads, hd);
        attention_kernel<<<dim3(ATTENTION_SPLITS, g->n_kv_heads), THREADS, attention_shared(g), s>>>(
            g->q, kc, vc, g->att, g->part_m, g->part_l, g->part_o, g->done, g->step, hd, kv_dim, group, g->scale);
        LAUNCH(g->dtype, matvec_add_kernel, out_blocks, 0, s, L->o.w, g->att, g->x, H, q_dim);
        LAUNCH(g->dtype, mlp_in_kernel, mlp_in_blocks, norm_shared, s, g->x, L->mlp_norm, g->eps, L->gate.w, L->up.w, g->act, H, g->intermediate);
        LAUNCH(g->dtype, matvec_add_kernel, out_blocks, 0, s, L->down.w, g->act, g->x, H, g->intermediate);
    }
}

/* The final RMSNorm and the output projection of the row at x, into g->logits. */
static void enqueue_head_of(const struct vitna_cuda_model* g, const float* x) {
    LAUNCH(g->dtype, head_kernel, grid_for(g, g->vocab), (size_t)g->hidden * sizeof(float), g->stream,
           x, g->final_norm, g->eps, g->lm_head.w, g->logits, g->hidden, g->vocab);
}

/* The head of a step. Captured once, into g->head. */
static void enqueue_head(const struct vitna_cuda_model* g) {
    enqueue_head_of(g, g->x);
}

/* Y = X W^T for T rows, on the model's stream: gemm_few_kernel for up to
 * FEW_TOKENS rows, which a tile of GEMM_BM would mostly leave empty, and
 * gemm_kernel for more. */
static void gemm(const struct vitna_cuda_model* g, const float* X, int ldx, const void* W, float* Y, int ldy, int T, int N, int K,
                 int add) {
    if (T <= FEW_TOKENS) {
        const dim3 few(blocks_for((size_t)T, FEW_TB), blocks_for((size_t)N, WARPS * FEW_MR));
        switch (g->dtype) {
            case VITNA_DTYPE_BF16: gemm_few_kernel<VITNA_DTYPE_BF16, FEW_MR, FEW_TB><<<few, THREADS, 0, g->stream>>>(X, ldx, W, Y, ldy, T, N, K, add); break;
            case VITNA_DTYPE_F16: gemm_few_kernel<VITNA_DTYPE_F16, FEW_MR, FEW_TB><<<few, THREADS, 0, g->stream>>>(X, ldx, W, Y, ldy, T, N, K, add); break;
            default: gemm_few_kernel<VITNA_DTYPE_F32, FEW_MR, FEW_TB><<<few, THREADS, 0, g->stream>>>(X, ldx, W, Y, ldy, T, N, K, add); break;
        }
        return;
    }
    const dim3 grid(blocks_for((size_t)N, GEMM_BN), blocks_for((size_t)T, GEMM_BM));
    LAUNCH(g->dtype, gemm_kernel, grid, 0, g->stream, X, ldx, W, Y, ldy, T, N, K, add);
}

/* A prompt's attention's dynamic shared memory: a tile's keys, padded, its
 * values, transposed and padded, and each warp's weights for the tile. */
static size_t prompt_attention_shared(const struct vitna_cuda_model* g) {
    const size_t group = (size_t)(g->n_heads / g->n_kv_heads), hd = (size_t)g->head_dim;
    return (32 * (hd + 4) + hd * 36 + WARPS * group * 32) * sizeof(float);
}

/* Whether a prompt can run many tokens at once: widths the matrix-matrix
 * product's GEMM_BK divides, and an attention kernel for this shape, which
 * needs a key-value head's query heads to be at most PROMPT_ROWS, head_dim at
 * most 128, and a tile's keys, values and weights within 48 KB of shared
 * memory. A model that fails runs its prompts a token at a time. */
static bool prompt_batches(const struct vitna_cuda_model* g) {
    const int group = g->n_heads / g->n_kv_heads;
    return g->hidden % GEMM_BK == 0 && g->intermediate % GEMM_BK == 0 && (g->n_heads * g->head_dim) % GEMM_BK == 0 &&
           group <= PROMPT_ROWS && g->head_dim <= 128 && prompt_attention_shared(g) <= 48 * 1024;
}

/* A prompt's attention, for T tokens at positions p0 onwards: a warp a token. */
static void enqueue_prompt_attention(const struct vitna_cuda_model* g, const float* kc, const float* vc, int p0, int T) {
    const int hd = g->head_dim, group = g->n_heads / g->n_kv_heads;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const dim3 grid(blocks_for((size_t)T, WARPS), g->n_kv_heads);
    const size_t shared = prompt_attention_shared(g);
#define PROMPT_ATTENTION(RW, DS) \
    attention_prompt_kernel<RW, DS><<<grid, THREADS, shared, g->stream>>>(g->p_q, kc, vc, g->p_att, p0, T, hd, kv_dim, q_dim, g->scale)
#define PROMPT_ATTENTION_RW(RW)                  \
    switch ((hd + 31) / 32) {                    \
        case 1: PROMPT_ATTENTION(RW, 1); break;  \
        case 2: PROMPT_ATTENTION(RW, 2); break;  \
        case 3: PROMPT_ATTENTION(RW, 3); break;  \
        default: PROMPT_ATTENTION(RW, 4); break; \
    }
    switch (group) {
        case 1: PROMPT_ATTENTION_RW(1); break;
        case 2: PROMPT_ATTENTION_RW(2); break;
        case 3: PROMPT_ATTENTION_RW(3); break;
        case 4: PROMPT_ATTENTION_RW(4); break;
        case 5: PROMPT_ATTENTION_RW(5); break;
        case 6: PROMPT_ATTENTION_RW(6); break;
        case 7: PROMPT_ATTENTION_RW(7); break;
        default: PROMPT_ATTENTION_RW(8); break;
    }
#undef PROMPT_ATTENTION_RW
#undef PROMPT_ATTENTION
}

/* The embedding and every layer for T tokens of a prompt, g->p_tokens[c0]
 * onwards, at positions p0 onwards, in model.c's order, leaving the last
 * layer's output in g->p_x. Their keys and values go into the cache. */
static void enqueue_chunk(const struct vitna_cuda_model* g, size_t c0, int p0, int T) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden, I = g->intermediate, hd = g->head_dim;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const unsigned int rope_blocks = blocks_for((size_t)T * (g->n_heads + g->n_kv_heads) * (hd / 2), THREADS);
    const unsigned int silu_blocks = blocks_for((size_t)T * I, THREADS);

    LAUNCH(g->dtype, embed_rows_kernel, dim3(blocks_for((size_t)H, THREADS), T), 0, s, g->embed.w, g->p_tokens + c0, H, g->p_x);
    for (int l = 0; l < g->n_layers; l++) {
        const dlayer_t* L = &g->layers[l];
        float* kc = g->k_cache + (size_t)l * g->ctx * kv_dim;
        float* vc = g->v_cache + (size_t)l * g->ctx * kv_dim;

        rmsnorm_rows_kernel<<<T, THREADS, 0, s>>>(g->p_x, L->attn_norm, g->p_xn, H, g->eps);
        gemm(g, g->p_xn, H, L->q.w, g->p_q, q_dim, T, q_dim, H, 0);
        gemm(g, g->p_xn, H, L->k.w, kc + (size_t)p0 * kv_dim, kv_dim, T, kv_dim, H, 0);
        gemm(g, g->p_xn, H, L->v.w, vc + (size_t)p0 * kv_dim, kv_dim, T, kv_dim, H, 0);
        rope_rows_kernel<<<rope_blocks, THREADS, 0, s>>>(g->p_q, kc, g->cos_t, g->sin_t, p0, T, g->n_heads, g->n_kv_heads, hd);
        enqueue_prompt_attention(g, kc, vc, p0, T);
        gemm(g, g->p_att, q_dim, L->o.w, g->p_x, H, T, H, q_dim, 1);
        rmsnorm_rows_kernel<<<T, THREADS, 0, s>>>(g->p_x, L->mlp_norm, g->p_xn, H, g->eps);
        gemm(g, g->p_xn, H, L->gate.w, g->p_gate, I, T, I, H, 0);
        gemm(g, g->p_xn, H, L->up.w, g->p_up, I, T, I, H, 0);
        silu_mul_rows_kernel<<<silu_blocks, THREADS, 0, s>>>(g->p_gate, g->p_up, (size_t)T * I);
        gemm(g, g->p_gate, I, L->down.w, g->p_x, H, T, H, I, 1);
    }
}

static cudaError_t capture(struct vitna_cuda_model* g, void (*enqueue)(const struct vitna_cuda_model*), cudaGraphExec_t* out) {
    cudaGraph_t graph = NULL;
    cudaError_t e = cudaStreamBeginCapture(g->stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) return e;
    enqueue(g);
    e = cudaStreamEndCapture(g->stream, &graph);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaGraphInstantiate(out, graph, 0);
    if (graph) cudaGraphDestroy(graph);
    return e;
}

/* --- Host side --- */

/* Every piece of the arena starts on a 256-byte boundary, as cudaMalloc's own blocks do. */
static size_t padded(size_t bytes) {
    return (bytes + 255) & ~(size_t)255;
}

typedef struct {
    unsigned char* next;
} carver_t;

static void* carve(carver_t* c, size_t bytes) {
    void* p = c->next;
    c->next += padded(bytes);
    return p;
}

static size_t matrix_bytes(const vitna_matrix_t* m) {
    return m->rows * m->cols * vitna_dtype_size(m->dtype);
}

static bool fits_int(size_t v) {
    return v <= (size_t)INT_MAX;
}

bool vitna_cuda_probe(char* err, size_t err_len) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "no usable CUDA device; cudaGetDeviceCount reports", e);
    if (count < 1) return fail(err, err_len, "the CUDA runtime found no device");
    return true;
}

void vitna_cuda_free(struct vitna_cuda_model* g) {
    if (!g) return;
    cudaSetDevice(g->device);
    if (g->stream) cudaStreamSynchronize(g->stream);
    if (g->body) cudaGraphExecDestroy(g->body);
    if (g->head) cudaGraphExecDestroy(g->head);
    if (g->stream) cudaStreamDestroy(g->stream);
    if (g->host_logits) cudaFreeHost(g->host_logits);
    if (g->arena) cudaFree(g->arena);
    free(g->layers);
    free(g);
}

const char* vitna_cuda_device_name(const struct vitna_cuda_model* g) {
    return g ? g->name : "";
}

/* Whether the GPU kernels can take the model: every matrix in the dtype of
 * the embedding, and widths the 16-byte reads divide. */
static bool check_shapes(const vitna_llama_t* m, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    const vitna_dtype_t dt = m->embed.dtype;
    if (c->hidden % 8 != 0 || c->intermediate % 8 != 0 || (c->n_heads * c->head_dim) % 8 != 0) {
        return fail(err, err_len, "the GPU path needs the hidden, intermediate and attention widths to be multiples of 8");
    }
    if (c->hidden * sizeof(float) > 48 * 1024) {
        return fail(err, err_len, "the GPU path needs hidden to be at most 12288");
    }
    bool same = m->lm_head.dtype == dt;
    for (size_t l = 0; same && l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        same = L->q.dtype == dt && L->k.dtype == dt && L->v.dtype == dt && L->o.dtype == dt &&
               L->gate.dtype == dt && L->up.dtype == dt && L->down.dtype == dt;
    }
    if (!same) return fail(err, err_len, "the GPU path needs every weight matrix in one dtype, and this model mixes them");
    return true;
}

struct vitna_cuda_model* vitna_cuda_create(const vitna_llama_t* m, const float* cos_tab, const float* sin_tab, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    if (!vitna_cuda_probe(err, err_len)) return NULL;
    if (c->head_dim > THREADS || c->head_dim % 4 != 0) {
        fail(err, err_len, "head_dim %zu is not supported on the GPU: it must be a multiple of 4 and at most %d", c->head_dim, THREADS);
        return NULL;
    }
    if (!check_shapes(m, err, err_len)) return NULL;
    const size_t q_dim = c->n_heads * c->head_dim;
    const size_t kv_dim = c->n_kv_heads * c->head_dim;
    const size_t half = c->head_dim / 2;
    const size_t cache_floats = c->n_layers * m->ctx * kv_dim;
    if (!fits_int(c->vocab) || !fits_int(c->hidden) || !fits_int(c->intermediate) || !fits_int(q_dim) ||
        !fits_int(m->ctx) || !fits_int(c->n_layers)) {
        fail(err, err_len, "the model's dimensions are too large for the GPU path");
        return NULL;
    }

    struct vitna_cuda_model* g = (struct vitna_cuda_model*)calloc(1, sizeof(*g));
    if (g) g->layers = (dlayer_t*)calloc(c->n_layers, sizeof(dlayer_t));
    if (!g || !g->layers) {
        free(g);
        fail(err, err_len, "out of memory");
        return NULL;
    }
    g->device = 0;
    g->n_layers = (int)c->n_layers;
    g->hidden = (int)c->hidden;
    g->intermediate = (int)c->intermediate;
    g->n_heads = (int)c->n_heads;
    g->n_kv_heads = (int)c->n_kv_heads;
    g->head_dim = (int)c->head_dim;
    g->vocab = (int)c->vocab;
    g->ctx = (int)m->ctx;
    g->eps = c->rms_eps;
    g->scale = 1.0f / sqrtf((float)c->head_dim); /* as model.c */
    g->dtype = m->embed.dtype;
    g->prompt = prompt_batches(g);

    cudaError_t e = cudaSetDevice(g->device);
    cudaDeviceProp prop;
    if (e == cudaSuccess) e = cudaGetDeviceProperties(&prop, g->device);
    if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaMallocHost((void**)&g->host_logits, c->vocab * sizeof(float));
    if (e != cudaSuccess) {
        fail_cuda(err, err_len, "cannot use CUDA device 0", e);
        vitna_cuda_free(g);
        return NULL;
    }
    snprintf(g->name, sizeof(g->name), "%s (sm_%d%d)", prop.name, prop.major, prop.minor);
    g->max_blocks = 8 * prop.multiProcessorCount;
    if (attention_shared(g) > 48 * 1024) {
        fail(err, err_len, "the GPU path's attention needs --ctx at most %d for this model", (int)(((48 * 1024 / sizeof(float)) / (g->n_heads / g->n_kv_heads) - g->head_dim) * ATTENTION_SPLITS));
        vitna_cuda_free(g);
        return NULL;
    }

    /* One allocation: the weights, the rotary tables, the key-value cache and the scratch. */
    const bool tied = m->lm_head.data == m->embed.data;
    size_t total = 0;
    for (size_t l = 0; l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        const vitna_matrix_t* mats[7] = { &L->q, &L->k, &L->v, &L->o, &L->gate, &L->up, &L->down };
        for (int i = 0; i < 7; i++) total += padded(matrix_bytes(mats[i]));
        total += 2 * padded(c->hidden * sizeof(float));
    }
    total += padded(matrix_bytes(&m->embed));
    if (!tied) total += padded(matrix_bytes(&m->lm_head));
    total += padded(c->hidden * sizeof(float));
    total += 2 * padded(m->ctx * half * sizeof(float));
    total += 2 * padded(cache_floats * sizeof(float));
    const size_t slices = c->n_heads * ATTENTION_SPLITS;
    total += padded(c->hidden * sizeof(float)) + 2 * padded(q_dim * sizeof(float)) + padded(c->intermediate * sizeof(float)) +
             padded(c->vocab * sizeof(float)) + 2 * padded(slices * sizeof(float)) + padded(slices * c->head_dim * sizeof(float)) +
             padded(c->n_kv_heads * sizeof(unsigned int)) + padded(sizeof(step_t));
    /* A prompt's chunk, its logits and its tokens. */
    size_t P = 0;
    if (g->prompt) {
        const size_t per_token = (2 * c->hidden + 2 * q_dim + 2 * c->intermediate) * sizeof(float);
        P = PREFILL_BYTES / per_token;
        if (P > PREFILL_MAX) P = PREFILL_MAX;
        if (P < PREFILL_MIN) P = PREFILL_MIN;
        if (P > m->ctx) P = m->ctx;
    }
    g->prefill = P;
    if (g->prompt) total += 2 * padded(P * c->hidden * sizeof(float)) + 2 * padded(P * q_dim * sizeof(float)) +
             2 * padded(P * c->intermediate * sizeof(float)) + padded((size_t)LOGIT_ROWS * c->vocab * sizeof(float)) +
             padded(m->ctx * sizeof(int32_t));

    e = cudaMalloc(&g->arena, total);
    if (e != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaGetLastError(); /* clear the allocation error before asking */
        cudaMemGetInfo(&free_b, &total_b);
        fail(err, err_len, "the model needs %.1f MiB on %s, and %.1f MiB of its %.1f MiB are free (%s); a smaller --ctx needs less",
             total / 1048576.0, g->name, free_b / 1048576.0, total_b / 1048576.0, cudaGetErrorName(e));
        g->arena = NULL;
        vitna_cuda_free(g);
        return NULL;
    }

    carver_t cv = { (unsigned char*)g->arena };
    bool ok = true;
#define UPLOAD(dst, src, bytes) \
    (ok && (e = cudaMemcpy((void*)(dst), (src), (bytes), cudaMemcpyHostToDevice)) == cudaSuccess)
    /* A matrix keeps its stored dtype on the device. */
#define UPLOAD_MATRIX(dm, hm)                                                                   \
    do {                                                                                        \
        (dm).rows = (int)(hm).rows;                                                             \
        (dm).cols = (int)(hm).cols;                                                             \
        (dm).w = carve(&cv, matrix_bytes(&(hm)));                                               \
        ok = UPLOAD((dm).w, (hm).data, matrix_bytes(&(hm)));                                    \
    } while (0)
#define UPLOAD_FLOATS(dst, src, n)                                                              \
    do {                                                                                        \
        float* d_ = (float*)carve(&cv, (n) * sizeof(float));                                    \
        (dst) = d_;                                                                             \
        ok = UPLOAD(d_, (src), (n) * sizeof(float));                                            \
    } while (0)

    for (size_t l = 0; ok && l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        dlayer_t* D = &g->layers[l];
        UPLOAD_MATRIX(D->q, L->q);
        UPLOAD_MATRIX(D->k, L->k);
        UPLOAD_MATRIX(D->v, L->v);
        UPLOAD_MATRIX(D->o, L->o);
        UPLOAD_MATRIX(D->gate, L->gate);
        UPLOAD_MATRIX(D->up, L->up);
        UPLOAD_MATRIX(D->down, L->down);
        UPLOAD_FLOATS(D->attn_norm, L->attn_norm, c->hidden);
        UPLOAD_FLOATS(D->mlp_norm, L->mlp_norm, c->hidden);
    }
    UPLOAD_MATRIX(g->embed, m->embed);
    if (tied) {
        g->lm_head = g->embed;
    } else {
        UPLOAD_MATRIX(g->lm_head, m->lm_head);
    }
    UPLOAD_FLOATS(g->final_norm, m->final_norm, c->hidden);
    UPLOAD_FLOATS(g->cos_t, cos_tab, m->ctx * half);
    UPLOAD_FLOATS(g->sin_t, sin_tab, m->ctx * half);
#undef UPLOAD_FLOATS
#undef UPLOAD_MATRIX
#undef UPLOAD

    if (ok) {
        g->k_cache = (float*)carve(&cv, cache_floats * sizeof(float));
        g->v_cache = (float*)carve(&cv, cache_floats * sizeof(float));
        g->x = (float*)carve(&cv, c->hidden * sizeof(float));
        g->q = (float*)carve(&cv, q_dim * sizeof(float));
        g->att = (float*)carve(&cv, q_dim * sizeof(float));
        g->act = (float*)carve(&cv, c->intermediate * sizeof(float));
        g->logits = (float*)carve(&cv, c->vocab * sizeof(float));
        g->part_m = (float*)carve(&cv, slices * sizeof(float));
        g->part_l = (float*)carve(&cv, slices * sizeof(float));
        g->part_o = (float*)carve(&cv, slices * c->head_dim * sizeof(float));
        g->done = (unsigned int*)carve(&cv, c->n_kv_heads * sizeof(unsigned int));
        g->step = (step_t*)carve(&cv, sizeof(step_t));
        if (g->prompt) {
            g->p_x = (float*)carve(&cv, P * c->hidden * sizeof(float));
            g->p_xn = (float*)carve(&cv, P * c->hidden * sizeof(float));
            g->p_q = (float*)carve(&cv, P * q_dim * sizeof(float));
            g->p_att = (float*)carve(&cv, P * q_dim * sizeof(float));
            g->p_gate = (float*)carve(&cv, P * c->intermediate * sizeof(float));
            g->p_up = (float*)carve(&cv, P * c->intermediate * sizeof(float));
            g->p_logits = (float*)carve(&cv, (size_t)LOGIT_ROWS * c->vocab * sizeof(float));
            g->p_tokens = (int32_t*)carve(&cv, m->ctx * sizeof(int32_t));
        }
        /* A position is always written before it is read; zeros make a mistake there repeatable. */
        e = cudaMemset(g->k_cache, 0, 2 * padded(cache_floats * sizeof(float)));
        /* Attention counts finished slices up from zero, and leaves the count at zero. */
        if (e == cudaSuccess) e = cudaMemset(g->done, 0, c->n_kv_heads * sizeof(unsigned int));
        /* The uploads and the memset ran on the default stream, which the
         * model's non-blocking stream does not wait for: finish them first. */
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        ok = e == cudaSuccess;
    }
    if (!ok) {
        fail_cuda(err, err_len, "copying the model to the device failed", e);
        vitna_cuda_free(g);
        return NULL;
    }

    e = capture(g, enqueue_body, &g->body);
    if (e == cudaSuccess) e = capture(g, enqueue_head, &g->head);
    if (e != cudaSuccess) {
        fail_cuda(err, err_len, "capturing the forward pass as a CUDA graph failed", e);
        vitna_cuda_free(g);
        return NULL;
    }

    /* Run the graphs once, at position 0, so a build with no code for this
     * GPU fails on load, not at the first token. What it writes to position
     * 0 of the cache is written again before it is read. */
    set_step_kernel<<<1, 1, 0, g->stream>>>(g->step, 0, 0);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaGraphLaunch(g->body, g->stream);
    if (e == cudaSuccess) e = cudaGraphLaunch(g->head, g->stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    if (e != cudaSuccess) {
        if (e == cudaErrorNoKernelImageForDevice || e == cudaErrorUnsupportedPtxVersion) {
            fail(err, err_len, "this engine has no code for %s; rebuild it for that architecture (%s)", g->name, cudaGetErrorName(e));
        } else {
            fail_cuda(err, err_len, "a first run on the device failed", e);
        }
        vitna_cuda_free(g);
        return NULL;
    }
    return g;
}

bool vitna_cuda_step(struct vitna_cuda_model* g, int32_t token, size_t pos, float* logits, char* err, size_t err_len) {
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);

    set_step_kernel<<<1, 1, 0, g->stream>>>(g->step, token, (int)pos);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaGraphLaunch(g->body, g->stream);
    if (e == cudaSuccess && logits) e = cudaGraphLaunch(g->head, g->stream);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass could not be launched", e);
    if (logits) {
        const size_t bytes = (size_t)g->vocab * sizeof(float);
        e = cudaMemcpyAsync(g->host_logits, g->logits, bytes, cudaMemcpyDeviceToHost, g->stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
        if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
        memcpy(logits, g->host_logits, bytes);
    }
    return true;
}

/* The logits of the n rows at x (n at most LOGIT_ROWS), copied to out, n x vocab. */
static cudaError_t logits_rows(struct vitna_cuda_model* g, const float* x, int n, float* out) {
    rmsnorm_rows_kernel<<<n, THREADS, 0, g->stream>>>(x, g->final_norm, g->p_xn, g->hidden, g->eps);
    gemm(g, g->p_xn, g->hidden, g->lm_head.w, g->p_logits, g->vocab, n, g->vocab, g->hidden, 0);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) {
        e = cudaMemcpyAsync(out, g->p_logits, (size_t)n * g->vocab * sizeof(float), cudaMemcpyDeviceToHost, g->stream);
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    return e;
}

size_t vitna_cuda_prompt_min(const struct vitna_cuda_model* g) {
    return g->prompt ? PROMPT_MIN : SIZE_MAX;
}

bool vitna_cuda_steps(struct vitna_cuda_model* g, const int32_t* tokens, size_t count, size_t pos, float* logits, size_t rows,
                      char* err, size_t err_len) {
    if (!g->prompt) return fail(err, err_len, "this model's prompts cannot run many tokens at once on the GPU");
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    if (!logits) rows = 0;
    const size_t first = count - rows; /* the first position whose logits are wanted */

    /* From pageable memory, so tokens may change as soon as this returns. */
    e = cudaMemcpyAsync(g->p_tokens, tokens, count * sizeof(int32_t), cudaMemcpyHostToDevice, g->stream);
    int T = 0;
    for (size_t c0 = 0; e == cudaSuccess && c0 < count; c0 += g->prefill) {
        T = (int)(count - c0 < g->prefill ? count - c0 : g->prefill);
        enqueue_chunk(g, c0, (int)(pos + c0), T);
        e = cudaGetLastError();
        /* The wanted logits among this chunk's rows, when more than the last are wanted. */
        for (size_t r = first > c0 ? first : c0; e == cudaSuccess && rows > 1 && r < c0 + T; r += LOGIT_ROWS) {
            const size_t n = c0 + T - r < LOGIT_ROWS ? c0 + T - r : LOGIT_ROWS;
            e = logits_rows(g, g->p_x + (r - c0) * g->hidden, (int)n, logits + (r - first) * g->vocab);
        }
    }
    /* The next token's logits alone come from the head a step uses, on the last row. */
    if (e == cudaSuccess && rows == 1) {
        const size_t bytes = (size_t)g->vocab * sizeof(float);
        enqueue_head_of(g, g->p_x + (size_t)(T - 1) * g->hidden);
        e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaMemcpyAsync(g->host_logits, g->logits, bytes, cudaMemcpyDeviceToHost, g->stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
        if (e == cudaSuccess) memcpy(logits, g->host_logits, bytes);
    }
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
    return true;
}
