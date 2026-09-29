/**
 * model_cuda.cu - The forward pass of model.c on an NVIDIA GPU, in float32.
 *
 * The computation is model.c's vitna_llama_step, kernel for kernel: the
 * token's embedding row, then per layer RMSNorm, the query, key and value
 * projections, the half-split rotary embedding, attention over the key-value
 * cache, the output projection added to the residual, RMSNorm, the gate and
 * up projections, SiLU(gate) * up and the down projection added to the
 * residual; then the final RMSNorm and the output projection.
 *
 * Weights are uploaded once, in their stored dtype (F32, BF16 or F16), and
 * widened to float32 as they are read, as ops.c does. The key-value cache is
 * float32 on the device, in model.c's layout, [layer][position][key-value
 * head * head_dim], and a token's keys and values are written straight into
 * their slot. The rotary cos and sin of every position come from model.c,
 * computed on the host the way the CPU path computes them.
 *
 * Arithmetic is float32 on CUDA cores. Nothing here uses tensor cores, so
 * TF32 never applies, and the build does not pass --use_fast_math, so
 * division, square root and expf keep their accurate forms. nvcc's default
 * fused multiply-add stays on; the CPU path's AVX2 matvec uses FMA too.
 *
 * A token is a replay of CUDA graphs, not a launch of every kernel. After the
 * upload, the kernels for the embedding and every layer are captured into one
 * graph, and the final RMSNorm and output projection into a second. The token
 * and its position reach the kernels through a small struct on the device,
 * which one kernel writes before each replay, so the same graphs serve every
 * position. Everything runs in order on the model's stream. A step that
 * returns no logits does not wait for the device; one that does waits for the
 * copy back.
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

/* Minus infinity, from its bits: MSVC's INFINITY macro overflows a constant to get it. */
#define NEG_INF __uint_as_float(0xff800000u)

typedef struct {
    const void* w;
    vitna_dtype_t dtype;
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

    int n_layers, hidden, intermediate, n_heads, n_kv_heads, head_dim, vocab, ctx;
    float eps, scale;

    void* arena;            /* one allocation, carved into everything below */
    dlayer_t* layers;       /* host array of device pointers */
    dmat_t embed, lm_head;
    const float* final_norm;
    const float* cos_t;     /* [ctx][head_dim / 2] */
    const float* sin_t;
    float* k_cache;         /* [n_layers][ctx][n_kv_heads * head_dim] */
    float* v_cache;
    float *x, *xn, *q, *att, *gate, *up, *logits, *scores;
    step_t* step;

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

/* Element i of a weight matrix, widened to float32 exactly. */
template <vitna_dtype_t DT>
__device__ __forceinline__ float widen(const void* w, size_t i);

template <>
__device__ __forceinline__ float widen<VITNA_DTYPE_F32>(const void* w, size_t i) {
    return static_cast<const float*>(w)[i];
}

template <>
__device__ __forceinline__ float widen<VITNA_DTYPE_BF16>(const void* w, size_t i) {
    /* bfloat16 is the high half of a float32, as in vitna_bf16_to_f32. */
    return __uint_as_float(static_cast<unsigned int>(static_cast<const unsigned short*>(w)[i]) << 16);
}

template <>
__device__ __forceinline__ float widen<VITNA_DTYPE_F16>(const void* w, size_t i) {
    /* Every half is exactly a float32; the conversion keeps subnormals, as vitna_f16_to_f32 does. */
    const unsigned short h = static_cast<const unsigned short*>(w)[i];
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
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

/* y = x / sqrt(mean(x^2) + eps) * w, as kernels.c's vitna_rmsnorm. One block. */
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n, float eps) {
    __shared__ float red[WARPS];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += THREADS) ss = fmaf(x[i], x[i], ss);
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += THREADS) y[i] = x[i] * inv * w[i];
}

/* y = W x, one warp per row of W. With add, the product is added to y
 * instead of replacing it: the residual add after a projection. With
 * per_pos, y is the position's slot, y + pos * per_pos: a row of the cache. */
template <vitna_dtype_t DT>
__global__ void matvec_kernel(const void* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                              int rows, int cols, int add, const step_t* __restrict__ step, int per_pos) {
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return; /* the whole warp leaves together */
    if (per_pos) y += (size_t)step->pos * per_pos;
    const size_t base = (size_t)row * cols;
    float s = 0.0f;
    for (int c = lane; c < cols; c += 32) s = fmaf(widen<DT>(w, base + c), x[c], s);
    s = warp_sum(s);
    if (lane == 0) y[row] = add ? y[row] + s : s;
}

/* matvec_kernel for BF16 rows that are a whole number of 16-byte chunks:
 * each lane reads eight weights at a time instead of one. The weights are
 * widened exactly as widen<VITNA_DTYPE_BF16> does; in a 32-bit word the
 * lower-addressed bfloat16 is the low half. */
__global__ void matvec_bf16x8_kernel(const uint4* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                     int rows, int cols, int add, const step_t* __restrict__ step, int per_pos) {
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    if (per_pos) y += (size_t)step->pos * per_pos;
    const int chunks = cols >> 3;
    const uint4* wr = w + (size_t)row * chunks;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float s = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 p = wr[c];
        const float4 a = x4[2 * c];
        const float4 b = x4[2 * c + 1];
        s = fmaf(__uint_as_float(p.x << 16), a.x, s);
        s = fmaf(__uint_as_float(p.x & 0xffff0000u), a.y, s);
        s = fmaf(__uint_as_float(p.y << 16), a.z, s);
        s = fmaf(__uint_as_float(p.y & 0xffff0000u), a.w, s);
        s = fmaf(__uint_as_float(p.z << 16), b.x, s);
        s = fmaf(__uint_as_float(p.z & 0xffff0000u), b.y, s);
        s = fmaf(__uint_as_float(p.w << 16), b.z, s);
        s = fmaf(__uint_as_float(p.w & 0xffff0000u), b.w, s);
    }
    s = warp_sum(s);
    if (lane == 0) y[row] = add ? y[row] + s : s;
}

/* The half-split rotary embedding of ops.c's vitna_rope_half, element j
 * paired with j + head_dim / 2, on every query head and on the position's
 * key heads in the cache kc. */
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ kc, const float* __restrict__ cos_all,
                            const float* __restrict__ sin_all, const step_t* __restrict__ step,
                            int n_heads, int n_kv_heads, int head_dim) {
    const int half = head_dim / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (n_heads + n_kv_heads) * half) return;
    const size_t pos = (size_t)step->pos;
    const float* cos_t = cos_all + pos * half;
    const float* sin_t = sin_all + pos * half;
    float* k = kc + pos * n_kv_heads * head_dim;
    const int h = i / half, j = i % half;
    float* v = h < n_heads ? q + (size_t)h * head_dim : k + (size_t)(h - n_heads) * head_dim;
    const float a = v[j];
    const float b = v[j + half];
    v[j] = a * cos_t[j] - b * sin_t[j];
    v[j + half] = b * cos_t[j] + a * sin_t[j];
}

/* Attention for one query head per block, as model.c's loop over heads:
 * scores against the cached keys of positions 0 to pos, a softmax, and the
 * weighted sum of the cached values. Query head h reads key-value head
 * h / group (repeat_kv). scores has room for ctx floats per head. */
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
                                 float* __restrict__ att, float* __restrict__ scores, const step_t* __restrict__ step,
                                 int head_dim, int kv_dim, int group, int ctx, float scale) {
    __shared__ float red[WARPS];
    __shared__ float part[THREADS];
    const int n = step->pos + 1;
    const int h = blockIdx.x, kvh = h / group;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* qh = q + (size_t)h * head_dim;
    const float* kh = kc + (size_t)kvh * head_dim;
    const float* vh = vc + (size_t)kvh * head_dim;
    float* s = scores + (size_t)h * ctx;

    /* Scores: a warp per position, its lanes across the head's dimensions. */
    for (int t = warp; t < n; t += WARPS) {
        const float* kt = kh + (size_t)t * kv_dim;
        float d = 0.0f;
        for (int i = lane; i < head_dim; i += 32) d = fmaf(qh[i], kt[i], d);
        d = warp_sum(d);
        if (lane == 0) s[t] = d * scale;
    }
    __syncthreads();

    float mx = NEG_INF;
    for (int t = threadIdx.x; t < n; t += THREADS) mx = fmaxf(mx, s[t]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int t = threadIdx.x; t < n; t += THREADS) {
        const float e = expf(s[t] - mx);
        s[t] = e;
        sum += e;
    }
    sum = block_sum(sum, red); /* its barriers also publish every s[t] written above */
    const float inv = 1.0f / sum;

    /* Values: thread (g, i) sums positions g, g + groups, ... of dimension i,
     * then the groups' partial sums are added. */
    const int groups = THREADS / head_dim;
    const int g = threadIdx.x / head_dim, i = threadIdx.x % head_dim;
    float acc = 0.0f;
    if (g < groups) {
        for (int t = g; t < n; t += groups) acc = fmaf(s[t] * inv, vh[(size_t)t * kv_dim + i], acc);
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x < head_dim) {
        float o = 0.0f;
        for (int k = 0; k < groups; k++) o += part[k * head_dim + threadIdx.x];
        att[(size_t)h * head_dim + threadIdx.x] = o;
    }
}

/* gate[i] = silu(gate[i]) * up[i], with silu(g) = g / (1 + exp(-g)), as ops.c's vitna_silu_mul. */
__global__ void silu_mul_kernel(float* __restrict__ gate, const float* __restrict__ up, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* --- Launches --- */

static unsigned int blocks_for(size_t n, size_t per_block) {
    return (unsigned int)((n + per_block - 1) / per_block);
}

static void matvec(cudaStream_t s, const dmat_t* m, const float* x, float* y, int add, const step_t* step, int per_pos) {
    const unsigned int blocks = blocks_for((size_t)m->rows, WARPS);
    /* Every matrix starts on a 256-byte boundary in the arena, so rows of a
     * multiple of eight weights stay 16-byte aligned, and so do the vectors. */
    if (m->dtype == VITNA_DTYPE_BF16 && m->cols % 8 == 0) {
        matvec_bf16x8_kernel<<<blocks, THREADS, 0, s>>>(static_cast<const uint4*>(m->w), x, y, m->rows, m->cols, add, step, per_pos);
        return;
    }
    switch (m->dtype) {
        case VITNA_DTYPE_BF16: matvec_kernel<VITNA_DTYPE_BF16><<<blocks, THREADS, 0, s>>>(m->w, x, y, m->rows, m->cols, add, step, per_pos); break;
        case VITNA_DTYPE_F16: matvec_kernel<VITNA_DTYPE_F16><<<blocks, THREADS, 0, s>>>(m->w, x, y, m->rows, m->cols, add, step, per_pos); break;
        default: matvec_kernel<VITNA_DTYPE_F32><<<blocks, THREADS, 0, s>>>(m->w, x, y, m->rows, m->cols, add, step, per_pos); break;
    }
}

static void embed(cudaStream_t s, const dmat_t* table, const step_t* step, float* x) {
    const unsigned int blocks = blocks_for((size_t)table->cols, THREADS);
    switch (table->dtype) {
        case VITNA_DTYPE_BF16: embed_kernel<VITNA_DTYPE_BF16><<<blocks, THREADS, 0, s>>>(table->w, step, table->cols, x); break;
        case VITNA_DTYPE_F16: embed_kernel<VITNA_DTYPE_F16><<<blocks, THREADS, 0, s>>>(table->w, step, table->cols, x); break;
        default: embed_kernel<VITNA_DTYPE_F32><<<blocks, THREADS, 0, s>>>(table->w, step, table->cols, x); break;
    }
}

/* The embedding and every layer, in model.c's order. Captured once, into g->body. */
static void enqueue_body(const struct vitna_cuda_model* g) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden, hd = g->head_dim, half = hd / 2;
    const int kv_dim = g->n_kv_heads * hd;
    const int group = g->n_heads / g->n_kv_heads;
    const unsigned int rope_blocks = blocks_for((size_t)(g->n_heads + g->n_kv_heads) * half, THREADS);
    const unsigned int mlp_blocks = blocks_for((size_t)g->intermediate, THREADS);

    embed(s, &g->embed, g->step, g->x);
    for (int l = 0; l < g->n_layers; l++) {
        const dlayer_t* L = &g->layers[l];
        float* kc = g->k_cache + (size_t)l * g->ctx * kv_dim;
        float* vc = g->v_cache + (size_t)l * g->ctx * kv_dim;

        /* Attention */
        rmsnorm_kernel<<<1, THREADS, 0, s>>>(g->x, L->attn_norm, g->xn, H, g->eps);
        matvec(s, &L->q, g->xn, g->q, 0, g->step, 0);
        matvec(s, &L->k, g->xn, kc, 0, g->step, kv_dim);
        matvec(s, &L->v, g->xn, vc, 0, g->step, kv_dim);
        rope_kernel<<<rope_blocks, THREADS, 0, s>>>(g->q, kc, g->cos_t, g->sin_t, g->step, g->n_heads, g->n_kv_heads, hd);
        attention_kernel<<<g->n_heads, THREADS, 0, s>>>(g->q, kc, vc, g->att, g->scores, g->step, hd, kv_dim, group, g->ctx, g->scale);
        matvec(s, &L->o, g->att, g->x, 1, g->step, 0);

        /* MLP */
        rmsnorm_kernel<<<1, THREADS, 0, s>>>(g->x, L->mlp_norm, g->xn, H, g->eps);
        matvec(s, &L->gate, g->xn, g->gate, 0, g->step, 0);
        matvec(s, &L->up, g->xn, g->up, 0, g->step, 0);
        silu_mul_kernel<<<mlp_blocks, THREADS, 0, s>>>(g->gate, g->up, g->intermediate);
        matvec(s, &L->down, g->gate, g->x, 1, g->step, 0);
    }
}

/* The final RMSNorm and the output projection. Captured once, into g->head. */
static void enqueue_head(const struct vitna_cuda_model* g) {
    rmsnorm_kernel<<<1, THREADS, 0, g->stream>>>(g->x, g->final_norm, g->xn, g->hidden, g->eps);
    matvec(g->stream, &g->lm_head, g->xn, g->logits, 0, g->step, 0);
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

struct vitna_cuda_model* vitna_cuda_create(const vitna_llama_t* m, const float* cos_tab, const float* sin_tab, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    if (!vitna_cuda_probe(err, err_len)) return NULL;
    if (c->head_dim > THREADS || c->head_dim % 2 != 0) {
        fail(err, err_len, "head_dim %zu is not supported on the GPU: it must be even and at most %d", c->head_dim, THREADS);
        return NULL;
    }
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
    total += 2 * padded(c->hidden * sizeof(float)) + 2 * padded(q_dim * sizeof(float)) +
             2 * padded(c->intermediate * sizeof(float)) + padded(c->vocab * sizeof(float)) +
             padded(c->n_heads * m->ctx * sizeof(float)) + padded(sizeof(step_t));

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
        (dm).dtype = (hm).dtype;                                                                \
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
        g->xn = (float*)carve(&cv, c->hidden * sizeof(float));
        g->q = (float*)carve(&cv, q_dim * sizeof(float));
        g->att = (float*)carve(&cv, q_dim * sizeof(float));
        g->gate = (float*)carve(&cv, c->intermediate * sizeof(float));
        g->up = (float*)carve(&cv, c->intermediate * sizeof(float));
        g->logits = (float*)carve(&cv, c->vocab * sizeof(float));
        g->scores = (float*)carve(&cv, c->n_heads * m->ctx * sizeof(float));
        g->step = (step_t*)carve(&cv, sizeof(step_t));
        /* A position is always written before it is read; zeros make a mistake there repeatable. */
        e = cudaMemset(g->k_cache, 0, 2 * padded(cache_floats * sizeof(float)));
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
