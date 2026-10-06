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
 * they are read, as ops.c does: F32, BF16 or F16 sixteen bytes at a time, and
 * ggml's block formats (quant.h) a block's scales and eight weights at a
 * time, to the bits quant.c gives. Each kernel is instantiated for the dtype
 * of each matrix it reads, so a quantized file that keeps some matrices at
 * more bits than others (a Q4_K_M file's Q6_K value and down projections)
 * runs as stored. Widths must be multiples of eight; a model that is not so,
 * or that pairs dtypes no kernel is instantiated for, is refused on load,
 * with the reason (check_shapes). The
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
 *
 * Several steps at once (vitna_cuda_rows) are the step's kernels for up to
 * MULTI_MAX rows, each a token at a position of a sequence: each weight is
 * read once for all of them, and each row's arithmetic is its step's, in its
 * step's order, so every logit is the one those steps would give, bit for
 * bit. It is how drafted tokens are checked together, and how requests to a
 * server run together, without changing a reply (--speculate, serve
 * --parallel). The key-value cache holds several sequences, each ctx
 * positions, one per request a server runs at once.
 *
 * A mixture of experts (OLMoE) runs a token a layer at a time instead, with
 * the host between the layers' halves, because which experts a layer needs
 * is known only once its router has run. Everything but the experts is
 * uploaded once. A layer's attention ends in the router, whose logits over
 * every expert are copied back; model.c routes from them, as it does on the
 * CPU, and names the experts and their weights; the experts' gate and up
 * projections run in one kernel, and their down projections, each scaled by
 * its weight and added up in the order given, in a second, which adds the
 * sum to the residual. The query and key projections pass through QK-norm
 * before their rotary embedding, in a kernel of its own.
 *
 * The experts live in memory, and the device keeps a cache of them: slots
 * of one expert each, its gate, up and down matrices, given up least used
 * first, least recently used between equals, as the expert stream's are
 * (expert_stream.h). An expert the cache lacks is copied into it on a stream
 * of its own, and the layer's kernels wait for that copy alone. The copies
 * come from the expert stream's slots when the experts are read from the
 * drive (--expert-cache), or else from the mapped checkpoint. Either is
 * registered with the device, page-locked, so a copy runs at the bus's full
 * speed; a mapped file is registered read-only, and the pages a copy needs
 * are read in as it runs. Where registering fails, copies go from pageable
 * memory, slower. With the router's logits, the next layer's router scores
 * the same residual, as the CPU path's lookahead does, and the half of the
 * experts it names that it ranks highest start being copied behind the
 * layer's own: a guess, which costs a copy when it is wrong and decides
 * nothing (vitna_cuda_moe_experts says why half). An expert's arithmetic does not
 * depend on where it came from or which slot holds it, so neither the size
 * of the cache nor where the copies come from changes any logit.
 *
 * Several tokens of a mixture run together as rows (vitna_cuda_moe_rows_*):
 * a prompt's, up to MOE_ROWS at a time, and drafted tokens and requests at
 * once. Every row goes through each layer's attention and router with the
 * multi-row kernels, the host routes every row, and each expert wanted runs
 * once for all the rows routed to it, copied to the device once a layer for
 * them rather than once a token. Each row's arithmetic is its step's, in its
 * step's order, the experts' outputs added in the order a step adds them, so
 * rows give the values steps give, bit for bit, however they are grouped.
 */

#define NOMINMAX
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "expert_stream.h"
#include "model_cuda.h"
#include "pool.h"
#include "warp.h"

#include <chrono>

/* Threads per block for every kernel but set_row_kernel: eight warps. */
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
 * asked for, come LOGIT_ROWS rows at a time. */
#define PREFILL_MAX 2048
#define PREFILL_MIN 256
#define PREFILL_BYTES ((size_t)64 << 20)
#define LOGIT_ROWS 32

/* The fewest tokens that run together; fewer are faster a step at a time. */
#define PROMPT_MIN 2

/* The most rows a pass of vitna_cuda_rows takes: several steps at once, each
 * row computed exactly as its step would be. A power of two, at
 * most 32: its kernels leave token t's sums in the MULTI_LANES lanes from
 * t * MULTI_LANES (warp_sum_multi). */
#define MULTI_MAX 8
#define MULTI_LANES (32 / MULTI_MAX)

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

/* A mixture of experts: the most experts a token may go through on the GPU,
 * the copies whose source the expert stream holds for them until they
 * finish, and the device memory left free when the expert cache takes what
 * the device has (for the desktop, and whatever else shares the GPU). */
#define EXPERTS_MAX 16
#define PENDING_MAX 64
#define EXPERT_MARGIN ((size_t)512 << 20)

/* --cpu-experts (moe_experts_split): the layers with misses after which a
 * side the split has left out is given one expert, to measure it again. A
 * probe that loses costs a layer one expert's difference, about a
 * millisecond on OLMoE; 64 layers are four of its tokens. */
#define SPLIT_PROBE 64

/* The most rows a mixture of experts runs a layer at a time: a prompt goes
 * in chunks of this many. A chunk's experts are copied to the device once
 * a layer, so a larger chunk copies less a token, and costs about 70 KB a
 * row of the device's memory. */
#define MOE_ROWS 1024

typedef struct {
    const void* w;
    int rows, cols;
    vitna_dtype_t dtype;  /* as stored, which the kernels that read it are instantiated for */
} dmat_t;

typedef struct {
    dmat_t q, k, v, o, gate, up, down;  /* gate, up and down for a dense model */
    const float* attn_norm;
    const float* mlp_norm;
    const float* q_norm;  /* QK-norm, or NULL */
    const float* k_norm;
    dmat_t router;        /* a mixture of experts */
    vitna_dtype_t e_gu, e_down; /* its experts' gate and up, and their down: one dtype each for the whole layer */
} dlayer_t;

/* The experts a kernel runs, and their weights, passed by value: their
 * gate, up and down matrices on the device, in the order their outputs are
 * added. */
typedef struct {
    const void* gate[EXPERTS_MAX];
    const void* up[EXPERTS_MAX];
    const void* down[EXPERTS_MAX];
    float weight[EXPERTS_MAX];
    int k;
} experts_t;

/* A slot of the device's expert cache. */
typedef struct {
    int64_t place;        /* the expert it holds, layer * n_experts + expert; -1 when empty */
    uint64_t last_use;
    bool held;            /* wanted by the layer being run, or just copied for its guess: not to be given up now */
    bool filling;         /* a copy into it was queued and is not known to have finished */
    bool guessed;         /* copied for a guess, and not wanted since */
    cudaEvent_t filled;   /* recorded after the latest copy into it */
} eslot_t;

/* A copy whose source the expert stream holds until it finishes. */
typedef struct {
    cudaEvent_t done;
    uint32_t place;
} pending_t;

typedef struct {
    unsigned char* mem;   /* n_slots slots of slot_bytes */
    size_t slot_bytes;    /* the largest of any layer's three parts, each padded */
    /* [n_layers][3]: the bytes of an expert's gate, up and down in each layer,
     * and where each starts in a slot, one after another, each at a multiple
     * of 256. A quantized file may keep one layer's down projections at more
     * bits than another's, so a layer's experts may be larger than another's. */
    size_t* part_bytes;
    size_t* part_off;
    int n_layers;
    int n_slots;
    eslot_t* slots;
    size_t n_places;
    int32_t* where;       /* [n_places]: the slot holding each, or -1 */
    uint32_t* uses;       /* [n_places]: times each was wanted, halved every 65,536 */
    uint64_t clock, acquired;
    cudaStream_t copy;    /* the copies run here, beside the kernels */

    /* Where the experts are in memory: the expert stream's slots, or the
     * mapped checkpoint, whose parts are in parts, [n_places][3]. */
    vitna_expert_stream_t* stream;
    const void** parts;
    pending_t pending[PENDING_MAX]; /* a ring, finishing in order, as the copy stream runs them */
    int* held_list;       /* [n_slots]: the slots a pass of rows holds */
    int p_head, p_len;
    int p_max;            /* the most the stream may hold for copies, leaving room for a layer's reads */
    void** registered;    /* host memory registered with the device, to unregister */
    int n_registered;
    bool pinned;          /* whether the copies come from page-locked memory */

    uint64_t hits, in_flight, misses, prefetched, prefetch_used;
    uint64_t copies, bytes_copied;
    /* Gate A8's split (moe_experts_split): the experts run on the CPU and
     * the time it spent on them, and what a byte of expert has cost the CPU
     * and a copy, in ms, averaged as measured, 0 until first measured. */
    uint64_t on_cpu;
    double cpu_ms;
    double cpu_ms_b, copy_ms_b;
    int cpu_idle, copy_idle;  /* layers with misses since the CPU, and a copy, last took one */
    cudaEvent_t copy_t0, copy_t1; /* timing a layer's copies for the device */
    size_t copy_timed;        /* the bytes between them, until read; 0 for none */
} expert_cache_t;

/* A token the graphs run, its position, and the sequence whose cache it
 * reads and writes: vitna_cuda_row_t, which the host fills. A step's is the
 * first; a pass over several reads one a token. */
typedef vitna_cuda_row_t row_t;

struct vitna_cuda_model {
    int device;
    char name[320];
    int max_blocks;         /* a grid's cap: eight blocks per multiprocessor */

    int n_layers, hidden, intermediate, n_heads, n_kv_heads, head_dim, vocab, ctx;
    float eps, scale;

    void* arena;            /* one allocation, carved into everything below */
    dlayer_t* layers;       /* host array of device pointers */
    dmat_t embed, lm_head;
    const float* final_norm;
    const float* cos_t;     /* [ctx][head_dim / 2] */
    const float* sin_t;
    int seqs;               /* sequences the cache holds */
    float* k_cache;         /* [n_layers][seqs][ctx][n_kv_heads * head_dim] */
    float* v_cache;
    float *x, *q, *att, *act, *logits;
    float *part_m, *part_l, *part_o; /* attention's slices: [n_heads][splits], and [.][.][head_dim] */
    unsigned int* done;              /* per key-value head, the slices finished in this layer */
    row_t* rows;            /* [MULTI_MAX]: what the graphs run (row_t) */

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
    cudaGraphExec_t multi[MULTI_MAX + 1]; /* vitna_cuda_rows for that many rows, captured on first use */
    int multi_max;          /* the most tokens a pass of it takes for this model, 0 for none */
    int capture_T;          /* the tokens enqueue_multi is capturing */
    float* host_multi;      /* pinned, MULTI_MAX x vocab, for its logits */
    float* host_logits;     /* pinned, for the copy back */

    /* A mixture of experts (moe): the layers' gate, up and down go unused,
     * and the experts are in the cache below. */
    bool moe;
    bool qk_norm;
    bool qk_head_norm;      /* QK-norm a head at a time (Qwen3), else across the heads (OLMoE) */
    int n_experts, n_used;
    float* k_raw;           /* QK-norm: a token's keys before it, n_kv_heads * head_dim */
    float* router_out;      /* [2][n_experts]: a layer's router logits, then the next layer's guess */
    float* host_router;     /* pinned, the same */
    float* moe_act;         /* [n_used][intermediate]: silu(gate) * up, expert by expert */
    expert_cache_t ec;

    /* Gate A8, vitna_cuda_cpu_experts: a step runs the experts the device
     * lacks on the CPU, in the GPU's arithmetic (warp.h), while the device
     * runs those it holds. */
    vitna_pool_t* cpu;      /* NULL: the experts it lacks are copied in and run here */
    float* moe_xs;          /* [hidden]: the layer's normalized residual, which the CPU's experts take */
    float* host_xs;         /* pinned, the same */
    float* moe_y;           /* [n_used][hidden]: the experts' outputs, the device's then the CPU's */
    float* host_y;          /* pinned, [n_used][hidden]: the CPU's */
    float* cpu_act;         /* [n_used][2][intermediate], the CPU's experts' activations */
    float* cpu_scratch;     /* vitna_warp_scratch_floats for each of the pool's threads */

    /* A mixture's rows (vitna_cuda_moe_rows_*): up to MOE_ROWS rows of a
     * pass, a layer at a time. r_x, r_xn, r_xn2, r_xg and r_proj hold
     * [MOE_ROWS][hidden], r_q and r_att [MOE_ROWS][n_heads * head_dim],
     * r_kraw [MOE_ROWS][n_kv_heads * head_dim], r_act [MOE_ROWS][intermediate],
     * r_router [2][MOE_ROWS][n_experts], r_logits [LOGIT_ROWS][vocab]; r_idx
     * and r_w each expert's rows and their weights, expert after expert. */
    int rows_g;             /* rows in a group that is normalized into shared memory */
    int r_n;                /* rows in the pass under way */
    row_t* r_rows;
    float *r_x, *r_xn, *r_xn2, *r_xg, *r_proj, *r_q, *r_att, *r_kraw, *r_act, *r_router, *r_logits;
    int32_t* r_idx;
    float* r_w;
    int32_t* h_idx;         /* pinned, as r_idx and r_w */
    float* h_w;
    float* h_rrouter;       /* pinned, as r_router */
    float* h_rlogits;       /* pinned, as r_logits */
    int32_t *e_count, *e_at, *e_fill; /* host [n_experts]: each expert's rows this layer, where its list starts, and a cursor */
    int32_t *e_list, *e_slot;           /* host [n_experts]: the experts wanted, in order, and the slot holding each */
    unsigned char* e_mark;              /* host [n_experts]: experts already in the guess */
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

/* --- Weights ---
 *
 * A matrix is rows of weights in one dtype: F32, BF16 or F16, a weight a
 * value, or one of ggml's block formats (quant.h), a row being whole blocks
 * of a scale or two and small integers. Every read below widens to float32
 * exactly what quant.c widens on the CPU, in the same arithmetic: each
 * product rounded on its own (__fmul_rn), never fused with the subtraction
 * after it (__fsub_rn), which nvcc would otherwise do. So a block widens to
 * the same bits on either device, and to gguf-py's. */

/* Bytes in a row of cols weights. A block format's cols is whole blocks. */
template <vitna_dtype_t DT>
__host__ __device__ __forceinline__ size_t row_bytes(int cols) {
    switch (DT) {
        case VITNA_DTYPE_F32: return (size_t)cols * 4;
        case VITNA_DTYPE_Q8_0: return (size_t)(cols / 32) * 34;
        case VITNA_DTYPE_Q4_K: return (size_t)(cols / 256) * 144;
        case VITNA_DTYPE_Q6_K: return (size_t)(cols / 256) * 210;
        default: return (size_t)cols * 2;
    }
}

/* The address of row r of a matrix cols wide. */
template <vitna_dtype_t DT>
__device__ __forceinline__ const void* row_at(const void* w, int r, int cols) {
    return static_cast<const char*>(w) + (size_t)r * row_bytes<DT>(cols);
}

__device__ __forceinline__ float half_at(const unsigned char* p) {
    return half_to_float(*reinterpret_cast<const unsigned short*>(p));
}

/* Q4_K's scale and min for group g of a block, from its 12 packed bytes s,
 * as quant.c's scale_min_k4. */
__device__ __forceinline__ void q4k_scale_min(const unsigned char* s, int g, float* scale, float* min) {
    int sc, mn;
    if (g < 4) {
        sc = s[g] & 63;
        mn = s[g + 4] & 63;
    } else {
        sc = (s[g + 4] & 0x0F) | ((s[g - 4] >> 6) << 4);
        mn = (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
    }
    *scale = (float)sc;
    *min = (float)mn;
}

/* Element i of a row, widened to float32 exactly. */
template <vitna_dtype_t DT>
__device__ __forceinline__ float widen_at(const void* row, int i) {
    const unsigned char* p = static_cast<const unsigned char*>(row);
    if (DT == VITNA_DTYPE_F32) return static_cast<const float*>(row)[i];
    if (DT == VITNA_DTYPE_BF16) return bf16_low(static_cast<const unsigned short*>(row)[i]);
    if (DT == VITNA_DTYPE_F16) return half_to_float(static_cast<const unsigned short*>(row)[i]);
    if (DT == VITNA_DTYPE_Q8_0) {
        const unsigned char* b = p + (i >> 5) * 34;
        return __fmul_rn((float)(signed char)b[2 + (i & 31)], half_at(b));
    }
    if (DT == VITNA_DTYPE_Q4_K) {
        const unsigned char* b = p + (i >> 8) * 144;
        const int j = i & 255, g = j >> 5, l = j & 31;
        float sc, mn;
        q4k_scale_min(b + 4, g, &sc, &mn);
        const float d1 = __fmul_rn(half_at(b), sc), m1 = __fmul_rn(half_at(b + 2), mn);
        const int q = (b[16 + 32 * (g >> 1) + l] >> ((g & 1) * 4)) & 0x0F;
        return __fsub_rn(__fmul_rn(d1, (float)q), m1);
    }
    /* Q6_K */
    const unsigned char* b = p + (i >> 8) * 210;
    const int j = i & 255, h = j >> 7, r = (j >> 5) & 3, l = j & 31;
    const int low = (b[64 * h + l + 32 * (r & 1)] >> (4 * (r >> 1))) & 0x0F;
    const int high = (b[128 + 32 * h + l] >> (2 * r)) & 0x03;
    const float ds = __fmul_rn(half_at(b + 208), (float)(signed char)b[192 + j / 16]);
    return __fmul_rn(ds, (float)((low | (high << 4)) - 32));
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

/* A block format's chunk lies inside one block: Q8_0's block holds four
 * chunks, Q4_K's and Q6_K's thirty-two. Q8_0's 34-byte blocks and Q6_K's
 * 210-byte ones put a row's blocks two bytes apart at worst, so those read
 * 16 bits at a time; Q4_K's 144-byte blocks keep 16-byte alignment, so its
 * eight 4-bit pairs come in one 8-byte read. */
template <>
__device__ __forceinline__ void load8<VITNA_DTYPE_Q8_0>(const void* row, int c, float w[8]) {
    const unsigned char* b = static_cast<const unsigned char*>(row) + (c >> 2) * 34;
    const float d = half_at(b);
    const unsigned short* q = reinterpret_cast<const unsigned short*>(b + 2 + (c & 3) * 8);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned short two = q[i];
        w[2 * i] = __fmul_rn((float)(signed char)(two & 0xff), d);
        w[2 * i + 1] = __fmul_rn((float)(signed char)(two >> 8), d);
    }
}

template <>
__device__ __forceinline__ void load8<VITNA_DTYPE_Q4_K>(const void* row, int c, float w[8]) {
    const unsigned char* b = static_cast<const unsigned char*>(row) + (c >> 5) * 144;
    const int e0 = (c & 31) * 8, g = e0 >> 5, l0 = e0 & 31;
    float sc, mn;
    q4k_scale_min(b + 4, g, &sc, &mn);
    const float d1 = __fmul_rn(half_at(b), sc), m1 = __fmul_rn(half_at(b + 2), mn);
    const uint2 qq = *reinterpret_cast<const uint2*>(b + 16 + 32 * (g >> 1) + l0);
    const int shift = (g & 1) * 4;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int word = i < 4 ? qq.x : qq.y;
        const int q = (word >> (8 * (i & 3) + shift)) & 0x0F;
        w[i] = __fsub_rn(__fmul_rn(d1, (float)q), m1);
    }
}

template <>
__device__ __forceinline__ void load8<VITNA_DTYPE_Q6_K>(const void* row, int c, float w[8]) {
    const unsigned char* b = static_cast<const unsigned char*>(row) + (c >> 5) * 210;
    const int e0 = (c & 31) * 8, h = e0 >> 7, r = (e0 >> 5) & 3, l0 = e0 & 31;
    const unsigned short* lo = reinterpret_cast<const unsigned short*>(b + 64 * h + l0 + 32 * (r & 1));
    const unsigned short* hi = reinterpret_cast<const unsigned short*>(b + 128 + 32 * h + l0);
    const float ds = __fmul_rn(half_at(b + 208), (float)(signed char)b[192 + 8 * h + 2 * r + (l0 >> 4)]);
    const int lshift = 4 * (r >> 1), hshift = 2 * r;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned short l2 = lo[i], h2 = hi[i];
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int low = ((l2 >> (8 * k)) >> lshift) & 0x0F;
            const int high = ((h2 >> (8 * k)) >> hshift) & 0x03;
            w[2 * i + k] = __fmul_rn(ds, (float)((low | (high << 4)) - 32));
        }
    }
}

/* e^x as exact.c's vitna_exp_exact computes it, operation for operation, so
 * the experts' activation is the same bits here as on the CPU (gate A8).
 * CUDA's expf finishes in ex2.approx, the hardware's approximation, which
 * the CPU cannot reproduce. */
__device__ __forceinline__ float vitna_exp_dev(float x) {
    if (x != x) return x;
    if (x > 88.72283935546875f) return __uint_as_float(0x7f800000u);
    if (x < -87.3365478515625f) return 0.0f;
    const float t = __fmul_rn(x, 1.44269502162933349609375f);
    const float n = rintf(t);
    float r = __fmaf_rn(n, -0.693359375f, x);
    r = __fmaf_rn(n, 2.12194440e-4f, r);
    const float z = __fmul_rn(r, r);
    float p = 1.9875691500e-4f;
    p = __fmaf_rn(p, r, 1.3981999507e-3f);
    p = __fmaf_rn(p, r, 8.3334519073e-3f);
    p = __fmaf_rn(p, r, 4.1665795894e-2f);
    p = __fmaf_rn(p, r, 1.6666665459e-1f);
    p = __fmaf_rn(p, r, 5.0000001201e-1f);
    p = __fmaf_rn(p, z, r);
    p = __fadd_rn(p, 1.0f);
    const int ni = (int)n;
    const int h = ni / 2;
    const float a = __fmul_rn(p, __uint_as_float((unsigned int)(h + 127) << 23));
    return __fmul_rn(a, __uint_as_float((unsigned int)(ni - h + 127) << 23));
}

/* An expert's activation, silu(g) * u, as exact.c's vitna_silu_mul_exact. */
__device__ __forceinline__ float expert_act(float g, float u) {
    const float e = vitna_exp_dev(-g);
    const float d = __fadd_rn(1.0f, e);
    const float s = __fdiv_rn(g, d);
    return __fmul_rn(s, u);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Where element i of a row of n (a multiple of 8) goes in a split row: the
 * first four of each chunk of eight in the first half, the other four in the
 * second. A pass over several tokens keeps the rows it multiplies by weights
 * this way, so a warp reading chunks lane, lane + 32, ... reads 512
 * contiguous bytes at a time (x_chunk). */
__device__ __forceinline__ int split_at(int i, int n) {
    return ((i & 4) ? n / 2 : 0) + ((i >> 3) << 2) + (i & 3);
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
    const void* row = row_at<DT>(w, r, cols);
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

/* The most elements of a vector norm_to_shared keeps in each thread's registers. */
#define NORM_REGS 8

/* xs = x / sqrt(mean(x^2) + eps) * w, as kernels.c's vitna_rmsnorm, written
 * to shared memory by the whole block. Every block computes the same values.
 * A vector of up to NORM_REGS * THREADS elements is read once, x and w
 * together, and kept in registers, so the block makes one trip to memory
 * where reading w only after the sum of squares made two, one after the
 * other; wider vectors take the loops at the end. The arithmetic is the same
 * either way, in the same order. */
__device__ void norm_to_shared(const float* __restrict__ x, const float* __restrict__ w, float* xs, int n, float eps, float* red) {
    if (n <= NORM_REGS * THREADS) {
        float xv[NORM_REGS], wv[NORM_REGS];
#pragma unroll
        for (int j = 0; j < NORM_REGS; j++) {
            const int i = threadIdx.x + j * THREADS;
            if (i < n) {
                xv[j] = x[i];
                wv[j] = w[i];
            }
        }
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < NORM_REGS; j++) {
            if (threadIdx.x + j * THREADS < n) ss = fmaf(xv[j], xv[j], ss);
        }
        ss = block_sum(ss, red);
        const float inv = 1.0f / sqrtf(ss / (float)n + eps);
#pragma unroll
        for (int j = 0; j < NORM_REGS; j++) {
            const int i = threadIdx.x + j * THREADS;
            if (i < n) xs[i] = xv[j] * inv * wv[j];
        }
        __syncthreads();
        return;
    }
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += THREADS) ss = fmaf(x[i], x[i], ss);
    ss = block_sum(ss, red);
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += THREADS) xs[i] = x[i] * inv * w[i];
    __syncthreads();
}

/* The one kernel a step launches outside the graphs: it says which token to
 * run, where, and in which sequence. Its arguments are copied at launch, so
 * the host can move on. */
__global__ void set_row_kernel(row_t* rows, int token, int pos, int seq) {
    rows[0].token = token;
    rows[0].pos = pos;
    rows[0].seq = seq;
}

template <vitna_dtype_t DT>
__global__ void embed_kernel(const void* __restrict__ table, const row_t* __restrict__ rows, int hidden, float* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[i] = widen_at<DT>(row_at<DT>(table, rows[0].token, hidden), i);
}

/* RMSNorm, then the query, key and value projections of one layer. The
 * half-split rotary embedding of ops.c's vitna_rope_half pairs element j of a
 * head with element j + head_dim / 2, so a warp computes those two rows
 * together and writes them rotated. Keys and values go straight to the
 * position's slot in the cache; value rows are taken two at a time, unrotated.
 * With k_raw, for QK-norm, which comes before the rotary embedding, the
 * queries and keys are written as projected, the keys to k_raw, and
 * qk_norm_rope_kernel finishes them. The query and key projections share
 * the dtype DT; the value projection's is DTV, which a quantized file may
 * keep at more bits than the others (Q6_K beside Q4_K). */
template <vitna_dtype_t DT, vitna_dtype_t DTV>
__global__ void attn_in_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                               const void* __restrict__ wq, const void* __restrict__ wk, const void* __restrict__ wv,
                               float* __restrict__ q, float* __restrict__ kc, float* __restrict__ vc,
                               const float* __restrict__ cos_all, const float* __restrict__ sin_all,
                               const row_t* __restrict__ rows, size_t seq_stride, int hidden, int n_heads, int n_kv_heads, int head_dim,
                               float* __restrict__ k_raw) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    norm_to_shared(x, norm_w, xs, hidden, eps, red);

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = head_dim / 2;
    const int kv_dim = n_kv_heads * head_dim;
    const size_t pos = (size_t)rows[0].pos;
    const float* cos_t = cos_all + pos * half;
    const float* sin_t = sin_all + pos * half;
    float* k = kc + (size_t)rows[0].seq * seq_stride + pos * kv_dim;
    float* v = vc + (size_t)rows[0].seq * seq_stride + pos * kv_dim;
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
            if (lane == 0 && k_raw) {
                float* out = is_q ? q : k_raw;
                out[ra] = a;
                out[rb] = b;
            } else if (lane == 0) {
                float* out = is_q ? q : k;
                out[ra] = a * cos_t[j] - b * sin_t[j];
                out[rb] = b * cos_t[j] + a * sin_t[j];
            }
        } else {
            const int ra = 2 * (u - units_qk);
            const float a = row_dot<DTV>(wv, ra, hidden, xs, lane);
            const float b = row_dot<DTV>(wv, ra + 1, hidden, xs, lane);
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
 * (head_dim + per) floats.
 *
 * The grid is (tokens, key-value heads, ATTENTION_SPLITS): the split varies
 * slowest, so every block with a slice goes out before any without. Four
 * blocks fit a multiprocessor where, at the 128 registers the kernel takes
 * unbounded, two did; a pass over eight tokens at position 160 has 144
 * blocks with slices, and a step past about 1,000 positions 96, which then
 * ran in two rounds on the RTX 3070's 46 multiprocessors. With split_out,
 * each token's output is written as a split row, as a pass's output
 * projection reads it. */
__global__ void __launch_bounds__(THREADS, 4)
    attention_kernel(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc, float* __restrict__ att,
                     float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_o, unsigned int* __restrict__ done,
                     const row_t* __restrict__ rows, size_t seq_stride, int head_dim, int kv_dim, int group, float scale, int split_out) {
    extern __shared__ float4 shared4[];
    __shared__ int last;
    const int split = blockIdx.z, kvh = blockIdx.y;
    /* Row blockIdx.x of a pass over several (vitna_cuda_rows) has a query, an
     * output, slices and a count of its own, and runs exactly as a step at
     * its position in its sequence would; a step is row 0 of a grid one wide. */
    const int tz = blockIdx.x;
    const size_t n_heads = (size_t)gridDim.y * group;
    q += (size_t)tz * n_heads * head_dim;
    att += (size_t)tz * n_heads * head_dim;
    part_m += (size_t)tz * n_heads * ATTENTION_SPLITS;
    part_l += (size_t)tz * n_heads * ATTENTION_SPLITS;
    part_o += (size_t)tz * n_heads * ATTENTION_SPLITS * head_dim;
    done += (size_t)tz * gridDim.y;
    const int n = rows[tz].pos + 1;
    const size_t seq_at = (size_t)rows[tz].seq * seq_stride;
    /* As many slices as n needs at ATTENTION_MIN_SLICE positions each, up
     * to the grid's depth; the blocks past them have nothing to do. */
    const int splits = min((int)gridDim.z, (n + ATTENTION_MIN_SLICE - 1) / ATTENTION_MIN_SLICE);
    if (split >= splits) return;
    const int per = (n + splits - 1) / splits;
    const int t0 = split * per;
    const int len = max(0, min(n, t0 + per) - t0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int h0 = kvh * group; /* the first query head that reads this key-value head */
    const int gd = group * head_dim;
    const float* kh = kc + seq_at + (size_t)kvh * head_dim;
    const float* vh = vc + seq_at + (size_t)kvh * head_dim;
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
        const size_t at = h * head_dim + i;
        att[split_out ? (size_t)split_at((int)at, (int)(n_heads * head_dim)) : at] = o / sum;
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

/* --- A mixture of experts (vitna_cuda_moe_route and vitna_cuda_moe_experts) --- */

/* QK-norm, then the rotary embedding: OLMoE normalizes the query projection
 * with an RMSNorm over all of its values at once, across the heads, and the
 * key projection the same way, as model.c does, then rotates each head as
 * attn_in_kernel would have. Block 0 takes the queries, in place; block 1
 * the keys, from k_raw into the position's slot in the cache. blockIdx.y is
 * the row, of q and k_raw's rows and of rows: 0 for a step. */
__global__ void qk_norm_rope_kernel(float* __restrict__ q, const float* __restrict__ k_raw, float* __restrict__ kc,
                                    const float* __restrict__ q_norm, const float* __restrict__ k_norm, float eps,
                                    const float* __restrict__ cos_all, const float* __restrict__ sin_all, const row_t* __restrict__ rows,
                                    size_t seq_stride, int n_heads, int n_kv_heads, int head_dim) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* vs = reinterpret_cast<float*>(shared4);
    const int t = blockIdx.y;
    q += (size_t)t * n_heads * head_dim;
    k_raw += (size_t)t * n_kv_heads * head_dim;
    const bool keys = blockIdx.x == 1;
    const int n = (keys ? n_kv_heads : n_heads) * head_dim;
    norm_to_shared(keys ? k_raw : q, keys ? k_norm : q_norm, vs, n, eps, red);
    const int half = head_dim / 2;
    const size_t pos = (size_t)rows[t].pos;
    const float* cos_t = cos_all + pos * half;
    const float* sin_t = sin_all + pos * half;
    float* out = keys ? kc + (size_t)rows[t].seq * seq_stride + pos * n : q;
    for (int u = threadIdx.x; u < n / 2; u += THREADS) {
        const int j = u % half;
        const int ra = (u / half) * head_dim + j, rb = ra + half;
        const float a = vs[ra], b = vs[rb];
        out[ra] = a * cos_t[j] - b * sin_t[j];
        out[rb] = b * cos_t[j] + a * sin_t[j];
    }
}

/* The same for Qwen3, whose QK-norm normalizes each head alone, every head
 * of the queries with the same head_dim weights and every head of the keys
 * with theirs, as Qwen3MoeAttention normalizes q and k viewed as
 * [heads][head_dim], as model.c does: a block a head. Block b < n_heads
 * takes query head b, in place; the rest key head b - n_heads, from k_raw
 * into the position's slot in the cache. Each head is rotated as
 * qk_norm_rope_kernel rotates it. blockIdx.y is the row. */
__global__ void qk_head_norm_rope_kernel(float* __restrict__ q, const float* __restrict__ k_raw, float* __restrict__ kc,
                                         const float* __restrict__ q_norm, const float* __restrict__ k_norm, float eps,
                                         const float* __restrict__ cos_all, const float* __restrict__ sin_all, const row_t* __restrict__ rows,
                                         size_t seq_stride, int n_heads, int n_kv_heads, int head_dim) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* vs = reinterpret_cast<float*>(shared4);
    const int t = blockIdx.y;
    const bool keys = (int)blockIdx.x >= n_heads;
    const int h = keys ? (int)blockIdx.x - n_heads : (int)blockIdx.x;
    const float* src = keys ? k_raw + ((size_t)t * n_kv_heads + h) * head_dim : q + ((size_t)t * n_heads + h) * head_dim;
    norm_to_shared(src, keys ? k_norm : q_norm, vs, head_dim, eps, red);
    const int half = head_dim / 2;
    const size_t pos = (size_t)rows[t].pos;
    const float* cos_t = cos_all + pos * half;
    const float* sin_t = sin_all + pos * half;
    float* out = keys ? kc + (size_t)rows[t].seq * seq_stride + pos * ((size_t)n_kv_heads * head_dim) + (size_t)h * head_dim
                      : q + ((size_t)t * n_heads + h) * head_dim;
    for (int j = threadIdx.x; j < half; j += THREADS) {
        const float a = vs[j], b = vs[j + half];
        out[j] = a * cos_t[j] - b * sin_t[j];
        out[j + half] = b * cos_t[j] + a * sin_t[j];
    }
}

/* RMSNorm with the layer's MLP norm, then its router: a logit for every
 * expert, into out. The blocks with blockIdx.y 1 do the same with the next
 * layer's norm and router, on the same residual, into out + n_experts: the
 * guess of the CPU path's lookahead. */
template <vitna_dtype_t DT>
__global__ void router_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, const void* __restrict__ w,
                              const float* __restrict__ next_norm_w, const void* __restrict__ next_w, float eps, float* __restrict__ out,
                              int hidden, int n_experts) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    const bool next = blockIdx.y == 1;
    norm_to_shared(x, next ? next_norm_w : norm_w, xs, hidden, eps, red);
    const void* router = next ? next_w : w;
    float* o = out + (next ? n_experts : 0);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * WARPS + warp; r < n_experts; r += gridDim.x * WARPS) {
        const float s = row_dot<DT>(router, r, hidden, xs, lane);
        if (lane == 0) o[r] = s;
    }
}

/* RMSNorm with the layer's MLP norm, then the gate and up projections of
 * each expert, as mlp_in_kernel computes a dense layer's: unit u is row
 * u % intermediate of expert u / intermediate, and silu(gate) * up goes to
 * act[u]. */
template <vitna_dtype_t DT>
__global__ void experts_in_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps, experts_t e,
                                  float* __restrict__ act, int hidden, int intermediate) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    norm_to_shared(x, norm_w, xs, hidden, eps, red);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int units = e.k * intermediate;
    for (int u = blockIdx.x * WARPS + warp; u < units; u += gridDim.x * WARPS) {
        const int k = u / intermediate, r = u % intermediate;
        const float g = row_dot<DT>(e.gate[k], r, hidden, xs, lane);
        const float v = row_dot<DT>(e.up[k], r, hidden, xs, lane);
        if (lane == 0) act[u] = expert_act(g, v);
    }
}

/* The experts' down projections, added to the residual: a warp takes row r
 * of every expert, and its output there is each expert's, multiplied by the
 * expert's weight and rounded, added to the sum so far and rounded, from
 * zero, expert after expert in the order given; then the sum is added to
 * the residual. That is transformers' eager experts, which scale each
 * expert's output and index_add_ it in order of expert, then the decoder
 * layer's residual add; model.c's mixture_of_experts adds the same way. The
 * activations of every expert are read into shared memory first. */
template <vitna_dtype_t DT>
__global__ void experts_down_kernel(const float* __restrict__ act, experts_t e, float* __restrict__ x, int hidden, int intermediate) {
    extern __shared__ float4 shared4[];
    float* as = reinterpret_cast<float*>(shared4);
    const int n4 = e.k * intermediate / 4;
    for (int i = threadIdx.x; i < n4; i += THREADS) shared4[i] = reinterpret_cast<const float4*>(act)[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = blockIdx.x * WARPS + warp; r < hidden; r += gridDim.x * WARPS) {
        float sum = 0.0f;
        for (int k = 0; k < e.k; k++) {
            const float y = row_dot<DT>(e.down[k], r, intermediate, as + (size_t)k * intermediate, lane);
            sum = __fadd_rn(sum, __fmul_rn(y, e.weight[k]));
        }
        if (lane == 0) x[r] = x[r] + sum;
    }
}

/* The experts' down projections as experts_down_kernel computes them, each
 * expert's output kept apart rather than weighted and added: y holds k rows
 * of hidden, row j expert j's. The sum experts_down_kernel would form from
 * them, in the same order, is its own, bit for bit (gate A8, where some of
 * the outputs come from the CPU). */
template <vitna_dtype_t DT>
__global__ void experts_parts_kernel(const float* __restrict__ act, experts_t e, float* __restrict__ y, int hidden, int intermediate) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int u = blockIdx.x * WARPS + warp; u < e.k * hidden; u += gridDim.x * WARPS) {
        const int k = u / hidden, r = u % hidden;
        const float s = row_dot<DT>(e.down[k], r, intermediate, act + (size_t)k * intermediate, lane);
        if (lane == 0) y[u] = s;
    }
}

/* RMSNorm of x into xs in global memory, by one block, as norm_to_shared
 * computes it for every kernel that normalizes a token. */
__global__ void norm_out_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps, float* __restrict__ xs, int n) {
    extern __shared__ float4 shared4[];
    __shared__ float red[WARPS];
    float* s = reinterpret_cast<float*>(shared4);
    norm_to_shared(x, w, s, n, eps, red);
    for (int i = threadIdx.x; i < n; i += THREADS) xs[i] = s[i];
}

/* The experts a step ran, on the device and on the CPU, mixed: output i, in
 * the order the experts are added, is row at[i] of y, weighted by weight[i]. */
typedef struct {
    int at[EXPERTS_MAX];
    float weight[EXPERTS_MAX];
    int k;
} mix_t;

/* The outputs in y, each multiplied by its weight and rounded, added to the
 * sum so far and rounded, from zero, in the order given, and the sum added
 * to the residual: what experts_down_kernel does with the outputs it
 * computes itself, so the residual is the bits it would be (gate A8). */
__global__ void experts_mix_kernel(const float* __restrict__ y, mix_t m, float* __restrict__ x, int hidden) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < hidden; r += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < m.k; i++) sum = __fadd_rn(sum, __fmul_rn(y[(size_t)m.at[i] * hidden + r], m.weight[i]));
        x[r] = x[r] + sum;
    }
}

/* --- Several steps at once, exactly (vitna_cuda_rows) ---
 *
 * The decode kernels above for up to MULTI_MAX tokens, each at its position
 * in its sequence: a warp reads a row of weights once and applies it to
 * every token. Each token's arithmetic is a step's, in a step's order: the RMSNorm
 * sums its squares as norm_to_shared does and reduces them in block_sum's
 * tree, and each row's product with a token runs row_dot's chunks and lanes
 * in row_dot's order, then warp_sum's. So every value is the one that many
 * steps would compute, bit for bit, which is what lets several drafted
 * tokens be checked in one pass without changing any reply. */

static_assert(MULTI_MAX >= 1 && MULTI_MAX <= 32 && (MULTI_MAX & (MULTI_MAX - 1)) == 0, "MULTI_MAX is a power of two, at most 32");

/* warp_sum of N values a lane holds, N a power of two, all at once. At the
 * level with offset o a lane keeps half of its values and gives the other
 * half to lane ^ o, which keeps those: the lane with bit o clear keeps the
 * lower half. Each value's partial still adds lane ^ o's at each level, as
 * warp_sum adds it, so each sum is warp_sum's bit for bit, and N values cost
 * N - 1 shuffles down to one a lane, then one a level left, where warp_sum
 * would spend five each. Value i ends in the lanes l with l / (32 / N) == i. */
template <int N>
struct warp_halving {
    static __device__ __forceinline__ float sum(float* v, int lane, int o) {
        const bool upper = (lane & o) != 0;
#pragma unroll
        for (int i = 0; i < N / 2; i++) {
            const float keep = upper ? v[i + N / 2] : v[i];
            const float give = upper ? v[i] : v[i + N / 2];
            v[i] = keep + __shfl_xor_sync(0xffffffffu, give, o);
        }
        return warp_halving<N / 2>::sum(v, lane, o >> 1);
    }
};

template <>
struct warp_halving<1> {
    static __device__ __forceinline__ float sum(float* v, int, int o) {
        float s = v[0];
        for (; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        return s;
    }
};

/* The warp sums of a token's partials for each of MULTI_MAX tokens (9
 * shuffles for 8, where warp_sum would spend 40): returns token
 * lane / MULTI_LANES's, bit for bit warp_sum's. v is spent. */
__device__ __forceinline__ float warp_sum_multi(float v[MULTI_MAX], int lane) {
    return warp_halving<MULTI_MAX>::sum(v, lane, 16);
}

/* block_sum's total from its warps' sums r[0] to r[WARPS - 1], in one
 * thread: the tree of block_sum's second warp_sum. The lanes past WARPS hold
 * zeros there, and adding a zero leaves a sum of squares as it was, so the
 * tree is warp_sum's butterfly over r[0] to r[WARPS - 1] alone. */
static_assert(WARPS <= 32 && (WARPS & (WARPS - 1)) == 0, "WARPS is a power of two, at most 32");
__device__ __forceinline__ float sum_warps(const float* r) {
    float v[WARPS];
#pragma unroll
    for (int i = 0; i < WARPS; i++) v[i] = r[i];
#pragma unroll
    for (int o = WARPS / 2; o > 0; o >>= 1) {
#pragma unroll
        for (int i = 0; i < o; i++) v[i] = v[i] + v[i + o];
    }
    return v[0];
}

/* The most elements of a row norm_multi_to_shared keeps in registers. */
#define NORM_MULTI_REGS 4

/* x's T rows (stride n) through RMSNorm into xs (T split rows, stride n),
 * each as norm_to_shared computes its one. red holds MULTI_MAX * WARPS
 * floats. Rows of up to NORM_MULTI_REGS * THREADS elements are read once,
 * every row and the weights together, and kept in registers, one trip to
 * memory where the loops make two. A warp's sums of squares go through
 * warp_sum_multi, and lane t of each warp finishes token t's total and its
 * scale, which the warp then shares: the arithmetic is the same, in the same
 * order, for a fraction of the shuffles, and no thread works out all T scales. */
__device__ void norm_multi_to_shared(const float* __restrict__ x, const float* __restrict__ w, float* xs, int n, float eps, float* red,
                                     int T) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const bool in_regs = n <= NORM_MULTI_REGS * THREADS;
    float ss[MULTI_MAX];
    float xv[MULTI_MAX][NORM_MULTI_REGS], wv[NORM_MULTI_REGS];
    if (in_regs) {
#pragma unroll
        for (int j = 0; j < NORM_MULTI_REGS; j++) {
            const int i = threadIdx.x + j * THREADS;
            if (i < n) {
                wv[j] = w[i];
#pragma unroll
                for (int t = 0; t < MULTI_MAX; t++) {
                    if (t < T) xv[t][j] = x[(size_t)t * n + i];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MULTI_MAX; t++) {
            ss[t] = 0.0f;
#pragma unroll
            for (int j = 0; j < NORM_MULTI_REGS; j++) {
                if (t < T && threadIdx.x + j * THREADS < n) ss[t] = fmaf(xv[t][j], xv[t][j], ss[t]);
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < MULTI_MAX; t++) {
            ss[t] = 0.0f;
            if (t < T) {
                for (int i = threadIdx.x; i < n; i += THREADS) ss[t] = fmaf(x[(size_t)t * n + i], x[(size_t)t * n + i], ss[t]);
            }
        }
    }
    __syncthreads(); /* red may still be read from the reduction before */
    const float part = warp_sum_multi(ss, lane);
    if (lane % MULTI_LANES == 0 && lane / MULTI_LANES < T) red[(lane / MULTI_LANES) * WARPS + warp] = part;
    __syncthreads();
    float scale = 0.0f;
    if (lane < T) scale = 1.0f / sqrtf(sum_warps(red + lane * WARPS) / (float)n + eps);
    float inv[MULTI_MAX];
#pragma unroll
    for (int t = 0; t < MULTI_MAX; t++) {
        if (t < T) inv[t] = __shfl_sync(0xffffffffu, scale, t);
    }
    if (in_regs) {
#pragma unroll
        for (int t = 0; t < MULTI_MAX; t++) {
            if (t < T) {
#pragma unroll
                for (int j = 0; j < NORM_MULTI_REGS; j++) {
                    const int i = threadIdx.x + j * THREADS;
                    if (i < n) xs[(size_t)t * n + split_at(i, n)] = xv[t][j] * inv[t] * wv[j];
                }
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < MULTI_MAX; t++) {
            if (t < T) {
                for (int i = threadIdx.x; i < n; i += THREADS) xs[(size_t)t * n + split_at(i, n)] = x[(size_t)t * n + i] * inv[t] * w[i];
            }
        }
    }
    __syncthreads();
}

/* x's chunk c of row t (stride ld): its elements 8c to 8c + 7, as a and b.
 * A SPLIT row (split_at's layout, where ld is the row's length) keeps the
 * first four at 4c and the rest at ld / 2 + 4c: a warp reading chunks lane,
 * lane + 32, ... then reads 512 contiguous bytes at once, which shared
 * memory serves in four wavefronts, and L1 in four lines. In a plain row the
 * lanes' reads are 32 bytes apart and take eight. */
template <bool SPLIT>
__device__ __forceinline__ void x_chunk(const float* x, size_t ld, int t, int c, float4& a, float4& b) {
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)t * ld);
    if (SPLIT) {
        a = x4[c];
        b = x4[ld / 8 + c];
    } else {
        a = x4[2 * c];
        b = x4[2 * c + 1];
    }
}

/* R rows of weights, row[k] each, cols wide, times x's T rows (stride ld,
 * SPLIT as x_chunk reads them), each product summed across the warp as
 * row_dot sums it: out[k] is row k's for token lane / MULTI_LANES (zero past
 * T). A lane takes row_dot's chunks lane, lane + 32, ..., and each chunk's
 * eight fused multiply-adds in order, for every row and token. It reads each
 * chunk of x once for all R rows: the weights are read once in all and x
 * once a token, so x is where a pass over several tokens spends its reads,
 * and R rows at a time divide them by R. */
template <vitna_dtype_t DT, bool SPLIT, int R>
__device__ __forceinline__ void rows_dot_multi(const void* const (&row)[R], int cols, const float* x, size_t ld, int T, int lane,
                                               float (&out)[R]) {
    float s[R][MULTI_MAX];
#pragma unroll
    for (int k = 0; k < R; k++) {
#pragma unroll
        for (int t = 0; t < MULTI_MAX; t++) s[k][t] = 0.0f;
    }
    for (int c = lane; c < cols / 8; c += 32) {
        float v[R][8];
#pragma unroll
        for (int k = 0; k < R; k++) load8<DT>(row[k], c, v[k]);
#pragma unroll
        for (int t = 0; t < MULTI_MAX; t++) {
            if (t < T) {
                float4 lo, hi;
                x_chunk<SPLIT>(x, ld, t, c, lo, hi);
                const float xv[8] = { lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w };
#pragma unroll
                for (int e = 0; e < 8; e++) {
#pragma unroll
                    for (int k = 0; k < R; k++) s[k][t] = fmaf(v[k][e], xv[e], s[k][t]);
                }
            }
        }
    }
#pragma unroll
    for (int k = 0; k < R; k++) out[k] = warp_sum_multi(s[k], lane);
}

/* The embedding rows of a pass's T tokens: embed_rows_kernel, the tokens read from rows. */
template <vitna_dtype_t DT>
__global__ void embed_multi_kernel(const void* __restrict__ table, const row_t* __restrict__ rows, int hidden, float* __restrict__ x) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[(size_t)t * hidden + i] = widen_at<DT>(row_at<DT>(table, rows[t].token, hidden), i);
}

/* attn_in_kernel for T rows, each at its position in its sequence: x and q
 * hold T rows. Each token's two sums land in lanes of their own, and the first of
 * those lanes writes the token: T tokens' writes, and their reads of the
 * rotation, go out together. Of n_rows rows, blockIdx.y takes the group of
 * up to T from blockIdx.y * T: a pass of vitna_cuda_rows is one group. With
 * k_raw, for QK-norm, the queries and keys are written as projected, as
 * attn_in_kernel writes them then. DT and DTV as attn_in_kernel's. */
template <vitna_dtype_t DT, vitna_dtype_t DTV>
__global__ void attn_in_multi_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                                     const void* __restrict__ wq, const void* __restrict__ wk, const void* __restrict__ wv,
                                     float* __restrict__ q, float* __restrict__ kc, float* __restrict__ vc,
                                     const float* __restrict__ cos_all, const float* __restrict__ sin_all,
                                     const row_t* __restrict__ rows, size_t seq_stride, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                     int T, int n_rows, float* __restrict__ k_raw) {
    extern __shared__ float4 shared4[];
    __shared__ float red[MULTI_MAX * WARPS];
    float* xs = reinterpret_cast<float*>(shared4); /* [T][hidden] */
    const int r0 = blockIdx.y * T;
    T = min(T, n_rows - r0);
    x += (size_t)r0 * hidden;
    q += (size_t)r0 * n_heads * head_dim;
    if (k_raw) k_raw += (size_t)r0 * n_kv_heads * head_dim;
    rows += r0;
    norm_multi_to_shared(x, norm_w, xs, hidden, eps, red, T);

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    const int half = head_dim / 2;
    const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
    const row_t row = rows[t < T ? t : 0];
    const size_t pos = (size_t)row.pos;
    kc += (size_t)row.seq * seq_stride;
    vc += (size_t)row.seq * seq_stride;
    const int units_q = n_heads * half, units_qk = units_q + n_kv_heads * half;
    const int units = units_qk + kv_dim / 2;
    for (int u = blockIdx.x * WARPS + warp; u < units; u += gridDim.x * WARPS) {
        if (u < units_qk) {
            const bool is_q = u < units_q;
            const int uu = is_q ? u : u - units_q;
            const int j = uu % half;
            const int ra = (uu / half) * head_dim + j, rb = ra + half;
            const void* w = is_q ? wq : wk;
            const void* rows[2] = { row_at<DT>(w, ra, hidden), row_at<DT>(w, rb, hidden) };
            float ab[2];
            rows_dot_multi<DT, true, 2>(rows, hidden, xs, hidden, T, lane, ab);
            const float a = ab[0], b = ab[1];
            if (writes && k_raw) {
                float* out = is_q ? q + (size_t)t * q_dim : k_raw + (size_t)t * kv_dim;
                out[ra] = a;
                out[rb] = b;
            } else if (writes) {
                const float c = cos_all[pos * half + j], s = sin_all[pos * half + j];
                float* out = is_q ? q + (size_t)t * q_dim : kc + pos * kv_dim;
                out[ra] = a * c - b * s;
                out[rb] = b * c + a * s;
            }
        } else {
            const int ra = 2 * (u - units_qk);
            const void* rows[2] = { row_at<DTV>(wv, ra, hidden), row_at<DTV>(wv, ra + 1, hidden) };
            float ab[2];
            rows_dot_multi<DTV, true, 2>(rows, hidden, xs, hidden, T, lane, ab);
            if (writes) {
                float* v = vc + pos * kv_dim;
                v[ra] = ab[0];
                v[ra + 1] = ab[1];
            }
        }
    }
}

/* matvec_add_kernel for T tokens: y's T rows (stride rows) += W x's T split rows (stride cols).
 * Of n_rows rows, blockIdx.y takes the group of up to T from blockIdx.y * T. */
template <vitna_dtype_t DT>
__global__ void matvec_add_multi_kernel(const void* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int rows,
                                        int cols, int T, int n_rows) {
    const int r0 = blockIdx.y * T;
    T = min(T, n_rows - r0);
    x += (size_t)r0 * cols;
    y += (size_t)r0 * rows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    for (int r = blockIdx.x * WARPS + warp; r < rows; r += gridDim.x * WARPS) {
        const void* row[1] = { row_at<DT>(w, r, cols) };
        float s[1];
        rows_dot_multi<DT, true, 1>(row, cols, x, cols, T, lane, s);
        if (writes) y[(size_t)t * rows + r] = y[(size_t)t * rows + r] + s[0];
    }
}

/* mlp_in_kernel for T tokens: x holds T rows, act T split rows. A warp takes
 * rows r and r + 1 of both matrices, four rows to each read of x, and each
 * block's norm serves twice the rows. Held to three blocks a multiprocessor
 * (unbounded it takes 113 registers and two fit), SmolLM2-135M's 96 blocks
 * run in one wave on the RTX 3070's 46 multiprocessors. */
template <vitna_dtype_t DT>
__global__ void __launch_bounds__(THREADS, 3) mlp_in_multi_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                                                                  const void* __restrict__ wg, const void* __restrict__ wu,
                                                                  float* __restrict__ act, int hidden, int intermediate, int T) {
    extern __shared__ float4 shared4[];
    __shared__ float red[MULTI_MAX * WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    norm_multi_to_shared(x, norm_w, xs, hidden, eps, red, T);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    for (int r = 2 * (blockIdx.x * WARPS + warp); r < intermediate; r += 2 * gridDim.x * WARPS) {
        const void* rows[4] = { row_at<DT>(wg, r, hidden), row_at<DT>(wu, r, hidden), row_at<DT>(wg, r + 1, hidden),
                                row_at<DT>(wu, r + 1, hidden) };
        float gu[4];
        rows_dot_multi<DT, true, 4>(rows, hidden, xs, hidden, T, lane, gu);
        if (writes) {
            float* a = act + (size_t)t * intermediate;
            a[split_at(r, intermediate)] = (gu[0] / (1.0f + expf(-gu[0]))) * gu[1];
            a[split_at(r + 1, intermediate)] = (gu[2] / (1.0f + expf(-gu[2]))) * gu[3];
        }
    }
}

/* head_kernel for T tokens: x and logits hold T rows. Of n_rows rows,
 * blockIdx.y takes the group of up to T from blockIdx.y * T. */
template <vitna_dtype_t DT>
__global__ void head_multi_kernel(const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
                                  const void* __restrict__ w, float* __restrict__ logits, int hidden, int vocab, int T, int n_rows) {
    extern __shared__ float4 shared4[];
    __shared__ float red[MULTI_MAX * WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    const int r0 = blockIdx.y * T;
    T = min(T, n_rows - r0);
    x += (size_t)r0 * hidden;
    logits += (size_t)r0 * vocab;
    norm_multi_to_shared(x, norm_w, xs, hidden, eps, red, T);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    for (int r = blockIdx.x * WARPS + warp; r < vocab; r += gridDim.x * WARPS) {
        const void* row[1] = { row_at<DT>(w, r, hidden) };
        float s[1];
        rows_dot_multi<DT, true, 1>(row, hidden, xs, hidden, T, lane, s);
        if (writes) logits[(size_t)t * vocab + r] = s[0];
    }
}

/* --- A mixture's rows (vitna_cuda_moe_rows_*) ---
 *
 * Many rows of a mixture of experts, a layer at a time, each row's
 * arithmetic its step's, in its step's order, so that every value is the one
 * those steps would give, bit for bit: the multi-row kernels above for the
 * attention, and those below for the router and the experts. An expert runs
 * once for all the rows routed to it, so it is copied to the device once a
 * layer for all of them, where steps would want it once a token. */

/* RMSNorm of x's n_rows rows (stride hidden) into out, as split rows, each
 * as norm_to_shared computes its one: block b takes the group of up to T
 * rows from b * T, through shared memory. */
__global__ void norm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps, float* __restrict__ out, int hidden,
                                 int T, int n_rows) {
    extern __shared__ float4 shared4[];
    __shared__ float red[MULTI_MAX * WARPS];
    float* xs = reinterpret_cast<float*>(shared4);
    const int r0 = blockIdx.x * T;
    T = min(T, n_rows - r0);
    norm_multi_to_shared(x + (size_t)r0 * hidden, w, xs, hidden, eps, red, T);
    float4* o = reinterpret_cast<float4*>(out + (size_t)r0 * hidden);
    for (int i = threadIdx.x; i < T * hidden / 4; i += THREADS) o[i] = shared4[i];
}

/* The router's logits for n_rows rows, from their normalized split rows xn
 * (stride hidden): row_dot's sums, as router_kernel's, for groups of up to
 * MULTI_MAX rows, blockIdx.y the group. Blocks with blockIdx.z 1 do the same
 * for the next layer's router from xn_next, into out + stride_rows *
 * n_experts: the guess. */
template <vitna_dtype_t DT>
__global__ void router_rows_kernel(const float* __restrict__ xn, const float* __restrict__ xn_next, const void* __restrict__ w,
                                   const void* __restrict__ next_w, float* __restrict__ out, int hidden, int n_experts, int n_rows,
                                   int stride_rows) {
    const bool next = blockIdx.z == 1;
    const int r0 = blockIdx.y * MULTI_MAX;
    const int T = min(MULTI_MAX, n_rows - r0);
    const float* x = (next ? xn_next : xn) + (size_t)r0 * hidden;
    const void* router = next ? next_w : w;
    float* o = out + ((size_t)(next ? stride_rows : 0) + r0) * n_experts;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    for (int r = blockIdx.x * WARPS + warp; r < n_experts; r += gridDim.x * WARPS) {
        const void* row[1] = { row_at<DT>(router, r, hidden) };
        float s[1];
        rows_dot_multi<DT, true, 1>(row, hidden, x, hidden, T, lane, s);
        if (writes) o[(size_t)t * n_experts + r] = s[0];
    }
}

/* Rows idx[0] to idx[n - 1] of src (stride hidden) into dst, one after another: blockIdx.y the row. */
__global__ void gather_rows_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx, int hidden, float* __restrict__ dst) {
    const float4* s = reinterpret_cast<const float4*>(src + (size_t)idx[blockIdx.y] * hidden);
    float4* d = reinterpret_cast<float4*>(dst + (size_t)blockIdx.y * hidden);
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < hidden / 4; i += gridDim.x * THREADS) d[i] = s[i];
}

/* An expert's gate and up projections for the n_rows rows routed to it,
 * from their normalized split rows x (stride hidden), as experts_in_kernel
 * computes a token's: groups of up to MULTI_MAX rows, blockIdx.y the group.
 * A warp takes rows r and r + 1 of both matrices, as mlp_in_multi_kernel
 * does, and act gets silu(gate) * up, as split rows (stride intermediate). */
template <vitna_dtype_t DT>
__global__ void __launch_bounds__(THREADS, 3) experts_in_rows_kernel(const float* __restrict__ x, const void* __restrict__ wg,
                                                                     const void* __restrict__ wu, float* __restrict__ act, int hidden,
                                                                     int intermediate, int n_rows) {
    const int r0 = blockIdx.y * MULTI_MAX;
    const int T = min(MULTI_MAX, n_rows - r0);
    x += (size_t)r0 * hidden;
    act += (size_t)r0 * intermediate;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    for (int r = 2 * (blockIdx.x * WARPS + warp); r < intermediate; r += 2 * gridDim.x * WARPS) {
        const void* rows[4] = { row_at<DT>(wg, r, hidden), row_at<DT>(wu, r, hidden), row_at<DT>(wg, r + 1, hidden),
                                row_at<DT>(wu, r + 1, hidden) };
        float gu[4];
        rows_dot_multi<DT, true, 4>(rows, hidden, x, hidden, T, lane, gu);
        if (writes) {
            float* a = act + (size_t)t * intermediate;
            a[split_at(r, intermediate)] = expert_act(gu[0], gu[1]);
            a[split_at(r + 1, intermediate)] = expert_act(gu[2], gu[3]);
        }
    }
}

/* An expert's down projection for the n_rows rows routed to it (act, split,
 * stride intermediate), each output multiplied by that row's weight w[t] and
 * rounded, then added to row idx[t] of proj and rounded, as
 * experts_down_kernel adds an expert into a token's sum: proj starts at
 * zero, and the experts come in the order the step adds them, so each row's
 * sum is its step's. Groups of up to MULTI_MAX rows, blockIdx.y the group. */
template <vitna_dtype_t DT>
__global__ void experts_down_rows_kernel(const float* __restrict__ act, const void* __restrict__ wd, const int32_t* __restrict__ idx,
                                         const float* __restrict__ w, float* __restrict__ proj, int hidden, int intermediate, int n_rows) {
    const int r0 = blockIdx.y * MULTI_MAX;
    const int T = min(MULTI_MAX, n_rows - r0);
    act += (size_t)r0 * intermediate;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane / MULTI_LANES;
    const bool writes = lane % MULTI_LANES == 0 && t < T;
    float* out = writes ? proj + (size_t)idx[r0 + t] * hidden : proj;
    const float wt = writes ? w[r0 + t] : 0.0f;
    for (int r = blockIdx.x * WARPS + warp; r < hidden; r += gridDim.x * WARPS) {
        const void* row[1] = { row_at<DT>(wd, r, intermediate) };
        float s[1];
        rows_dot_multi<DT, true, 1>(row, intermediate, act, intermediate, T, lane, s);
        if (writes) out[r] = __fadd_rn(out[r], __fmul_rn(s[0], wt));
    }
}

/* x += proj over n floats: the experts' sums added to the residual, as experts_down_kernel adds a token's. */
__global__ void add_rows_kernel(float* __restrict__ x, const float* __restrict__ proj, size_t n) {
    const size_t i = (size_t)blockIdx.x * THREADS + threadIdx.x;
    if (i < n) x[i] = x[i] + proj[i];
}

/* --- A prompt, many tokens at once --- */

/* Row t of x is token t's embedding row, widened to float32. */
template <vitna_dtype_t DT>
__global__ void embed_rows_kernel(const void* __restrict__ table, const int32_t* __restrict__ tokens, int hidden, float* __restrict__ x) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) x[(size_t)t * hidden + i] = widen_at<DT>(row_at<DT>(table, tokens[t], hidden), i);
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

/* Weights i to i + 3 of a row, widened to float32 exactly; i is a multiple
 * of 4. A block format's four come from widen_at, one at a time. */
template <vitna_dtype_t DT>
__device__ __forceinline__ void load4(const void* row, int i, float out[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) out[k] = widen_at<DT>(row, i + k);
}

template <>
__device__ __forceinline__ void load4<VITNA_DTYPE_F32>(const void* row, int i, float out[4]) {
    const float4 p = static_cast<const float4*>(row)[i / 4];
    out[0] = p.x; out[1] = p.y; out[2] = p.z; out[3] = p.w;
}

template <>
__device__ __forceinline__ void load4<VITNA_DTYPE_BF16>(const void* row, int i, float out[4]) {
    const uint2 p = static_cast<const uint2*>(row)[i / 4];
    out[0] = bf16_low(p.x); out[1] = bf16_high(p.x);
    out[2] = bf16_low(p.y); out[3] = bf16_high(p.y);
}

template <>
__device__ __forceinline__ void load4<VITNA_DTYPE_F16>(const void* row, int i, float out[4]) {
    const uint2 p = static_cast<const uint2*>(row)[i / 4];
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
    const void* wr = wrow ? row_at<DT>(W, n0 + lr, K) : W;
    if (xrow) xv = *reinterpret_cast<const float4*>(X + (size_t)(t0 + lr) * ldx + lk);
    if (wrow) load4<DT>(wr, lk, wv);
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
            if (wrow) load4<DT>(wr, k0 + GEMM_BK + lk, wv);
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
                load8<DT>(row_at<DT>(W, n, K), c, wv);
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

/* Causal attention for a prompt's tokens, tiled as a matrix product is, for
 * head_dim ATT_D. Block (x, h) takes query head h and ATT_BQ of the chunk's
 * tokens, the last tokens first, since they read the most positions. It goes
 * through the positions ATT_BK at a time, staging each tile's keys
 * (transposed) and values in shared memory once for all its rows, and reads
 * the next tile into registers while this one computes. Thread
 * (ty, tx) scores the ATT_RM rows from ATT_RM ty against the ATT_PX
 * positions from ATT_PX tx, masked after each row's own position, and keeps
 * dimensions 4 tx to 4 tx + 3 of those rows' weighted values. Each row keeps
 * its softmax as it goes, a running max and sum across the 16 threads that
 * share it, with the sum and the weighted values scaled by exp(old max - new
 * max) when the max moves, and the tile's weights go through shared memory to
 * the values' product. That is model.c's softmax, summed in another order,
 * and a row's sums run the same way whatever chunk it came in. */
#define ATT_D 64
#define ATT_RM 4
#define ATT_PX 2
#define ATT_BQ (16 * ATT_RM)
#define ATT_BK (16 * ATT_PX)
static_assert(THREADS == 256 && ATT_D == 64 && ATT_RM % 4 == 0 && (ATT_PX == 2 || ATT_PX == 4),
              "16 by 16 threads, each 4 dimensions, rows four at a time, positions two or four at a time");

/* The dynamic shared memory: the queries, transposed, ATT_D rows of ATT_BQ +
 * 4; a tile's keys, transposed, ATT_D rows of ATT_BK + 4; its values, ATT_BK
 * rows of ATT_D + 4; and its weights, ATT_BQ rows of ATT_BK + 4. */
#define ATT_SHARED \
    ((size_t)(ATT_D * (ATT_BQ + 4) + ATT_D * (ATT_BK + 4) + ATT_BK * (ATT_D + 4) + ATT_BQ * (ATT_BK + 4)) * sizeof(float))
static_assert(ATT_SHARED <= 48 * 1024, "within what a block gets without asking for more");

/* ATT_PX floats from shared memory, in one load. */
__device__ __forceinline__ void load_px(const float* p, float out[ATT_PX]) {
#if ATT_PX == 4
    const float4 v = *reinterpret_cast<const float4*>(p);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
#else
    const float2 v = *reinterpret_cast<const float2*>(p);
    out[0] = v.x;
    out[1] = v.y;
#endif
}

__device__ __forceinline__ void store_px(float* p, const float in[ATT_PX]) {
#if ATT_PX == 4
    *reinterpret_cast<float4*>(p) = make_float4(in[0], in[1], in[2], in[3]);
#else
    *reinterpret_cast<float2*>(p) = make_float2(in[0], in[1]);
#endif
}

/* A tile's keys and values come ATT_SLOTS float4s of each to a thread. */
#define ATT_SLOTS (ATT_BK * (ATT_D / 4) / THREADS)
static_assert(ATT_BK * (ATT_D / 4) % THREADS == 0, "a tile's float4s divide among the threads");

/* This thread's share of the tile at position k0, zeros past end: masked, and a weight of 0. */
__device__ __forceinline__ void fetch_tile(const float* __restrict__ kc, const float* __restrict__ vc, int k0, int end, int kv_dim,
                                           int kvh, int tid, float4 kr[ATT_SLOTS], float4 vr[ATT_SLOTS]) {
#pragma unroll
    for (int u = 0; u < ATT_SLOTS; u++) {
        const int i = tid + u * THREADS;
        const int j = i / (ATT_D / 4), c = i % (ATT_D / 4);
        kr[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        vr[u] = kr[u];
        if (k0 + j < end) {
            const size_t src = (size_t)(k0 + j) * kv_dim + (size_t)kvh * ATT_D + 4 * c;
            kr[u] = *reinterpret_cast<const float4*>(kc + src);
            vr[u] = *reinterpret_cast<const float4*>(vc + src);
        }
    }
}

__global__ void __launch_bounds__(THREADS) attention_tiled_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                                                  const float* __restrict__ vc, float* __restrict__ att, int p0,
                                                                  int T, int kv_dim, int q_dim, int group, float scale) {
    extern __shared__ float4 shared4[];
    const int QS = ATT_BQ + 4, KS = ATT_BK + 4, VS = ATT_D + 4;
    float* qs = reinterpret_cast<float*>(shared4); /* [ATT_D][QS] */
    float* ks = qs + ATT_D * QS;                   /* [ATT_D][KS] */
    float* vs = ks + ATT_D * KS;                   /* [ATT_BK][VS] */
    float* ps = vs + ATT_BK * VS;                  /* [ATT_BQ][KS] */
    const int h = blockIdx.y, kvh = h / group;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * ATT_BQ; /* the block's first token in the chunk */
    const int rows = min(ATT_BQ, T - t0);
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int r0 = ty * ATT_RM;     /* the thread's first row */
    const int end = p0 + t0 + rows; /* the positions the block reads: 0 to end - 1 */

    float4 kr[ATT_SLOTS], vr[ATT_SLOTS];
    fetch_tile(kc, vc, 0, end, kv_dim, kvh, tid, kr, vr); /* the first tile, read while the queries are staged */

    for (int i = tid; i < ATT_BQ * (ATT_D / 4); i += THREADS) {
        const int r = i / (ATT_D / 4), c = i % (ATT_D / 4);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < rows) v = *reinterpret_cast<const float4*>(q + (size_t)(t0 + r) * q_dim + (size_t)h * ATT_D + 4 * c);
        qs[(4 * c) * QS + r] = v.x;
        qs[(4 * c + 1) * QS + r] = v.y;
        qs[(4 * c + 2) * QS + r] = v.z;
        qs[(4 * c + 3) * QS + r] = v.w;
    }

    float m[ATT_RM], l[ATT_RM], o[ATT_RM][4];
#pragma unroll
    for (int i = 0; i < ATT_RM; i++) {
        m[i] = NEG_INF;
        l[i] = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; k++) o[i][k] = 0.0f;
    }
    for (int k0 = 0; k0 < end; k0 += ATT_BK) {
        __syncthreads(); /* the previous tile has been read, and the queries written */
#pragma unroll
        for (int u = 0; u < ATT_SLOTS; u++) {
            const int i = tid + u * THREADS;
            const int j = i / (ATT_D / 4), c = i % (ATT_D / 4);
            ks[(4 * c) * KS + j] = kr[u].x;
            ks[(4 * c + 1) * KS + j] = kr[u].y;
            ks[(4 * c + 2) * KS + j] = kr[u].z;
            ks[(4 * c + 3) * KS + j] = kr[u].w;
            *reinterpret_cast<float4*>(vs + j * VS + 4 * c) = vr[u];
        }
        __syncthreads();
        /* The next tile, in flight while this one computes. */
        if (k0 + ATT_BK < end) fetch_tile(kc, vc, k0 + ATT_BK, end, kv_dim, kvh, tid, kr, vr);

        float s[ATT_RM][ATT_PX];
#pragma unroll
        for (int i = 0; i < ATT_RM; i++) {
#pragma unroll
            for (int jj = 0; jj < ATT_PX; jj++) s[i][jj] = 0.0f;
        }
#pragma unroll 8
        for (int d = 0; d < ATT_D; d++) {
            float b[ATT_PX];
            load_px(ks + d * KS + ATT_PX * tx, b);
#pragma unroll
            for (int i4 = 0; i4 < ATT_RM; i4 += 4) {
                const float4 a = *reinterpret_cast<const float4*>(qs + d * QS + r0 + i4);
                const float av[4] = { a.x, a.y, a.z, a.w };
#pragma unroll
                for (int i = 0; i < 4; i++) {
#pragma unroll
                    for (int jj = 0; jj < ATT_PX; jj++) s[i4 + i][jj] = fmaf(av[i], b[jj], s[i4 + i][jj]);
                }
            }
        }

        /* The running softmax of each row. A row with nothing valid yet keeps
         * a max of minus infinity, and its scale is then 1, not exp(nan). */
#pragma unroll
        for (int i = 0; i < ATT_RM; i++) {
            const int r = r0 + i;
            const int pos = p0 + t0 + r; /* the row's own position */
            float sc[ATT_PX];
            float mt = NEG_INF;
#pragma unroll
            for (int jj = 0; jj < ATT_PX; jj++) {
                const int kp = k0 + ATT_PX * tx + jj;
                sc[jj] = (r < rows && kp <= pos) ? s[i][jj] * scale : NEG_INF;
                mt = fmaxf(mt, sc[jj]);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
            const float mnew = fmaxf(m[i], mt);
            float e[ATT_PX], sum = 0.0f;
#pragma unroll
            for (int jj = 0; jj < ATT_PX; jj++) {
                e[jj] = sc[jj] == NEG_INF ? 0.0f : expf(sc[jj] - mnew);
                sum += e[jj];
            }
            store_px(ps + r * KS + ATT_PX * tx, e);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            const float a = mnew == m[i] ? 1.0f : expf(m[i] - mnew); /* 1 while the max stays, 0 at a row's first valid tile */
            l[i] = l[i] * a + sum;
            m[i] = mnew;
#pragma unroll
            for (int k = 0; k < 4; k++) o[i][k] *= a;
        }
        __syncthreads();

        /* The weighted values, four positions at a time: each row's four
         * weights in one load, the same for all 16 threads of the row. */
#pragma unroll 2
        for (int j = 0; j < ATT_BK; j += 4) {
            float4 v[4];
#pragma unroll
            for (int jj = 0; jj < 4; jj++) v[jj] = *reinterpret_cast<const float4*>(vs + (j + jj) * VS + tx * 4);
#pragma unroll
            for (int i = 0; i < ATT_RM; i++) {
                const float4 p = *reinterpret_cast<const float4*>(ps + (r0 + i) * KS + j);
                const float pv[4] = { p.x, p.y, p.z, p.w };
#pragma unroll
                for (int jj = 0; jj < 4; jj++) {
                    o[i][0] = fmaf(pv[jj], v[jj].x, o[i][0]);
                    o[i][1] = fmaf(pv[jj], v[jj].y, o[i][1]);
                    o[i][2] = fmaf(pv[jj], v[jj].z, o[i][2]);
                    o[i][3] = fmaf(pv[jj], v[jj].w, o[i][3]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < ATT_RM; i++) {
        const int r = r0 + i;
        if (r < rows) {
            const float4 out = make_float4(o[i][0] / l[i], o[i][1] / l[i], o[i][2] / l[i], o[i][3] / l[i]);
            *reinterpret_cast<float4*>(att + (size_t)(t0 + r) * q_dim + (size_t)h * ATT_D + tx * 4) = out;
        }
    }
}

/* --- Launches --- */

/* KERNEL<DT><<<...>>>(...), for the dtype of the matrix the kernel reads.
 * check_shapes refuses on load a model with a matrix in any other dtype, so
 * the last case is never reached; it stops the process rather than run
 * nothing and leave the output unwritten. */
#define LAUNCH(dtype, KERNEL, grid, shared, stream, ...)                                          \
    do {                                                                                          \
        switch (dtype) {                                                                          \
            case VITNA_DTYPE_F32: KERNEL<VITNA_DTYPE_F32><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break;   \
            case VITNA_DTYPE_BF16: KERNEL<VITNA_DTYPE_BF16><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
            case VITNA_DTYPE_F16: KERNEL<VITNA_DTYPE_F16><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break;   \
            case VITNA_DTYPE_Q8_0: KERNEL<VITNA_DTYPE_Q8_0><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
            case VITNA_DTYPE_Q4_K: KERNEL<VITNA_DTYPE_Q4_K><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
            case VITNA_DTYPE_Q6_K: KERNEL<VITNA_DTYPE_Q6_K><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
            default: unreachable_dtype(dtype, #KERNEL);                                           \
        }                                                                                         \
    } while (0)

/* KERNEL<DT, DTV> for a layer's query and key dtype qk and value dtype v:
 * the two equal and a plain dtype, or each a block format. check_shapes
 * refuses any other pairing on load. */
#define LAUNCH_QKV_Q(KERNEL, A, v, grid, shared, stream, ...)                                     \
    switch (v) {                                                                                  \
        case VITNA_DTYPE_Q8_0: KERNEL<A, VITNA_DTYPE_Q8_0><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
        case VITNA_DTYPE_Q4_K: KERNEL<A, VITNA_DTYPE_Q4_K><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
        case VITNA_DTYPE_Q6_K: KERNEL<A, VITNA_DTYPE_Q6_K><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); break; \
        default: unreachable_dtype(v, #KERNEL);                                                   \
    }
#define LAUNCH_QKV(qk, v, KERNEL, grid, shared, stream, ...)                                      \
    do {                                                                                          \
        if ((qk) == (v) && (qk) == VITNA_DTYPE_F32) {                                             \
            KERNEL<VITNA_DTYPE_F32, VITNA_DTYPE_F32><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); \
        } else if ((qk) == (v) && (qk) == VITNA_DTYPE_BF16) {                                     \
            KERNEL<VITNA_DTYPE_BF16, VITNA_DTYPE_BF16><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); \
        } else if ((qk) == (v) && (qk) == VITNA_DTYPE_F16) {                                      \
            KERNEL<VITNA_DTYPE_F16, VITNA_DTYPE_F16><<<grid, THREADS, shared, stream>>>(__VA_ARGS__); \
        } else {                                                                                  \
            switch (qk) {                                                                         \
                case VITNA_DTYPE_Q8_0: LAUNCH_QKV_Q(KERNEL, VITNA_DTYPE_Q8_0, v, grid, shared, stream, __VA_ARGS__) break; \
                case VITNA_DTYPE_Q4_K: LAUNCH_QKV_Q(KERNEL, VITNA_DTYPE_Q4_K, v, grid, shared, stream, __VA_ARGS__) break; \
                case VITNA_DTYPE_Q6_K: LAUNCH_QKV_Q(KERNEL, VITNA_DTYPE_Q6_K, v, grid, shared, stream, __VA_ARGS__) break; \
                default: unreachable_dtype(qk, #KERNEL);                                          \
            }                                                                                     \
        }                                                                                         \
    } while (0)

static void unreachable_dtype(vitna_dtype_t dtype, const char* kernel) {
    fprintf(stderr, "%s has no instance for weights in %s, which loading should have refused\n", kernel, vitna_dtype_name(dtype));
    abort();
}

/* Whether a layer's query and key dtype and its value dtype pair as
 * LAUNCH_QKV instantiates them. */
static bool qkv_pairs(vitna_dtype_t qk, vitna_dtype_t v) {
    if (vitna_dtype_is_block(qk) && vitna_dtype_is_block(v)) return true;
    return qk == v && (qk == VITNA_DTYPE_F32 || qk == VITNA_DTYPE_BF16 || qk == VITNA_DTYPE_F16);
}

/* Whether the kernels have an instance for a matrix in dtype. */
static bool known_dtype(vitna_dtype_t dtype) {
    return dtype == VITNA_DTYPE_F32 || dtype == VITNA_DTYPE_BF16 || dtype == VITNA_DTYPE_F16 || vitna_dtype_is_block(dtype);
}

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

/* Layer l's attention for the token g->rows names, in model.c's order: RMSNorm
 * and the query, key and value projections, QK-norm where the model has it,
 * the rotary embedding, attention over the cache, and the output projection
 * added to the residual. */
/* Layer L's QK-norm and rotation for n rows of q and k_raw, the keys into
 * the cache: across the heads (qk_norm_rope_kernel) or a head at a time
 * (qk_head_norm_rope_kernel), as the model normalizes them. */
static void enqueue_qk_norm(const struct vitna_cuda_model* g, cudaStream_t s, const dlayer_t* L, float* q, const float* k_raw, float* kc,
                            const row_t* rows, size_t seq_stride, int n) {
    const int hd = g->head_dim;
    const size_t q_dim = (size_t)g->n_heads * hd, kv_dim = (size_t)g->n_kv_heads * hd;
    if (g->qk_head_norm) {
        qk_head_norm_rope_kernel<<<dim3(g->n_heads + g->n_kv_heads, n), THREADS, (size_t)hd * sizeof(float), s>>>(
            q, k_raw, kc, L->q_norm, L->k_norm, g->eps, g->cos_t, g->sin_t, rows, seq_stride, g->n_heads, g->n_kv_heads, hd);
    } else {
        qk_norm_rope_kernel<<<dim3(2, n), THREADS, (q_dim > kv_dim ? q_dim : kv_dim) * sizeof(float), s>>>(
            q, k_raw, kc, L->q_norm, L->k_norm, g->eps, g->cos_t, g->sin_t, rows, seq_stride, g->n_heads, g->n_kv_heads, hd);
    }
}

static void enqueue_attention(const struct vitna_cuda_model* g, int l) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden, hd = g->head_dim, half = hd / 2;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const int group = g->n_heads / g->n_kv_heads;
    const size_t seq_stride = (size_t)g->ctx * kv_dim;
    const dlayer_t* L = &g->layers[l];
    float* kc = g->k_cache + (size_t)l * g->seqs * seq_stride;
    float* vc = g->v_cache + (size_t)l * g->seqs * seq_stride;

    LAUNCH_QKV(L->q.dtype, L->v.dtype, attn_in_kernel, grid_for(g, (g->n_heads + g->n_kv_heads) * half + kv_dim / 2), (size_t)H * sizeof(float),
               s, g->x, L->attn_norm, g->eps, L->q.w, L->k.w, L->v.w, g->q, kc, vc, g->cos_t, g->sin_t, g->rows, seq_stride, H, g->n_heads,
               g->n_kv_heads, hd, L->q_norm ? g->k_raw : (float*)NULL);
    if (L->q_norm) enqueue_qk_norm(g, s, L, g->q, g->k_raw, kc, g->rows, seq_stride, 1);
    attention_kernel<<<dim3(1, g->n_kv_heads, ATTENTION_SPLITS), THREADS, attention_shared(g), s>>>(
        g->q, kc, vc, g->att, g->part_m, g->part_l, g->part_o, g->done, g->rows, seq_stride, hd, kv_dim, group, g->scale, 0);
    LAUNCH(L->o.dtype, matvec_add_kernel, grid_for(g, H), 0, s, L->o.w, g->att, g->x, H, q_dim);
}

/* The embedding and every layer, in model.c's order. Captured once, into g->body. */
static void enqueue_body(const struct vitna_cuda_model* g) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden;
    const size_t norm_shared = (size_t)H * sizeof(float);
    const unsigned int mlp_in_blocks = grid_for(g, g->intermediate);
    const unsigned int out_blocks = grid_for(g, H);
    const unsigned int embed_blocks = blocks_for((size_t)H, THREADS);

    LAUNCH(g->embed.dtype, embed_kernel, embed_blocks, 0, s, g->embed.w, g->rows, H, g->x);
    for (int l = 0; l < g->n_layers; l++) {
        const dlayer_t* L = &g->layers[l];
        enqueue_attention(g, l);
        LAUNCH(L->gate.dtype, mlp_in_kernel, mlp_in_blocks, norm_shared, s, g->x, L->mlp_norm, g->eps, L->gate.w, L->up.w, g->act, H, g->intermediate);
        LAUNCH(L->down.dtype, matvec_add_kernel, out_blocks, 0, s, L->down.w, g->act, g->x, H, g->intermediate);
    }
}

/* The final RMSNorm and the output projection of the row at x, into g->logits. */
static void enqueue_head_of(const struct vitna_cuda_model* g, const float* x) {
    LAUNCH(g->lm_head.dtype, head_kernel, grid_for(g, g->vocab), (size_t)g->hidden * sizeof(float), g->stream,
           x, g->final_norm, g->eps, g->lm_head.w, g->logits, g->hidden, g->vocab);
}

/* The head of a step. Captured once, into g->head. */
static void enqueue_head(const struct vitna_cuda_model* g) {
    enqueue_head_of(g, g->x);
}

/* g->capture_T rows at once, as g->rows name them: the embedding, every
 * layer and the head, as enqueue_body and enqueue_head run one token, with
 * the kernels for several (vitna_cuda_rows). Rows go in the prompt's
 * scratch: p_x, p_q, p_att, p_gate for the activations and p_logits for the
 * logits. Captured once for each count. */
static void enqueue_multi(const struct vitna_cuda_model* g) {
    const cudaStream_t s = g->stream;
    const int T = g->capture_T;
    const int H = g->hidden, hd = g->head_dim, half = hd / 2;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const int group = g->n_heads / g->n_kv_heads;
    const size_t rows_shared = (size_t)T * H * sizeof(float);
    const unsigned int attn_in_blocks = grid_for(g, (g->n_heads + g->n_kv_heads) * half + kv_dim / 2);
    const unsigned int mlp_in_multi_blocks = grid_for(g, g->intermediate / 2); /* a warp takes two rows */
    const unsigned int out_blocks = grid_for(g, H);

    const size_t seq_stride = (size_t)g->ctx * kv_dim;

    LAUNCH(g->embed.dtype, embed_multi_kernel, dim3(blocks_for((size_t)H, THREADS), T), 0, s, g->embed.w, g->rows, H, g->p_x);
    for (int l = 0; l < g->n_layers; l++) {
        const dlayer_t* L = &g->layers[l];
        float* kc = g->k_cache + (size_t)l * g->seqs * seq_stride;
        float* vc = g->v_cache + (size_t)l * g->seqs * seq_stride;
        LAUNCH_QKV(L->q.dtype, L->v.dtype, attn_in_multi_kernel, attn_in_blocks, rows_shared, s, g->p_x, L->attn_norm, g->eps, L->q.w, L->k.w,
                   L->v.w, g->p_q, kc, vc, g->cos_t, g->sin_t, g->rows, seq_stride, H, g->n_heads, g->n_kv_heads, hd, T, T, (float*)NULL);
        attention_kernel<<<dim3(T, g->n_kv_heads, ATTENTION_SPLITS), THREADS, attention_shared(g), s>>>(
            g->p_q, kc, vc, g->p_att, g->part_m, g->part_l, g->part_o, g->done, g->rows, seq_stride, hd, kv_dim, group, g->scale, 1);
        LAUNCH(L->o.dtype, matvec_add_multi_kernel, out_blocks, 0, s, L->o.w, g->p_att, g->p_x, H, q_dim, T, T);
        LAUNCH(L->gate.dtype, mlp_in_multi_kernel, mlp_in_multi_blocks, rows_shared, s, g->p_x, L->mlp_norm, g->eps, L->gate.w, L->up.w, g->p_gate,
               H, g->intermediate, T);
        LAUNCH(L->down.dtype, matvec_add_multi_kernel, out_blocks, 0, s, L->down.w, g->p_gate, g->p_x, H, g->intermediate, T, T);
    }
    LAUNCH(g->lm_head.dtype, head_multi_kernel, grid_for(g, g->vocab), rows_shared, s, g->p_x, g->final_norm, g->eps, g->lm_head.w, g->p_logits,
           H, g->vocab, T, T);
}

/* gemm_few_kernel<DT, FEW_MR, FEW_TB>, whose template takes two more arguments than LAUNCH passes. */
template <vitna_dtype_t DT>
static void launch_gemm_few(dim3 grid, cudaStream_t s, const float* X, int ldx, const void* W, float* Y, int ldy, int T, int N, int K, int add) {
    gemm_few_kernel<DT, FEW_MR, FEW_TB><<<grid, THREADS, 0, s>>>(X, ldx, W, Y, ldy, T, N, K, add);
}

/* Y = X W^T for T rows, on the model's stream: gemm_few_kernel for up to
 * FEW_TOKENS rows, which a tile of GEMM_BM would mostly leave empty, and
 * gemm_kernel for more. */
static void gemm(const struct vitna_cuda_model* g, const float* X, int ldx, const dmat_t* W, float* Y, int ldy, int T, int N, int K,
                 int add) {
    if (T <= FEW_TOKENS) {
        const dim3 few(blocks_for((size_t)T, FEW_TB), blocks_for((size_t)N, WARPS * FEW_MR));
        switch (W->dtype) {
            case VITNA_DTYPE_F32: launch_gemm_few<VITNA_DTYPE_F32>(few, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add); break;
            case VITNA_DTYPE_BF16: launch_gemm_few<VITNA_DTYPE_BF16>(few, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add); break;
            case VITNA_DTYPE_F16: launch_gemm_few<VITNA_DTYPE_F16>(few, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add); break;
            case VITNA_DTYPE_Q8_0: launch_gemm_few<VITNA_DTYPE_Q8_0>(few, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add); break;
            case VITNA_DTYPE_Q4_K: launch_gemm_few<VITNA_DTYPE_Q4_K>(few, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add); break;
            case VITNA_DTYPE_Q6_K: launch_gemm_few<VITNA_DTYPE_Q6_K>(few, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add); break;
            default: unreachable_dtype(W->dtype, "gemm_few_kernel");
        }
        return;
    }
    const dim3 grid(blocks_for((size_t)N, GEMM_BN), blocks_for((size_t)T, GEMM_BM));
    LAUNCH(W->dtype, gemm_kernel, grid, 0, g->stream, X, ldx, W->w, Y, ldy, T, N, K, add);
}

/* Whether a prompt can run many tokens at once: widths the matrix-matrix
 * product's GEMM_BK divides, and head_dim ATT_D, the attention kernel's. A
 * model that fails runs its prompts a token at a time. */
static bool prompt_batches(const struct vitna_cuda_model* g) {
    return g->hidden % GEMM_BK == 0 && g->intermediate % GEMM_BK == 0 && (g->n_heads * g->head_dim) % GEMM_BK == 0 &&
           g->head_dim == ATT_D;
}

/* A prompt's attention, for T tokens at positions p0 onwards. */
static void enqueue_prompt_attention(const struct vitna_cuda_model* g, const float* kc, const float* vc, int p0, int T) {
    const int group = g->n_heads / g->n_kv_heads;
    const int q_dim = g->n_heads * g->head_dim, kv_dim = g->n_kv_heads * g->head_dim;
    const dim3 tiles(blocks_for((size_t)T, ATT_BQ), g->n_heads);
    attention_tiled_kernel<<<tiles, THREADS, ATT_SHARED, g->stream>>>(g->p_q, kc, vc, g->p_att, p0, T, kv_dim, q_dim, group, g->scale);
}

/* The embedding and every layer for T tokens of a prompt, g->p_tokens[c0]
 * onwards, at positions p0 onwards of sequence seq, in model.c's order,
 * leaving the last layer's output in g->p_x. Their keys and values go into
 * that sequence's cache. */
static void enqueue_chunk(const struct vitna_cuda_model* g, int seq, size_t c0, int p0, int T) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden, I = g->intermediate, hd = g->head_dim;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const unsigned int rope_blocks = blocks_for((size_t)T * (g->n_heads + g->n_kv_heads) * (hd / 2), THREADS);
    const unsigned int silu_blocks = blocks_for((size_t)T * I, THREADS);

    LAUNCH(g->embed.dtype, embed_rows_kernel, dim3(blocks_for((size_t)H, THREADS), T), 0, s, g->embed.w, g->p_tokens + c0, H, g->p_x);
    for (int l = 0; l < g->n_layers; l++) {
        const dlayer_t* L = &g->layers[l];
        float* kc = g->k_cache + ((size_t)l * g->seqs + seq) * g->ctx * kv_dim;
        float* vc = g->v_cache + ((size_t)l * g->seqs + seq) * g->ctx * kv_dim;

        rmsnorm_rows_kernel<<<T, THREADS, 0, s>>>(g->p_x, L->attn_norm, g->p_xn, H, g->eps);
        gemm(g, g->p_xn, H, &L->q, g->p_q, q_dim, T, q_dim, H, 0);
        gemm(g, g->p_xn, H, &L->k, kc + (size_t)p0 * kv_dim, kv_dim, T, kv_dim, H, 0);
        gemm(g, g->p_xn, H, &L->v, vc + (size_t)p0 * kv_dim, kv_dim, T, kv_dim, H, 0);
        rope_rows_kernel<<<rope_blocks, THREADS, 0, s>>>(g->p_q, kc, g->cos_t, g->sin_t, p0, T, g->n_heads, g->n_kv_heads, hd);
        enqueue_prompt_attention(g, kc, vc, p0, T);
        gemm(g, g->p_att, q_dim, &L->o, g->p_x, H, T, H, q_dim, 1);
        rmsnorm_rows_kernel<<<T, THREADS, 0, s>>>(g->p_x, L->mlp_norm, g->p_xn, H, g->eps);
        gemm(g, g->p_xn, H, &L->gate, g->p_gate, I, T, I, H, 0);
        gemm(g, g->p_xn, H, &L->up, g->p_up, I, T, I, H, 0);
        silu_mul_rows_kernel<<<silu_blocks, THREADS, 0, s>>>(g->p_gate, g->p_up, (size_t)T * I);
        gemm(g, g->p_gate, I, &L->down, g->p_x, H, T, H, I, 1);
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
    return m->rows * (size_t)vitna_row_bytes(m->dtype, m->cols);
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

/* The expert cache's part of vitna_cuda_free: the copies finished and their
 * sources released to the expert stream, which the model closes after
 * this, and the memory registered for them unregistered. */
static void free_experts(struct vitna_cuda_model* g) {
    expert_cache_t* c = &g->ec;
    if (c->copy) cudaStreamSynchronize(c->copy);
    while (c->p_len > 0) {
        vitna_expert_stream_release(c->stream, &c->pending[c->p_head].place, 1);
        c->p_head = (c->p_head + 1) % PENDING_MAX;
        c->p_len--;
    }
    for (int i = 0; i < PENDING_MAX; i++) {
        if (c->pending[i].done) cudaEventDestroy(c->pending[i].done);
    }
    for (int i = 0; c->slots && i < c->n_slots; i++) {
        if (c->slots[i].filled) cudaEventDestroy(c->slots[i].filled);
    }
    for (int i = 0; i < c->n_registered; i++) cudaHostUnregister(c->registered[i]);
    if (c->copy) cudaStreamDestroy(c->copy);
    if (c->mem) cudaFree(c->mem);
    free(c->registered);
    free(c->slots);
    free(c->where);
    free(c->uses);
    free(c->part_bytes);
    free(c->part_off);
    free((void*)c->parts);
    if (g->host_router) cudaFreeHost(g->host_router);
    if (g->h_idx) cudaFreeHost(g->h_idx);
    if (g->h_w) cudaFreeHost(g->h_w);
    if (g->h_rrouter) cudaFreeHost(g->h_rrouter);
    if (g->h_rlogits) cudaFreeHost(g->h_rlogits);
    vitna_pool_free(g->cpu);
    if (c->copy_t0) cudaEventDestroy(c->copy_t0);
    if (c->copy_t1) cudaEventDestroy(c->copy_t1);
    if (g->moe_xs) cudaFree(g->moe_xs);
    if (g->moe_y) cudaFree(g->moe_y);
    if (g->host_xs) cudaFreeHost(g->host_xs);
    if (g->host_y) cudaFreeHost(g->host_y);
    free(g->cpu_act);
    free(g->cpu_scratch);
    free(g->e_count);
    free(g->e_at);
    free(g->e_fill);
    free(g->e_list);
    free(g->e_slot);
    free(g->e_mark);
    free(c->held_list);
}

void vitna_cuda_free(struct vitna_cuda_model* g) {
    if (!g) return;
    cudaSetDevice(g->device);
    if (g->stream) cudaStreamSynchronize(g->stream);
    free_experts(g);
    if (g->body) cudaGraphExecDestroy(g->body);
    if (g->head) cudaGraphExecDestroy(g->head);
    for (int t = 0; t <= MULTI_MAX; t++) {
        if (g->multi[t]) cudaGraphExecDestroy(g->multi[t]);
    }
    if (g->host_multi) cudaFreeHost(g->host_multi);
    if (g->stream) cudaStreamDestroy(g->stream);
    if (g->host_logits) cudaFreeHost(g->host_logits);
    if (g->arena) cudaFree(g->arena);
    free(g->layers);
    free(g);
}

const char* vitna_cuda_device_name(const struct vitna_cuda_model* g) {
    return g ? g->name : "";
}

void vitna_cuda_wait(struct vitna_cuda_model* g) {
    if (!g) return;
    cudaSetDevice(g->device);
    if (g->stream) cudaStreamSynchronize(g->stream);
}

bool vitna_cuda_lost(struct vitna_cuda_model* g, char* err, size_t err_len) {
    /* An error that is not sticky, as a launch refused for its shape, ends
     * here. cudaGetLastError cannot tell the two apart itself: after an
     * illegal address it answers cudaSuccess the second time it is asked,
     * while every launch, copy and wait still fails. */
    cudaGetLastError();
    cudaError_t e = cudaSetDevice(g->device);
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    if (e == cudaSuccess) return false;
    fail(err, err_len, "%s (%s)", cudaGetErrorString(e), cudaGetErrorName(e));
    return true;
}

/* Whether the GPU kernels can take the model: every matrix in a dtype they
 * have an instance for, the matrices one kernel reads in dtypes it pairs (a
 * layer's query and key in one, and its value in the same or, for block
 * formats, any block format; a dense layer's gate and up in one; a layer's
 * experts' gate and up in one, and their down in one; every router in one),
 * widths the 16-byte reads divide, and, for a mixture of experts, experts
 * all of one shape, few enough for a kernel's arguments, whose activations
 * fit in shared memory together. */
static bool check_shapes(const vitna_llama_t* m, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    if (c->hidden % 8 != 0 || c->intermediate % 8 != 0 || (c->n_heads * c->head_dim) % 8 != 0) {
        return fail(err, err_len, "the GPU path needs the hidden, intermediate and attention widths to be multiples of 8");
    }
    if (c->hidden * sizeof(float) > 48 * 1024) {
        return fail(err, err_len, "the GPU path needs hidden to be at most 12288");
    }
    if (c->qk_norm && !c->qk_norm_per_head && (c->n_heads * c->head_dim * sizeof(float) > 48 * 1024 || c->n_kv_heads * c->head_dim * sizeof(float) > 48 * 1024)) {
        return fail(err, err_len, "the GPU path's QK-norm needs n_heads * head_dim to be at most 12288");
    }
    if (c->n_experts && c->n_experts_used > EXPERTS_MAX) {
        return fail(err, err_len, "the GPU path takes a mixture whose tokens go through at most %d experts, and this one's go through %zu",
                    EXPERTS_MAX, c->n_experts_used);
    }
    if (c->n_experts && c->n_experts_used * c->intermediate * sizeof(float) > 48 * 1024) {
        return fail(err, err_len, "the GPU path needs the experts a token goes through times their intermediate width to be at most 12288");
    }
#define KNOWN(mat, what)                                                                                         \
    do {                                                                                                          \
        if (!known_dtype((mat).dtype)) {                                                                          \
            return fail(err, err_len, "the GPU path does not compute with %s in %s", what, vitna_dtype_name((mat).dtype)); \
        }                                                                                                         \
    } while (0)
    KNOWN(m->embed, "the embedding");
    KNOWN(m->lm_head, "the output layer");
    for (size_t l = 0; l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        KNOWN(L->q, "a query projection");
        KNOWN(L->v, "a value projection");
        KNOWN(L->o, "an output projection");
        if (L->k.dtype != L->q.dtype) {
            return fail(err, err_len, "the GPU path needs layer %zu's query and key projections in one dtype, and they are %s and %s", l,
                        vitna_dtype_name(L->q.dtype), vitna_dtype_name(L->k.dtype));
        }
        if (!qkv_pairs(L->q.dtype, L->v.dtype)) {
            return fail(err, err_len, "the GPU path does not pair layer %zu's query and key projections in %s with its value projection in %s", l,
                        vitna_dtype_name(L->q.dtype), vitna_dtype_name(L->v.dtype));
        }
        if (!c->n_experts) {
            KNOWN(L->gate, "an MLP's gate");
            KNOWN(L->down, "an MLP's down projection");
            if (L->up.dtype != L->gate.dtype) {
                return fail(err, err_len, "the GPU path needs layer %zu's gate and up projections in one dtype", l);
            }
            continue;
        }
        KNOWN(L->router, "a router");
        if (L->router.dtype != m->layers[0].router.dtype) {
            return fail(err, err_len, "the GPU path needs every layer's router in one dtype");
        }
        const vitna_expert_t* x0 = &m->layers[0].experts[0];
        const vitna_expert_t* l0 = &L->experts[0];
        KNOWN(l0->gate, "an expert's gate");
        KNOWN(l0->down, "an expert's down projection");
        for (size_t e = 0; e < c->n_experts; e++) {
            const vitna_expert_t* x = &L->experts[e];
            if (x->gate.dtype != l0->gate.dtype || x->up.dtype != l0->gate.dtype || x->down.dtype != l0->down.dtype) {
                return fail(err, err_len, "the GPU path needs a layer's experts' gate and up projections in one dtype, and their down projections in one");
            }
            if (x->gate.rows != x0->gate.rows || x->gate.cols != x0->gate.cols || x->up.rows != x0->up.rows ||
                x->up.cols != x0->up.cols || x->down.rows != x0->down.rows || x->down.cols != x0->down.cols) {
                return fail(err, err_len, "the GPU path needs every expert to have the same shape");
            }
        }
    }
#undef KNOWN
    return true;
}

/* A mixture of experts' cache on the device: as many slots as
 * expert_cache_bytes holds, or for 0 as the device has free less
 * EXPERT_MARGIN, and no more than one for every expert; the stream the
 * copies run on and the events that mark them; and the memory they come
 * from, registered with the device so they run at the bus's full speed. */
static bool create_experts(struct vitna_cuda_model* g, const vitna_llama_t* m, size_t expert_cache_bytes, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    expert_cache_t* ec = &g->ec;
    ec->n_layers = (int)c->n_layers;
    ec->part_bytes = (size_t*)calloc(c->n_layers * 3, sizeof(size_t));
    ec->part_off = (size_t*)calloc(c->n_layers * 3, sizeof(size_t));
    if (!ec->part_bytes || !ec->part_off) return fail(err, err_len, "out of memory");
    ec->slot_bytes = 0;
    for (size_t l = 0; l < c->n_layers; l++) {
        const vitna_expert_t* x0 = &m->layers[l].experts[0];
        size_t* bytes = ec->part_bytes + 3 * l;
        size_t* off = ec->part_off + 3 * l;
        bytes[0] = matrix_bytes(&x0->gate);
        bytes[1] = matrix_bytes(&x0->up);
        bytes[2] = matrix_bytes(&x0->down);
        off[0] = 0;
        off[1] = padded(bytes[0]);
        off[2] = off[1] + padded(bytes[1]);
        const size_t end = off[2] + padded(bytes[2]);
        if (end > ec->slot_bytes) ec->slot_bytes = end;
    }
    ec->n_places = c->n_layers * c->n_experts;
    const size_t need = 2 * c->n_experts_used; /* a layer's experts, and its guess at the next layer's */

    size_t free_b = 0, total_b = 0;
    cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot ask the device what memory it has free", e);
    size_t slots = expert_cache_bytes ? expert_cache_bytes / ec->slot_bytes
                                      : (free_b > EXPERT_MARGIN ? (free_b - EXPERT_MARGIN) / ec->slot_bytes : 0);
    if (slots > ec->n_places) slots = ec->n_places; /* a slot for every expert holds them all */
    if (slots < need || !fits_int(slots)) {
        return fail(err, err_len,
                    "the GPU's expert cache must hold twice the experts a token goes through: %zu MiB or more, with %.0f MiB of the device's %.0f MiB free",
                    (need * ec->slot_bytes + ((size_t)1 << 20) - 1) >> 20, free_b / 1048576.0, total_b / 1048576.0);
    }
    e = cudaMalloc((void**)&ec->mem, slots * ec->slot_bytes);
    if (e != cudaSuccess) {
        ec->mem = NULL;
        cudaGetLastError(); /* clear the allocation error */
        return fail(err, err_len, "the GPU's expert cache of %zu MiB does not fit, with %.0f MiB of the device's %.0f MiB free (%s); --gpu-expert-cache asks for less",
                    (slots * ec->slot_bytes) >> 20, free_b / 1048576.0, total_b / 1048576.0, cudaGetErrorName(e));
    }
    ec->n_slots = (int)slots;
    ec->slots = (eslot_t*)calloc(slots, sizeof(eslot_t));
    ec->where = (int32_t*)malloc(ec->n_places * sizeof(int32_t));
    ec->uses = (uint32_t*)calloc(ec->n_places, sizeof(uint32_t));
    ec->registered = (void**)calloc(m->n_shards + 1, sizeof(void*));
    ec->held_list = (int*)calloc(slots, sizeof(int));
    if (!ec->slots || !ec->where || !ec->uses || !ec->registered || !ec->held_list) return fail(err, err_len, "out of memory");
    for (size_t p = 0; p < ec->n_places; p++) ec->where[p] = -1;
    e = cudaStreamCreateWithFlags(&ec->copy, cudaStreamNonBlocking);
    for (int i = 0; e == cudaSuccess && i < ec->n_slots; i++) {
        ec->slots[i].place = -1;
        e = cudaEventCreateWithFlags(&ec->slots[i].filled, cudaEventDisableTiming);
    }
    for (int i = 0; e == cudaSuccess && i < PENDING_MAX; i++) e = cudaEventCreateWithFlags(&ec->pending[i].done, cudaEventDisableTiming);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot set up the GPU's expert cache", e);

    ec->pinned = true;
    if (m->stream) {
        /* The expert stream's slots. Each copy holds its source there until
         * it finishes, and no more are held than leave a layer's reads room. */
        ec->stream = m->stream;
        const size_t host_slots = vitna_expert_stream_slots(m->stream);
        ec->p_max = host_slots >= need + PENDING_MAX ? PENDING_MAX : (host_slots > need ? (int)(host_slots - need) : 0);
        size_t bytes = 0;
        void* mem = vitna_expert_stream_memory(m->stream, &bytes);
        if (cudaHostRegister(mem, bytes, cudaHostRegisterDefault) == cudaSuccess) {
            ec->registered[ec->n_registered++] = mem;
        } else {
            cudaGetLastError();
            ec->pinned = false;
        }
    } else {
        /* The mapped checkpoint, registered read-only: its pages are read in
         * as the copies need them, and stay the operating system's. */
        ec->parts = (const void**)malloc(ec->n_places * 3 * sizeof(void*));
        if (!ec->parts) return fail(err, err_len, "out of memory");
        for (size_t l = 0; l < c->n_layers; l++) {
            for (size_t x = 0; x < c->n_experts; x++) {
                const vitna_expert_t* ex = &m->layers[l].experts[x];
                const void** p = ec->parts + (l * c->n_experts + x) * 3;
                p[0] = ex->gate.data;
                p[1] = ex->up.data;
                p[2] = ex->down.data;
            }
        }
        int read_only = 0;
        if (cudaDeviceGetAttribute(&read_only, cudaDevAttrHostRegisterReadOnlySupported, g->device) != cudaSuccess) read_only = 0;
        for (size_t i = 0; i < m->n_shards; i++) {
            void* base = m->shards[i].mmap.data;
            const size_t bytes = (m->shards[i].mmap.size + 4095) / 4096 * 4096;
            if (read_only && cudaHostRegister(base, bytes, cudaHostRegisterReadOnly) == cudaSuccess) {
                ec->registered[ec->n_registered++] = base;
            } else {
                cudaGetLastError();
                ec->pinned = false;
            }
        }
    }
    /* Registered memory is made resident for the device at the first work
     * submitted after it, which locks every page, reading in those not in
     * memory: seconds for a whole checkpoint. Submit that now, so loading
     * pays for it rather than the first token. */
    set_row_kernel<<<1, 1, 0, g->stream>>>(g->rows, 0, 0, 0);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    if (e != cudaSuccess) {
        return fail(err, err_len,
                    "the memory the experts are copied from cannot be locked for the device: %s (%s); --expert-cache <MiB> copies them from "
                    "a cache that size in memory instead of the whole checkpoint",
                    cudaGetErrorString(e), cudaGetErrorName(e));
    }
    return true;
}

struct vitna_cuda_model* vitna_cuda_create(const vitna_llama_t* m, const float* cos_tab, const float* sin_tab, size_t expert_cache_bytes,
                                           char* err, size_t err_len) {
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
    const size_t cache_floats = c->n_layers * m->seqs * m->ctx * kv_dim;
    if (!fits_int(c->vocab) || !fits_int(c->hidden) || !fits_int(c->intermediate) || !fits_int(q_dim) ||
        !fits_int(m->ctx) || !fits_int(c->n_layers) || !fits_int(m->seqs)) {
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
    g->seqs = (int)m->seqs;
    g->eps = c->rms_eps;
    g->scale = 1.0f / sqrtf((float)c->head_dim); /* as model.c */
    g->moe = c->n_experts > 0;
    g->qk_norm = c->qk_norm;
    g->qk_head_norm = c->qk_norm_per_head;
    g->n_experts = (int)c->n_experts;
    g->n_used = (int)c->n_experts_used;
    /* A mixture of experts runs a token at a time, its prompts too. */
    g->prompt = !g->moe && prompt_batches(g);

    cudaError_t e = cudaSetDevice(g->device);
    cudaDeviceProp prop;
    if (e == cudaSuccess) e = cudaGetDeviceProperties(&prop, g->device);
    if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaMallocHost((void**)&g->host_logits, c->vocab * sizeof(float));
    if (e == cudaSuccess && g->moe) e = cudaMallocHost((void**)&g->host_router, 2 * c->n_experts * sizeof(float));
    /* A mixture's rows: their routing and their logits come back through pinned memory. */
    if (e == cudaSuccess && g->moe) e = cudaMallocHost((void**)&g->h_idx, (size_t)MOE_ROWS * c->n_experts_used * sizeof(int32_t));
    if (e == cudaSuccess && g->moe) e = cudaMallocHost((void**)&g->h_w, (size_t)MOE_ROWS * c->n_experts_used * sizeof(float));
    if (e == cudaSuccess && g->moe) e = cudaMallocHost((void**)&g->h_rrouter, (size_t)2 * MOE_ROWS * c->n_experts * sizeof(float));
    if (e == cudaSuccess && g->moe) e = cudaMallocHost((void**)&g->h_rlogits, (size_t)LOGIT_ROWS * c->vocab * sizeof(float));
    if (g->moe) {
        g->e_count = (int32_t*)calloc(c->n_experts, sizeof(int32_t));
        g->e_at = (int32_t*)calloc(c->n_experts, sizeof(int32_t));
        g->e_fill = (int32_t*)calloc(c->n_experts, sizeof(int32_t));
        g->e_list = (int32_t*)calloc(c->n_experts, sizeof(int32_t));
        g->e_slot = (int32_t*)calloc(c->n_experts, sizeof(int32_t));
        g->e_mark = (unsigned char*)calloc(c->n_experts, 1);
        if (e == cudaSuccess && (!g->e_count || !g->e_at || !g->e_fill || !g->e_list || !g->e_slot || !g->e_mark)) e = cudaErrorMemoryAllocation;
        /* As many rows as norm_multi_to_shared can hold in 48 KB beside its own reductions. */
        g->rows_g = (int)((48 * 1024 - 1024) / (c->hidden * sizeof(float)));
        if (g->rows_g > MULTI_MAX) g->rows_g = MULTI_MAX;
    }
    /* Several steps at once take up to MULTI_MAX tokens, as many as fit their
     * normalized rows in 48 KB of shared memory, in the prompt's scratch. */
    g->multi_max = g->prompt ? (int)(48 * 1024 / (c->hidden * sizeof(float))) : 0;
    if (g->multi_max > MULTI_MAX) g->multi_max = MULTI_MAX;
    if (g->multi_max < 2) g->multi_max = 0;
    if (e == cudaSuccess && g->multi_max) e = cudaMallocHost((void**)&g->host_multi, (size_t)MULTI_MAX * c->vocab * sizeof(float));
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

    /* One allocation: the weights, the rotary tables, the key-value cache and
     * the scratch. A mixture of experts' experts are not among the weights:
     * they go in the expert cache, allocated after this. */
    const bool tied = m->lm_head.data == m->embed.data;
    size_t total = 0;
    for (size_t l = 0; l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        const vitna_matrix_t* mats[8] = { &L->q, &L->k, &L->v, &L->o, &L->gate, &L->up, &L->down, &L->router };
        for (int i = 0; i < 8; i++) total += padded(matrix_bytes(mats[i])); /* those a model lacks are empty */
        total += 2 * padded(c->hidden * sizeof(float));
        if (c->qk_norm) {
            total += c->qk_norm_per_head ? 2 * padded(c->head_dim * sizeof(float)) : padded(q_dim * sizeof(float)) + padded(kv_dim * sizeof(float));
        }
    }
    if (g->moe) {
        total += padded(kv_dim * sizeof(float)) + padded(2 * c->n_experts * sizeof(float)) +
                 padded(c->n_experts_used * c->intermediate * sizeof(float));
        /* A pass of rows. */
        total += padded(MOE_ROWS * sizeof(row_t)) + 5 * padded((size_t)MOE_ROWS * c->hidden * sizeof(float)) +
                 2 * padded((size_t)MOE_ROWS * q_dim * sizeof(float)) + padded((size_t)MOE_ROWS * kv_dim * sizeof(float)) +
                 padded((size_t)MOE_ROWS * c->intermediate * sizeof(float)) + padded((size_t)2 * MOE_ROWS * c->n_experts * sizeof(float)) +
                 padded((size_t)LOGIT_ROWS * c->vocab * sizeof(float)) + padded((size_t)MOE_ROWS * c->n_experts_used * sizeof(int32_t)) +
                 padded((size_t)MOE_ROWS * c->n_experts_used * sizeof(float));
    }
    total += padded(matrix_bytes(&m->embed));
    if (!tied) total += padded(matrix_bytes(&m->lm_head));
    total += padded(c->hidden * sizeof(float));
    total += 2 * padded(m->ctx * half * sizeof(float));
    total += 2 * padded(cache_floats * sizeof(float));
    /* Attention's slices and counts, for each of up to MULTI_MAX tokens (a step uses the first). */
    const size_t slices = MULTI_MAX * c->n_heads * ATTENTION_SPLITS;
    total += padded(c->hidden * sizeof(float)) + 2 * padded(q_dim * sizeof(float)) + padded(c->intermediate * sizeof(float)) +
             padded(c->vocab * sizeof(float)) + 2 * padded(slices * sizeof(float)) + padded(slices * c->head_dim * sizeof(float)) +
             padded(MULTI_MAX * c->n_kv_heads * sizeof(unsigned int)) + padded(MULTI_MAX * sizeof(row_t));
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
        fail(err, err_len, "the model needs %.1f MiB on %s, and %.1f MiB of its %.1f MiB are free (%s); a smaller --ctx or --parallel needs less",
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
        (dm).dtype = (hm).dtype;                                                                \
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
        if (g->moe) {
            UPLOAD_MATRIX(D->router, L->router);
            D->e_gu = L->experts[0].gate.dtype;
            D->e_down = L->experts[0].down.dtype;
        } else {
            UPLOAD_MATRIX(D->gate, L->gate);
            UPLOAD_MATRIX(D->up, L->up);
            UPLOAD_MATRIX(D->down, L->down);
        }
        UPLOAD_FLOATS(D->attn_norm, L->attn_norm, c->hidden);
        UPLOAD_FLOATS(D->mlp_norm, L->mlp_norm, c->hidden);
        if (c->qk_norm) {
            UPLOAD_FLOATS(D->q_norm, L->q_norm, c->qk_norm_per_head ? c->head_dim : q_dim);
            UPLOAD_FLOATS(D->k_norm, L->k_norm, c->qk_norm_per_head ? c->head_dim : kv_dim);
        }
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
        g->done = (unsigned int*)carve(&cv, MULTI_MAX * c->n_kv_heads * sizeof(unsigned int));
        g->rows = (row_t*)carve(&cv, MULTI_MAX * sizeof(row_t));
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
        if (g->moe) {
            g->k_raw = (float*)carve(&cv, kv_dim * sizeof(float));
            g->router_out = (float*)carve(&cv, 2 * c->n_experts * sizeof(float));
            g->moe_act = (float*)carve(&cv, c->n_experts_used * c->intermediate * sizeof(float));
            g->r_rows = (row_t*)carve(&cv, MOE_ROWS * sizeof(row_t));
            g->r_x = (float*)carve(&cv, (size_t)MOE_ROWS * c->hidden * sizeof(float));
            g->r_xn = (float*)carve(&cv, (size_t)MOE_ROWS * c->hidden * sizeof(float));
            g->r_xn2 = (float*)carve(&cv, (size_t)MOE_ROWS * c->hidden * sizeof(float));
            g->r_xg = (float*)carve(&cv, (size_t)MOE_ROWS * c->hidden * sizeof(float));
            g->r_proj = (float*)carve(&cv, (size_t)MOE_ROWS * c->hidden * sizeof(float));
            g->r_q = (float*)carve(&cv, (size_t)MOE_ROWS * q_dim * sizeof(float));
            g->r_att = (float*)carve(&cv, (size_t)MOE_ROWS * q_dim * sizeof(float));
            g->r_kraw = (float*)carve(&cv, (size_t)MOE_ROWS * kv_dim * sizeof(float));
            g->r_act = (float*)carve(&cv, (size_t)MOE_ROWS * c->intermediate * sizeof(float));
            g->r_router = (float*)carve(&cv, (size_t)2 * MOE_ROWS * c->n_experts * sizeof(float));
            g->r_logits = (float*)carve(&cv, (size_t)LOGIT_ROWS * c->vocab * sizeof(float));
            g->r_idx = (int32_t*)carve(&cv, (size_t)MOE_ROWS * c->n_experts_used * sizeof(int32_t));
            g->r_w = (float*)carve(&cv, (size_t)MOE_ROWS * c->n_experts_used * sizeof(float));
        }
        /* A position is always written before it is read; zeros make a mistake there repeatable. */
        e = cudaMemset(g->k_cache, 0, 2 * padded(cache_floats * sizeof(float)));
        /* Attention counts finished slices up from zero, and leaves the count at zero. */
        if (e == cudaSuccess) e = cudaMemset(g->done, 0, MULTI_MAX * c->n_kv_heads * sizeof(unsigned int));
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

    /* A mixture of experts runs layer by layer, routed in between, so its
     * forward pass is not captured. */
    if (!g->moe) {
        e = capture(g, enqueue_body, &g->body);
        if (e == cudaSuccess) e = capture(g, enqueue_head, &g->head);
        if (e != cudaSuccess) {
            fail_cuda(err, err_len, "capturing the forward pass as a CUDA graph failed", e);
            vitna_cuda_free(g);
            return NULL;
        }
    }

    /* Run the graphs once, at position 0 of the first sequence, so a build
     * with no code for this GPU fails on load, not at the first token. What
     * it writes to that position is written again before it is read. A
     * mixture of experts runs its embedding and head instead, as there are
     * no experts on the device yet. */
    set_row_kernel<<<1, 1, 0, g->stream>>>(g->rows, 0, 0, 0);
    e = cudaGetLastError();
    if (e == cudaSuccess && g->moe) {
        LAUNCH(g->embed.dtype, embed_kernel, blocks_for(c->hidden, THREADS), 0, g->stream, g->embed.w, g->rows, g->hidden, g->x);
        enqueue_head(g);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess && !g->moe) e = cudaGraphLaunch(g->body, g->stream);
    if (e == cudaSuccess && !g->moe) e = cudaGraphLaunch(g->head, g->stream);
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
    if (g->moe && !create_experts(g, m, expert_cache_bytes, err, err_len)) {
        vitna_cuda_free(g);
        return NULL;
    }
    return g;
}

bool vitna_cuda_step(struct vitna_cuda_model* g, size_t seq, int32_t token, size_t pos, float* logits, char* err, size_t err_len) {
    if (g->moe) return fail(err, err_len, "a mixture of experts runs a layer at a time, routed in between (vitna_cuda_moe_route)");
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);

    set_row_kernel<<<1, 1, 0, g->stream>>>(g->rows, token, (int)pos, (int)seq);
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
    gemm(g, g->p_xn, g->hidden, &g->lm_head, g->p_logits, g->vocab, n, g->vocab, g->hidden, 0);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) {
        e = cudaMemcpyAsync(out, g->p_logits, (size_t)n * g->vocab * sizeof(float), cudaMemcpyDeviceToHost, g->stream);
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    return e;
}

size_t vitna_cuda_prompt_piece_min(const struct vitna_cuda_model* g) {
    return g->prompt ? FEW_TOKENS + 1 : 1;
}

size_t vitna_cuda_prompt_min(const struct vitna_cuda_model* g) {
    return g->prompt ? PROMPT_MIN : SIZE_MAX;
}

bool vitna_cuda_steps(struct vitna_cuda_model* g, size_t seq, const int32_t* tokens, size_t count, size_t pos, float* logits, size_t rows,
                      char* err, size_t err_len) {
    if (!g->prompt) return fail(err, err_len, "this model's prompts cannot run many tokens at once on the GPU");
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    if (!logits) rows = 0;
    const size_t first = count - rows; /* the first position whose logits are wanted */

    /* From pageable memory, so tokens may change as soon as this returns. */
    e = cudaMemcpyAsync(g->p_tokens, tokens, count * sizeof(int32_t), cudaMemcpyHostToDevice, g->stream);
    int T = 0;
    for (size_t c0 = 0; e == cudaSuccess && c0 < count; c0 += (size_t)T) {
        /* Chunks of g->prefill tokens, but no chunk of FEW_TOKENS or fewer
         * after one: those go through gemm_few_kernel, whose sums round
         * differently, so a prompt split anywhere into pieces of more than
         * FEW_TOKENS runs exactly as it does in one call
         * (vitna_cuda_prompt_piece_min). The chunk before such a remainder
         * leaves FEW_TOKENS + 1 for the last. */
        const size_t rest = count - c0;
        size_t t = rest < g->prefill ? rest : g->prefill;
        if (rest > t && rest - t <= FEW_TOKENS) t = rest - (FEW_TOKENS + 1);
        T = (int)t;
        enqueue_chunk(g, (int)seq, c0, (int)(pos + c0), T);
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

size_t vitna_cuda_exact_max(const struct vitna_cuda_model* g) {
    if (!g) return 0;
    /* A mixture's rows go through vitna_cuda_moe_rows_*, which model.c drives. */
    return g->moe ? MOE_ROWS : (size_t)g->multi_max;
}

bool vitna_cuda_rows(struct vitna_cuda_model* g, const vitna_cuda_row_t* rows, size_t count, float* const* logits, char* err,
                     size_t err_len) {
    if (count < 1 || count > (size_t)g->multi_max) return fail(err, err_len, "%zu rows at once is more than this model takes (%d)", count, g->multi_max);
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    if (!g->multi[count]) {
        g->capture_T = (int)count;
        e = capture(g, enqueue_multi, &g->multi[count]);
        if (e != cudaSuccess) return fail_cuda(err, err_len, "capturing several steps as a CUDA graph failed", e);
    }
    const size_t bytes = count * g->vocab * sizeof(float);
    /* From pageable memory, so rows may change as soon as this returns. */
    e = cudaMemcpyAsync(g->rows, rows, count * sizeof(row_t), cudaMemcpyHostToDevice, g->stream);
    if (e == cudaSuccess) e = cudaGraphLaunch(g->multi[count], g->stream);
    if (e == cudaSuccess) e = cudaMemcpyAsync(g->host_multi, g->p_logits, bytes, cudaMemcpyDeviceToHost, g->stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
    for (size_t i = 0; i < count; i++) {
        if (logits[i]) memcpy(logits[i], g->host_multi + i * g->vocab, g->vocab * sizeof(float));
    }
    return true;
}

/* --- A mixture of experts, layer by layer --- */

/* The copies that have finished, oldest first, their sources released to
 * the expert stream. */
static void settle(expert_cache_t* c) {
    while (c->p_len > 0) {
        const cudaError_t e = cudaEventQuery(c->pending[c->p_head].done);
        if (e != cudaSuccess) {
            if (e == cudaErrorNotReady) cudaGetLastError(); /* not an error: the copy is still running */
            return;
        }
        vitna_expert_stream_release(c->stream, &c->pending[c->p_head].place, 1);
        c->p_head = (c->p_head + 1) % PENDING_MAX;
        c->p_len--;
    }
}

/* Wait for the oldest copies until at most n are left, releasing their sources. */
static cudaError_t drain(expert_cache_t* c, int n) {
    while (c->p_len > n) {
        const cudaError_t e = cudaEventSynchronize(c->pending[c->p_head].done);
        if (e != cudaSuccess) return e;
        vitna_expert_stream_release(c->stream, &c->pending[c->p_head].place, 1);
        c->p_head = (c->p_head + 1) % PENDING_MAX;
        c->p_len--;
    }
    return cudaSuccess;
}

/* A slot to give an expert: an empty one, or else the least used of those
 * not held, the least recently used between equals, as the expert stream
 * chooses (expert_stream.c). -1 if every slot is held. */
static int free_slot(const expert_cache_t* c) {
    int best = -1;
    for (int i = 0; i < c->n_slots; i++) {
        const eslot_t* x = &c->slots[i];
        if (x->held) continue;
        if (x->place < 0) return i;
        if (best < 0) {
            best = i;
            continue;
        }
        const eslot_t* b = &c->slots[best];
        const uint32_t cx = c->uses[x->place], cb = c->uses[b->place];
        if (cx < cb || (cx == cb && x->last_use < b->last_use)) best = i;
    }
    return best;
}

/* Give slot s to the expert at place and queue the copy of its gate, up and
 * down matrices from data into it, on the copy stream, then the event that
 * marks the copy done. The slot is held. A source the expert stream holds
 * stays held until the copy has finished (settle, drain). */
static cudaError_t fill(expert_cache_t* c, int s, uint32_t place, const vitna_expert_data_t* data, bool guessed) {
    eslot_t* x = &c->slots[s];
    if (x->place >= 0) c->where[x->place] = -1;
    x->place = (int64_t)place;
    c->where[place] = s;
    x->held = true;
    x->guessed = guessed;
    x->last_use = ++c->clock;
    x->filling = true;
    unsigned char* dst = c->mem + (size_t)s * c->slot_bytes;
    const size_t layer = place / (c->n_places / (size_t)c->n_layers);
    const size_t* bytes = c->part_bytes + 3 * layer;
    const size_t* off = c->part_off + 3 * layer;
    cudaError_t e = cudaSuccess;
    for (int p = 0; p < 3 && e == cudaSuccess; p++) {
        e = cudaMemcpyAsync(dst + off[p], data->part[p], bytes[p], cudaMemcpyHostToDevice, c->copy);
        c->bytes_copied += bytes[p];
    }
    if (e == cudaSuccess) e = cudaEventRecord(x->filled, c->copy);
    c->copies++;
    if (c->stream) {
        if (e == cudaSuccess && c->p_len == PENDING_MAX) e = drain(c, PENDING_MAX - 1);
        pending_t* p = &c->pending[(c->p_head + c->p_len) % PENDING_MAX];
        if (e == cudaSuccess) e = cudaEventRecord(p->done, c->copy);
        if (e == cudaSuccess) {
            p->place = place;
            c->p_len++;
        } else {
            vitna_expert_stream_release(c->stream, &place, 1);
        }
    }
    if (e != cudaSuccess) {
        /* The slot cannot be trusted: let it be empty. */
        c->where[place] = -1;
        x->place = -1;
    }
    return e;
}

bool vitna_cuda_moe_route(struct vitna_cuda_model* g, size_t seq, int32_t token, size_t pos, size_t layer, float* logits, float* next,
                          char* err, size_t err_len) {
    if (!g->moe || layer >= (size_t)g->n_layers || (next && layer + 1 >= (size_t)g->n_layers)) {
        return fail(err, err_len, "layer %zu of this model cannot be routed", layer);
    }
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    const cudaStream_t s = g->stream;
    const int H = g->hidden, E = g->n_experts, l = (int)layer;
    if (l == 0) {
        set_row_kernel<<<1, 1, 0, s>>>(g->rows, token, (int)pos, (int)seq);
        LAUNCH(g->embed.dtype, embed_kernel, blocks_for((size_t)H, THREADS), 0, s, g->embed.w, g->rows, H, g->x);
    }
    enqueue_attention(g, l);
    const dlayer_t* L = &g->layers[l];
    const dlayer_t* N = next ? &g->layers[l + 1] : L;
    LAUNCH(L->router.dtype, router_kernel, dim3(blocks_for((size_t)E, WARPS), next ? 2 : 1), (size_t)H * sizeof(float), s, g->x, L->mlp_norm,
           L->router.w, N->mlp_norm, N->router.w, g->eps, g->router_out, H, E);
    /* The CPU, running the experts the device lacks, takes the residual as
     * the experts' kernels normalize it, which norm_out_kernel computes the
     * same way, block for block. */
    if (g->cpu) norm_out_kernel<<<1, THREADS, (size_t)H * sizeof(float), s>>>(g->x, L->mlp_norm, g->eps, g->moe_xs, H);
    e = cudaGetLastError();
    const size_t n = (next ? 2 : 1) * (size_t)E;
    if (e == cudaSuccess) e = cudaMemcpyAsync(g->host_router, g->router_out, n * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess && g->cpu) e = cudaMemcpyAsync(g->host_xs, g->moe_xs, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
    memcpy(logits, g->host_router, (size_t)E * sizeof(float));
    if (next) memcpy(next, g->host_router + E, (size_t)E * sizeof(float));
    return true;
}

/* A cost a byte, averaged as it is measured: the latest counts a tenth. */
static void average(double* v, double x) {
    *v = *v > 0 ? 0.9 * *v + 0.1 * x : x;
}

/* Gate A8: layer's experts, those the device lacks shared between the CPU
 * and the bus (vitna_cuda_cpu_experts). The device runs those it holds, as
 * vitna_cuda_moe_experts does, and of those it lacks it has some copied in
 * and runs them too, while the CPU runs the rest from memory in the GPU's
 * arithmetic (warp.h), so each of its outputs is the bits the device's would
 * be. The device keeps each output apart (experts_parts_kernel), the CPU's
 * are copied up beside them, and experts_mix_kernel weights and adds them
 * all in the order given, as experts_down_kernel does: the residual is the
 * one vitna_cuda_moe_experts leaves, bit for bit, whichever experts ran where.
 *
 * The split follows what each costs, measured as they run. Of the experts a
 * layer lacks, the CPU takes as many as let the layer finish soonest, given
 * what an expert has cost each lately, the least used first, and none when
 * the copies alone finish sooner; the rest are copied in, where they stay
 * for the tokens after. A side left out for SPLIT_PROBE layers is given one
 * expert, so that neither cost goes stale. The CPU's outputs leave host_y
 * on the device's stream, which the next layer's vitna_cuda_moe_route waits
 * for before this runs again. */
static bool moe_experts_split(struct vitna_cuda_model* g, size_t layer, const int32_t* ids, const float* weights, size_t k,
                              const int32_t* guess, char* err, size_t err_len) {
    expert_cache_t* c = &g->ec;
    const size_t E = (size_t)g->n_experts, H = (size_t)g->hidden, I = (size_t)g->intermediate;
    const dlayer_t* L = &g->layers[layer];
    const size_t* part = c->part_bytes + 3 * layer;
    const size_t expert_bytes = part[0] + part[1] + part[2];

    /* What the copies made for the layer before cost, now that they are done. */
    if (c->copy_timed) {
        float ms = 0.0f;
        const cudaError_t q = cudaEventQuery(c->copy_t1);
        if (q == cudaSuccess) {
            if (cudaEventElapsedTime(&ms, c->copy_t0, c->copy_t1) == cudaSuccess && ms > 0.0f) average(&c->copy_ms_b, ms / (double)c->copy_timed);
            c->copy_timed = 0;
        }
        cudaGetLastError(); /* not ready is not an error */
    }

    /* The experts the device holds, finished (ready) or still being copied
     * (waited for), and those it lacks. Each held is held until this
     * returns, so neither a copy for another nor the guess takes its slot. */
    int ready_at[EXPERTS_MAX], ready_slot[EXPERTS_MAX], n_ready = 0;
    int wait_at[EXPERTS_MAX], wait_slot[EXPERTS_MAX], n_wait = 0;
    int miss_at[EXPERTS_MAX], n_miss = 0;
    uint32_t miss_place[EXPERTS_MAX];
    int held[3 * EXPERTS_MAX], n_held = 0;
    for (size_t i = 0; i < k; i++) {
        const uint32_t place = (uint32_t)(layer * E + (size_t)ids[i]);
        c->uses[place]++;
        if (++c->acquired % 65536 == 0) {
            for (size_t p = 0; p < c->n_places; p++) c->uses[p] /= 2;
        }
        const int s = c->where[place];
        if (s < 0) {
            c->misses++;
            miss_place[n_miss] = place;
            miss_at[n_miss++] = (int)i;
            continue;
        }
        eslot_t* x = &c->slots[s];
        x->held = true;
        held[n_held++] = s;
        x->last_use = ++c->clock;
        if (x->filling) {
            const cudaError_t q = cudaEventQuery(x->filled);
            if (q == cudaSuccess) x->filling = false;
            else if (q == cudaErrorNotReady) cudaGetLastError();
        }
        if (x->filling) {
            c->in_flight++;
            wait_slot[n_wait] = s;
            wait_at[n_wait++] = (int)i;
        } else {
            c->hits++;
            ready_slot[n_ready] = s;
            ready_at[n_ready++] = (int)i;
        }
        if (x->guessed) {
            c->prefetch_used++;
            x->guessed = false;
        }
    }

    /* The CPU's share of those it lacks, the least used of them: as many
     * as let the layer finish soonest, the CPU running its share while the
     * rest are copied, by what an expert has cost each lately. Until both
     * costs are known, both sides take some. Every SPLIT_PROBE layers with
     * misses that one side has not been given any, it is given one, so that
     * its cost is never long out of date: the box's load changes it. */
    bool on_cpu[EXPERTS_MAX] = { false };
    int n_cpu = 0;
    if (n_miss) {
        const double cpu_e = c->cpu_ms_b * (double)expert_bytes, copy_e = c->copy_ms_b * (double)expert_bytes;
        if (cpu_e <= 0) {
            n_cpu = (n_miss + 1) / 2;
        } else if (copy_e <= 0) {
            n_cpu = n_miss / 2;
        } else {
            double best = copy_e * (double)n_miss;
            for (int n = 1; n <= n_miss; n++) {
                const double t = fmax(cpu_e * (double)n, copy_e * (double)(n_miss - n));
                if (t < best) {
                    best = t;
                    n_cpu = n;
                }
            }
        }
        c->cpu_idle = n_cpu > 0 ? 0 : c->cpu_idle + 1;
        c->copy_idle = n_cpu < n_miss ? 0 : c->copy_idle + 1;
        if (c->cpu_idle >= SPLIT_PROBE) {
            n_cpu = 1;
            c->cpu_idle = 0;
        } else if (c->copy_idle >= SPLIT_PROBE) {
            n_cpu--;
            c->copy_idle = 0;
        }
        int order[EXPERTS_MAX];
        for (int j = 0; j < n_miss; j++) {
            int at = j;
            while (at > 0 && c->uses[miss_place[order[at - 1]]] > c->uses[miss_place[j]]) {
                order[at] = order[at - 1];
                at--;
            }
            order[at] = j;
        }
        for (int j = 0; j < n_cpu; j++) on_cpu[order[j]] = true;
    }

    /* The device's finished experts' gate and up, started first, so that
     * they run while the rest are found, copied and run. */
    cudaError_t e = cudaSuccess;
    bool ok = true;
    experts_t ex;
    memset(&ex, 0, sizeof(ex));
    const size_t* off = c->part_off + 3 * layer;
    const auto put = [&](int d, int s, int at) {
        const unsigned char* base = c->mem + (size_t)s * c->slot_bytes;
        ex.gate[d] = base + off[0];
        ex.up[d] = base + off[1];
        ex.down[d] = base + off[2];
        ex.weight[d] = weights[at];
    };
    for (int d = 0; d < n_ready; d++) put(d, ready_slot[d], ready_at[d]);
    ex.k = n_ready;
    if (n_ready) {
        LAUNCH(L->e_gu, experts_in_kernel, grid_for(g, n_ready * (int)I), H * sizeof(float), g->stream, g->x, L->mlp_norm, g->eps, ex,
               g->moe_act, (int)H, (int)I);
        e = cudaGetLastError();
        ok = e == cudaSuccess;
    }

    /* The guess's places, in the next layer, and those of them the device lacks. */
    uint32_t next_places[EXPERTS_MAX], lacking[EXPERTS_MAX];
    size_t n_lacking = 0;
    for (size_t j = 0; guess && j < k; j++) {
        next_places[j] = (uint32_t)((layer + 1) * E + (size_t)guess[j]);
        if (c->where[next_places[j]] < 0) lacking[n_lacking++] = next_places[j];
    }

    /* Where those it lacks are: held in the expert stream, read from the
     * drive where it lacks them too, or in the mapped checkpoint. */
    vitna_expert_data_t data[EXPERTS_MAX];
    if (ok && n_miss && c->stream) {
        e = drain(c, c->p_max);
        ok = e == cudaSuccess;
        if (ok) {
            vitna_expert_stream_hold(c->stream, miss_place, (size_t)n_miss);
            if (n_lacking) vitna_expert_stream_prefetch(c->stream, lacking, n_lacking);
            if (!vitna_expert_stream_wait(c->stream, miss_place, (size_t)n_miss, data)) {
                for (int i = 0; i < n_held; i++) c->slots[held[i]].held = false;
                return fail(err, err_len, "reading an expert from the drive failed");
            }
        }
    } else if (ok && n_miss) {
        for (int j = 0; j < n_miss; j++) {
            for (int p = 0; p < 3; p++) data[j].part[p] = c->parts[(size_t)miss_place[j] * 3 + p];
        }
    } else if (ok && c->stream && n_lacking) {
        vitna_expert_stream_prefetch(c->stream, lacking, n_lacking);
    }
    const bool stream_held = ok && n_miss && c->stream;

    /* The bus's share, copied in, timed; each fill takes over its source's
     * hold in the expert stream, which the copy lets go of when done. One
     * there is no slot for goes to the CPU instead. */
    size_t copied = 0;
    int j = 0;
    for (; ok && j < n_miss; j++) {
        if (on_cpu[j]) continue;
        const int s = free_slot(c);
        if (s < 0) {
            on_cpu[j] = true;
            n_cpu++;
            continue;
        }
        if (copied == 0 && !c->copy_timed) {
            e = cudaEventRecord(c->copy_t0, c->copy);
            ok = e == cudaSuccess;
            if (!ok) break;
        }
        e = fill(c, s, miss_place[j], &data[j], false);
        ok = e == cudaSuccess;
        held[n_held++] = s;
        wait_slot[n_wait] = s;
        wait_at[n_wait++] = miss_at[j];
        copied += expert_bytes;
    }
    /* After a failure, the bus's share not yet given to a copy (a fill that
     * fails lets go of its own) is held for nothing. */
    for (int r = j; !ok && stream_held && r < n_miss; r++) {
        if (!on_cpu[r]) vitna_expert_stream_release(c->stream, &miss_place[r], 1);
    }
    if (ok && copied && !c->copy_timed) {
        e = cudaEventRecord(c->copy_t1, c->copy);
        ok = e == cudaSuccess;
        if (ok) c->copy_timed = copied;
    }

    /* Those being copied, once they are there, then every device expert's
     * output apart; or, with all of them finished and none on the CPU,
     * experts_down_kernel adds them as vitna_cuda_moe_experts does. */
    for (int w = 0; ok && w < n_wait; w++) {
        e = cudaStreamWaitEvent(g->stream, c->slots[wait_slot[w]].filled, 0);
        ok = e == cudaSuccess;
    }
    const int n_dev = n_ready + n_wait;
    if (ok && n_wait) {
        experts_t wx;
        memset(&wx, 0, sizeof(wx));
        for (int w = 0; w < n_wait; w++) put(n_ready + w, wait_slot[w], wait_at[w]);
        for (int w = 0; w < n_wait; w++) {
            wx.gate[w] = ex.gate[n_ready + w];
            wx.up[w] = ex.up[n_ready + w];
        }
        wx.k = n_wait;
        LAUNCH(L->e_gu, experts_in_kernel, grid_for(g, n_wait * (int)I), H * sizeof(float), g->stream, g->x, L->mlp_norm, g->eps, wx,
               g->moe_act + (size_t)n_ready * I, (int)H, (int)I);
        e = cudaGetLastError();
        ok = e == cudaSuccess;
    }
    ex.k = n_dev;
    const bool mixed = n_cpu > 0 || n_wait > 0;
    if (ok && n_dev) {
        if (mixed) {
            LAUNCH(L->e_down, experts_parts_kernel, grid_for(g, n_dev * (int)H), 0, g->stream, g->moe_act, ex, g->moe_y, (int)H, (int)I);
        } else {
            LAUNCH(L->e_down, experts_down_kernel, grid_for(g, (int)H), (size_t)n_dev * I * sizeof(float), g->stream, g->moe_act, ex, g->x,
                   (int)H, (int)I);
        }
        e = cudaGetLastError();
        ok = e == cudaSuccess;
    }

    /* The guess, copied as vitna_cuda_moe_experts copies it, behind this layer's copies. */
    for (size_t j = 0; ok && guess && j < (k + 1) / 2; j++) {
        uint32_t place = next_places[j];
        if (c->where[place] >= 0) continue;
        vitna_expert_data_t d;
        if (c->stream) {
            if (c->p_len >= c->p_max || !vitna_expert_stream_hold_ready(c->stream, place, &d)) continue;
        } else {
            for (int p = 0; p < 3; p++) d.part[p] = c->parts[(size_t)place * 3 + p];
        }
        const int s = free_slot(c);
        if (s < 0) {
            if (c->stream) vitna_expert_stream_release(c->stream, &place, 1);
            break;
        }
        e = fill(c, s, place, &d, true);
        ok = e == cudaSuccess;
        held[n_held++] = s;
        c->prefetched++;
    }

    /* The CPU's share, timed, its holds let go of, and its outputs copied up
     * and mixed with the device's in the order given. */
    uint32_t cpu_place[EXPERTS_MAX];
    int cpu_at[EXPERTS_MAX], n_run = 0;
    vitna_warp_expert_t jobs[EXPERTS_MAX];
    for (int j = 0; j < n_miss; j++) {
        if (!on_cpu[j]) continue;
        jobs[n_run].gate = data[j].part[0];
        jobs[n_run].up = data[j].part[1];
        jobs[n_run].down = data[j].part[2];
        jobs[n_run].y = g->host_y + (size_t)n_run * H;
        cpu_place[n_run] = miss_place[j];
        cpu_at[n_run++] = miss_at[j];
    }
    if (ok && n_run) {
        const auto t0 = std::chrono::steady_clock::now();
        vitna_warp_experts(g->cpu, jobs, (size_t)n_run, L->e_gu, L->e_down, H, I, g->host_xs, g->cpu_act, g->cpu_scratch);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        c->cpu_ms += ms;
        c->on_cpu += (uint64_t)n_run;
        average(&c->cpu_ms_b, ms / ((double)n_run * (double)expert_bytes));
    }
    if (stream_held && n_run) vitna_expert_stream_release(c->stream, cpu_place, (size_t)n_run);
    if (ok && mixed) {
        mix_t m;
        memset(&m, 0, sizeof(m));
        m.k = (int)k;
        for (int d = 0; d < n_ready; d++) m.at[ready_at[d]] = d;
        for (int w = 0; w < n_wait; w++) m.at[wait_at[w]] = n_ready + w;
        for (int j = 0; j < n_run; j++) m.at[cpu_at[j]] = n_dev + j;
        for (size_t i = 0; i < k; i++) m.weight[i] = weights[i];
        if (n_run) {
            e = cudaMemcpyAsync(g->moe_y + (size_t)n_dev * H, g->host_y, (size_t)n_run * H * sizeof(float), cudaMemcpyHostToDevice, g->stream);
        }
        if (e == cudaSuccess) {
            experts_mix_kernel<<<blocks_for(H, THREADS), THREADS, 0, g->stream>>>(g->moe_y, m, g->x, (int)H);
            e = cudaGetLastError();
        }
        ok = e == cudaSuccess;
    }
    for (int i = 0; i < n_held; i++) c->slots[held[i]].held = false;
    if (!ok) return fail_cuda(err, err_len, "the experts could not be run", e);
    return true;
}

bool vitna_cuda_moe_experts(struct vitna_cuda_model* g, size_t layer, const int32_t* ids, const float* weights, size_t k,
                            const int32_t* guess, char* err, size_t err_len) {
    expert_cache_t* c = &g->ec;
    const size_t E = (size_t)g->n_experts;
    /* moe_act holds n_used experts' activations, and the cache's slots are
     * at least twice n_used, so that this layer's and its guess fit. */
    if (!g->moe || layer >= (size_t)g->n_layers || k < 1 || k > (size_t)g->n_used || (guess && layer + 1 >= (size_t)g->n_layers)) {
        return fail(err, err_len, "%zu experts of layer %zu cannot be run", k, layer);
    }
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    if (c->stream) settle(c);
    if (g->cpu) return moe_experts_split(g, layer, ids, weights, k, guess, err, err_len);

    /* The experts the device holds already, and those it lacks. Each is held
     * until this returns, so neither a copy for another nor the guess can
     * take its slot. Before this call the device finished every earlier
     * layer (vitna_cuda_moe_route waited for this layer's router), so a
     * slot nobody holds is read by nothing queued. */
    int slot_of[EXPERTS_MAX], held[2 * EXPERTS_MAX], n_held = 0;
    uint32_t missing[EXPERTS_MAX];
    int miss_at[EXPERTS_MAX], n_missing = 0;
    for (size_t i = 0; i < k; i++) {
        const uint32_t place = (uint32_t)(layer * E + (size_t)ids[i]);
        c->uses[place]++;
        if (++c->acquired % 65536 == 0) {
            for (size_t p = 0; p < c->n_places; p++) c->uses[p] /= 2;
        }
        const int s = c->where[place];
        slot_of[i] = s;
        if (s < 0) {
            missing[n_missing] = place;
            miss_at[n_missing++] = (int)i;
            c->misses++;
            continue;
        }
        eslot_t* x = &c->slots[s];
        x->held = true;
        held[n_held++] = s;
        x->last_use = ++c->clock;
        if (x->filling) {
            const cudaError_t q = cudaEventQuery(x->filled);
            if (q == cudaSuccess) x->filling = false;
            else if (q == cudaErrorNotReady) cudaGetLastError();
        }
        if (x->filling) c->in_flight++;
        else c->hits++;
        if (x->guessed) {
            c->prefetch_used++;
            x->guessed = false;
        }
    }

    /* The guess's places, in the next layer, and those of them the device
     * lacks: only those will be wanted from memory, so only those are
     * worth reading from the drive ahead. */
    uint32_t next_places[EXPERTS_MAX], lacking[EXPERTS_MAX];
    size_t n_lacking = 0;
    for (size_t j = 0; guess && j < k; j++) {
        next_places[j] = (uint32_t)((layer + 1) * E + (size_t)guess[j]);
        if (c->where[next_places[j]] < 0) lacking[n_lacking++] = next_places[j];
    }

    /* Those it lacks, copied in. Read from the drive where the expert
     * stream lacks them too: hold, start reading the guess there as the
     * CPU path does, then wait. */
    bool ok = true;
    if (n_missing) {
        vitna_expert_data_t data[EXPERTS_MAX];
        if (c->stream) {
            e = drain(c, c->p_max);
            ok = e == cudaSuccess;
            if (ok) {
                vitna_expert_stream_hold(c->stream, missing, (size_t)n_missing);
                if (n_lacking) vitna_expert_stream_prefetch(c->stream, lacking, n_lacking);
                if (!vitna_expert_stream_wait(c->stream, missing, (size_t)n_missing, data)) {
                    for (int i = 0; i < n_held; i++) c->slots[held[i]].held = false;
                    return fail(err, err_len, "reading an expert from the drive failed");
                }
            }
        } else {
            for (int j = 0; j < n_missing; j++) {
                for (int p = 0; p < 3; p++) data[j].part[p] = c->parts[(size_t)missing[j] * 3 + p];
            }
        }
        for (int j = 0; ok && j < n_missing; j++) {
            /* At most k slots are held, and there are at least twice k. */
            const int s = free_slot(c);
            e = fill(c, s, missing[j], &data[j], false);
            ok = e == cudaSuccess;
            held[n_held++] = s;
            slot_of[miss_at[j]] = s;
            if (!ok && c->stream) vitna_expert_stream_release(c->stream, missing + j + 1, (size_t)(n_missing - j - 1));
        }
    } else if (c->stream && n_lacking) {
        vitna_expert_stream_prefetch(c->stream, lacking, n_lacking);
    }

    /* The layer's kernels, after the copies into their slots. */
    experts_t ex;
    memset(&ex, 0, sizeof(ex));
    ex.k = (int)k;
    for (size_t i = 0; ok && i < k; i++) {
        const eslot_t* x = &c->slots[slot_of[i]];
        if (x->filling) {
            e = cudaStreamWaitEvent(g->stream, x->filled, 0);
            ok = e == cudaSuccess;
        }
        const unsigned char* base = c->mem + (size_t)slot_of[i] * c->slot_bytes;
        const size_t* off = c->part_off + 3 * layer;
        ex.gate[i] = base + off[0];
        ex.up[i] = base + off[1];
        ex.down[i] = base + off[2];
        ex.weight[i] = weights[i];
    }
    if (ok) {
        const dlayer_t* L = &g->layers[layer];
        LAUNCH(L->e_gu, experts_in_kernel, grid_for(g, (int)k * g->intermediate), (size_t)g->hidden * sizeof(float), g->stream, g->x,
               L->mlp_norm, g->eps, ex, g->moe_act, g->hidden, g->intermediate);
        LAUNCH(L->e_down, experts_down_kernel, grid_for(g, g->hidden), k * (size_t)g->intermediate * sizeof(float), g->stream, g->moe_act, ex,
               g->x, g->hidden, g->intermediate);
        e = cudaGetLastError();
        ok = e == cudaSuccess;
    }

    /* The guess: experts of the next layer the device lacks, copied behind
     * this layer's, from memory that has them now; one the expert stream is
     * still reading is left to be copied when it is wanted. Only the half
     * the next layer's router ranks highest are copied. A token's time goes
     * to copies, so a wrong one costs what a right one saves, and the lower
     * half is wrong too often: on OLMoE with an RTX 3070 (PCIe 4.0 x16,
     * 2026-10-01), of the copies made for each rank of the guess, the layer
     * then wanted 98% of the first two's, 90% of the third and fourth's,
     * 74% of the fifth and sixth's and 48% of the last two's. Copying the
     * first four decoded at 16.1 ms a token with the cache the device
     * allowed and 38.5 ms with 2 GiB, against 17.1 and 45.6 copying all
     * eight, 16.0 and 39.9 copying six, and 18.0 and 43.5 copying none. */
    for (size_t j = 0; ok && guess && j < (k + 1) / 2; j++) {
        uint32_t place = next_places[j];
        if (c->where[place] >= 0) continue;
        vitna_expert_data_t d;
        if (c->stream) {
            if (c->p_len >= c->p_max || !vitna_expert_stream_hold_ready(c->stream, place, &d)) continue;
        } else {
            for (int p = 0; p < 3; p++) d.part[p] = c->parts[(size_t)place * 3 + p];
        }
        const int s = free_slot(c);
        if (s < 0) {
            if (c->stream) vitna_expert_stream_release(c->stream, &place, 1);
            break;
        }
        e = fill(c, s, place, &d, true);
        ok = e == cudaSuccess;
        held[n_held++] = s;
        c->prefetched++;
    }
    for (int i = 0; i < n_held; i++) c->slots[held[i]].held = false;
    if (!ok) return fail_cuda(err, err_len, "the experts could not be run on the device", e);
    return true;
}

bool vitna_cuda_moe_head(struct vitna_cuda_model* g, float* logits, char* err, size_t err_len) {
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    enqueue_head(g);
    e = cudaGetLastError();
    const size_t bytes = (size_t)g->vocab * sizeof(float);
    if (e == cudaSuccess) e = cudaMemcpyAsync(g->host_logits, g->logits, bytes, cudaMemcpyDeviceToHost, g->stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
    memcpy(logits, g->host_logits, bytes);
    return true;
}

const char* vitna_cuda_moe_report(const struct vitna_cuda_model* g, char* buf, size_t len) {
    if (len == 0) return buf;
    buf[0] = '\0';
    if (!g || !g->moe) return buf;
    const expert_cache_t* c = &g->ec;
    snprintf(buf, len,
             "experts on the GPU: %d slots, %.0f MiB, copied from %s, %s. %llu acquired: %llu already on the GPU, %llu still being copied "
             "for a guess, %llu copied when asked for. %llu copied for a guess, %llu of those used. %.1f MiB copied in %llu copies",
             c->n_slots, (double)c->n_slots * (double)c->slot_bytes / 1048576.0, c->stream ? "the expert cache in memory" : "the mapped checkpoint",
             c->pinned ? "page-locked" : "pageable", (unsigned long long)c->acquired, (unsigned long long)c->hits,
             (unsigned long long)c->in_flight, (unsigned long long)c->misses, (unsigned long long)c->prefetched,
             (unsigned long long)c->prefetch_used, (double)c->bytes_copied / 1048576.0, (unsigned long long)c->copies);
    if (g->cpu) {
        /* Of the experts counted above as copied when asked for, a step's
         * CPU share ran there instead; rows copy in all they lack, as
         * without the CPU. */
        const size_t at = strlen(buf);
        snprintf(buf + at, len - at,
                 ". In steps, %llu of those the GPU lacked ran on the CPU, on %zu threads, in %.0f ms; lately a GiB of experts cost the "
                 "CPU %.0f ms and a copy %.0f ms",
                 (unsigned long long)c->on_cpu, vitna_pool_threads(g->cpu), c->cpu_ms, c->cpu_ms_b * 1073741824.0, c->copy_ms_b * 1073741824.0);
    }
    return buf;
}

bool vitna_cuda_cpu_experts(struct vitna_cuda_model* g, size_t threads, char* err, size_t err_len) {
    if (!g->moe) return fail(err, err_len, "this model has no experts to run on the CPU");
    if (g->cpu) return fail(err, err_len, "the CPU runs experts already");
    if (threads == 0) return fail(err, err_len, "the CPU needs a thread or more to run experts on");
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    const size_t H = (size_t)g->hidden, I = (size_t)g->intermediate, K = (size_t)g->n_used;
    e = cudaMalloc((void**)&g->moe_xs, H * sizeof(float));
    if (e == cudaSuccess) e = cudaMalloc((void**)&g->moe_y, K * H * sizeof(float));
    if (e == cudaSuccess) e = cudaMallocHost((void**)&g->host_xs, H * sizeof(float));
    if (e == cudaSuccess) e = cudaMallocHost((void**)&g->host_y, K * H * sizeof(float));
    if (e == cudaSuccess && !g->ec.copy_t0) e = cudaEventCreate(&g->ec.copy_t0);
    if (e == cudaSuccess && !g->ec.copy_t1) e = cudaEventCreate(&g->ec.copy_t1);
    if (e == cudaSuccess) {
        g->cpu_act = (float*)malloc(K * 2 * I * sizeof(float));
        g->cpu_scratch = (float*)malloc(threads * vitna_warp_scratch_floats(H, I) * sizeof(float));
        if (g->cpu_act && g->cpu_scratch) g->cpu = vitna_pool_create(threads);
    }
    if (g->cpu) return true;
    if (g->moe_xs) cudaFree(g->moe_xs);
    if (g->moe_y) cudaFree(g->moe_y);
    if (g->host_xs) cudaFreeHost(g->host_xs);
    if (g->host_y) cudaFreeHost(g->host_y);
    free(g->cpu_act);
    free(g->cpu_scratch);
    g->moe_xs = g->moe_y = g->host_xs = g->host_y = g->cpu_act = g->cpu_scratch = NULL;
    if (e != cudaSuccess) {
        cudaGetLastError();
        return fail_cuda(err, err_len, "cannot set the CPU up to run experts", e);
    }
    return fail(err, err_len, "cannot start %zu threads to run experts on", threads);
}

/* --- A mixture's rows, layer by layer --- */

size_t vitna_cuda_moe_rows_max(const struct vitna_cuda_model* g) {
    return g && g->moe ? MOE_ROWS : 0;
}

/* Layer l's attention for the pass's n rows, as enqueue_attention runs one,
 * with the multi-row kernels, each row's arithmetic its step's. */
static void enqueue_rows_attention(const struct vitna_cuda_model* g, int l, int n) {
    const cudaStream_t s = g->stream;
    const int H = g->hidden, hd = g->head_dim, half = hd / 2;
    const int q_dim = g->n_heads * hd, kv_dim = g->n_kv_heads * hd;
    const int group = g->n_heads / g->n_kv_heads;
    const int G = g->rows_g;
    const size_t seq_stride = (size_t)g->ctx * kv_dim;
    const dlayer_t* L = &g->layers[l];
    float* kc = g->k_cache + (size_t)l * g->seqs * seq_stride;
    float* vc = g->v_cache + (size_t)l * g->seqs * seq_stride;

    LAUNCH_QKV(L->q.dtype, L->v.dtype, attn_in_multi_kernel,
               dim3(grid_for(g, (g->n_heads + g->n_kv_heads) * half + kv_dim / 2), blocks_for((size_t)n, (size_t)G)), (size_t)G * H * sizeof(float),
               s, g->r_x, L->attn_norm, g->eps, L->q.w, L->k.w, L->v.w, g->r_q, kc, vc, g->cos_t, g->sin_t, g->r_rows, seq_stride, H,
               g->n_heads, g->n_kv_heads, hd, G, n, L->q_norm ? g->r_kraw : (float*)NULL);
    if (L->q_norm) enqueue_qk_norm(g, s, L, g->r_q, g->r_kraw, kc, g->r_rows, seq_stride, n);
    /* Attention keeps its slices for MULTI_MAX rows, so the rows go that many at a time. */
    for (int r0 = 0; r0 < n; r0 += MULTI_MAX) {
        const int t = n - r0 < MULTI_MAX ? n - r0 : MULTI_MAX;
        attention_kernel<<<dim3(t, g->n_kv_heads, ATTENTION_SPLITS), THREADS, attention_shared(g), s>>>(
            g->r_q + (size_t)r0 * q_dim, kc, vc, g->r_att + (size_t)r0 * q_dim, g->part_m, g->part_l, g->part_o, g->done, g->r_rows + r0,
            seq_stride, hd, kv_dim, group, g->scale, 1);
    }
    LAUNCH(L->o.dtype, matvec_add_multi_kernel, dim3(grid_for(g, H), blocks_for((size_t)n, MULTI_MAX)), 0, s, L->o.w, g->r_att, g->r_x, H,
           q_dim, MULTI_MAX, n);
}

bool vitna_cuda_moe_rows_route(struct vitna_cuda_model* g, const vitna_cuda_row_t* rows, size_t n, size_t layer, float* logits, float* next,
                               char* err, size_t err_len) {
    if (!g->moe || n < 1 || n > MOE_ROWS || layer >= (size_t)g->n_layers || (next && layer + 1 >= (size_t)g->n_layers)) {
        return fail(err, err_len, "%zu rows at layer %zu of this model cannot be routed", n, layer);
    }
    if (layer > 0 && n != (size_t)g->r_n) return fail(err, err_len, "a pass's rows changed between its layers");
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    const cudaStream_t s = g->stream;
    const int H = g->hidden, E = g->n_experts, G = g->rows_g, l = (int)layer, nn = (int)n;
    if (l == 0) {
        g->r_n = nn;
        /* From pageable memory, so rows may change as soon as this returns. */
        e = cudaMemcpyAsync(g->r_rows, rows, n * sizeof(row_t), cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) return fail_cuda(err, err_len, "the rows could not be given to the device", e);
        LAUNCH(g->embed.dtype, embed_multi_kernel, dim3(blocks_for((size_t)H, THREADS), nn), 0, s, g->embed.w, g->r_rows, H, g->r_x);
    }
    enqueue_rows_attention(g, l, nn);
    const dlayer_t* L = &g->layers[l];
    const dlayer_t* N = next ? &g->layers[l + 1] : L;
    const size_t norm_shared = (size_t)G * H * sizeof(float);
    norm_rows_kernel<<<blocks_for(n, (size_t)G), THREADS, norm_shared, s>>>(g->r_x, L->mlp_norm, g->eps, g->r_xn, H, G, nn);
    if (next) norm_rows_kernel<<<blocks_for(n, (size_t)G), THREADS, norm_shared, s>>>(g->r_x, N->mlp_norm, g->eps, g->r_xn2, H, G, nn);
    LAUNCH(L->router.dtype, router_rows_kernel, dim3(blocks_for((size_t)E, WARPS), blocks_for(n, MULTI_MAX), next ? 2 : 1), 0, s, g->r_xn,
           g->r_xn2, L->router.w, N->router.w, g->r_router, H, E, nn, MOE_ROWS);
    e = cudaGetLastError();
    const size_t bytes = n * (size_t)E * sizeof(float);
    const size_t half = (size_t)MOE_ROWS * E;
    if (e == cudaSuccess) e = cudaMemcpyAsync(g->h_rrouter, g->r_router, bytes, cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess && next) e = cudaMemcpyAsync(g->h_rrouter + half, g->r_router + half, bytes, cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
    memcpy(logits, g->h_rrouter, bytes);
    if (next) memcpy(next, g->h_rrouter + half, bytes);
    return true;
}

bool vitna_cuda_moe_rows_experts(struct vitna_cuda_model* g, size_t layer, size_t n, const int32_t* ids, const float* weights, size_t k,
                                 const int32_t* guess, char* err, size_t err_len) {
    expert_cache_t* c = &g->ec;
    const size_t E = (size_t)g->n_experts;
    if (!g->moe || n < 1 || n != (size_t)g->r_n || layer >= (size_t)g->n_layers || k < 1 || k > (size_t)g->n_used ||
        (guess && layer + 1 >= (size_t)g->n_layers)) {
        return fail(err, err_len, "%zu rows' experts at layer %zu cannot be run", n, layer);
    }
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    const cudaStream_t s = g->stream;
    const int H = g->hidden, I = g->intermediate;
    if (c->stream) settle(c);

    /* Each expert's rows, in order of row, expert after expert in order of
     * expert, with each row's weight for it. */
    memset(g->e_count, 0, E * sizeof(int32_t));
    for (size_t i = 0; i < n * k; i++) g->e_count[ids[i]]++;
    int32_t at = 0;
    for (size_t x = 0; x < E; x++) {
        g->e_at[x] = at;
        g->e_fill[x] = at;
        at += g->e_count[x];
    }
    for (size_t t = 0; t < n; t++) {
        for (size_t j = 0; j < k; j++) {
            const int32_t x = ids[t * k + j];
            g->h_idx[g->e_fill[x]] = (int32_t)t;
            g->h_w[g->e_fill[x]++] = weights[t * k + j];
        }
    }
    e = cudaMemcpyAsync(g->r_idx, g->h_idx, n * k * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    if (e == cudaSuccess) e = cudaMemcpyAsync(g->r_w, g->h_w, n * k * sizeof(float), cudaMemcpyHostToDevice, s);
    if (e == cudaSuccess) e = cudaMemsetAsync(g->r_proj, 0, n * (size_t)H * sizeof(float), s);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "the rows' experts could not be given to the device", e);

    /* The experts wanted, in order of expert, each counted once for each row
     * that wants it, as steps would count it. */
    size_t n_want = 0;
    for (size_t x = 0; x < E; x++) {
        if (!g->e_count[x]) continue;
        g->e_list[n_want++] = (int32_t)x;
        c->uses[layer * E + x] += (uint32_t)g->e_count[x];
        const uint64_t before = c->acquired;
        c->acquired += (uint64_t)g->e_count[x];
        if (before / 65536 != c->acquired / 65536) {
            for (size_t p = 0; p < c->n_places; p++) c->uses[p] /= 2;
        }
    }

    /* The experts in waves the cache can hold, in order of expert. A wave
     * holds its experts' slots, those the device lacks copied in n_used at a
     * time, then runs them; before the next wave can give a slot up, the
     * device finishes the wave. The last wave's slots stay held through the
     * guess, whose copies run while its kernels may still be reading them. */
    const size_t W = (size_t)(c->n_slots - g->n_used); /* at least n_used: the cache holds twice that */
    bool ok = true;
    int n_held = 0;
    for (size_t w0 = 0; ok && w0 < n_want; w0 += W) {
        const size_t w1 = w0 + W < n_want ? w0 + W : n_want;
        if (w0 > 0) {
            e = cudaStreamSynchronize(s);
            ok = e == cudaSuccess;
            for (int i = 0; i < n_held; i++) c->slots[c->held_list[i]].held = false;
            n_held = 0;
        }
        uint32_t batch[EXPERTS_MAX];
        int32_t batch_x[EXPERTS_MAX];
        size_t n_batch = 0;
        for (size_t i = w0; ok && i <= w1; i++) {
            if (i < w1) {
                const int32_t x = g->e_list[i];
                const uint32_t place = (uint32_t)(layer * E + (size_t)x);
                const int sl = c->where[place];
                if (sl >= 0) {
                    eslot_t* xs = &c->slots[sl];
                    xs->held = true;
                    c->held_list[n_held++] = sl;
                    xs->last_use = ++c->clock;
                    if (xs->filling) {
                        const cudaError_t q = cudaEventQuery(xs->filled);
                        if (q == cudaSuccess) xs->filling = false;
                        else if (q == cudaErrorNotReady) cudaGetLastError();
                    }
                    if (xs->filling) c->in_flight += (uint64_t)g->e_count[x];
                    else c->hits += (uint64_t)g->e_count[x];
                    if (xs->guessed) {
                        c->prefetch_used++;
                        xs->guessed = false;
                    }
                    g->e_slot[x] = sl;
                    continue;
                }
                /* Copied once for all its rows: the first row's copy, the others' a hit. */
                c->misses++;
                c->hits += (uint64_t)g->e_count[x] - 1;
                batch[n_batch] = place;
                batch_x[n_batch++] = x;
                if (n_batch < (size_t)g->n_used) continue;
            }
            if (n_batch == 0) continue;
            vitna_expert_data_t data[EXPERTS_MAX];
            if (c->stream) {
                e = drain(c, c->p_max);
                ok = e == cudaSuccess;
                if (ok) {
                    vitna_expert_stream_hold(c->stream, batch, n_batch);
                    if (!vitna_expert_stream_wait(c->stream, batch, n_batch, data)) {
                        for (int h = 0; h < n_held; h++) c->slots[c->held_list[h]].held = false;
                        return fail(err, err_len, "reading an expert from the drive failed");
                    }
                }
            } else {
                for (size_t j = 0; j < n_batch; j++) {
                    for (int p = 0; p < 3; p++) data[j].part[p] = c->parts[(size_t)batch[j] * 3 + p];
                }
            }
            for (size_t j = 0; ok && j < n_batch; j++) {
                /* At most W slots are held, and there are n_used more. */
                const int sl = free_slot(c);
                e = fill(c, sl, batch[j], &data[j], false);
                ok = e == cudaSuccess;
                c->held_list[n_held++] = sl;
                g->e_slot[batch_x[j]] = sl;
                if (!ok && c->stream) vitna_expert_stream_release(c->stream, batch + j + 1, n_batch - j - 1);
            }
            n_batch = 0;
        }
        /* The wave's kernels, in order of expert, each after the copy into its slot. */
        for (size_t i = w0; ok && i < w1; i++) {
            const int32_t x = g->e_list[i];
            const eslot_t* xs = &c->slots[g->e_slot[x]];
            if (xs->filling) {
                e = cudaStreamWaitEvent(s, xs->filled, 0);
                ok = e == cudaSuccess;
                if (!ok) break;
            }
            const unsigned char* base = c->mem + (size_t)g->e_slot[x] * c->slot_bytes;
            const size_t* part = c->part_off + 3 * layer;
            const dlayer_t* L = &g->layers[layer];
            const int cnt = g->e_count[x], off = g->e_at[x];
            gather_rows_kernel<<<dim3(blocks_for((size_t)H / 4, THREADS), cnt), THREADS, 0, s>>>(g->r_xn, g->r_idx + off, H, g->r_xg);
            LAUNCH(L->e_gu, experts_in_rows_kernel, dim3(grid_for(g, I / 2), blocks_for((size_t)cnt, MULTI_MAX)), 0, s, g->r_xg, base + part[0],
                   base + part[1], g->r_act, H, I, cnt);
            LAUNCH(L->e_down, experts_down_rows_kernel, dim3(grid_for(g, H), blocks_for((size_t)cnt, MULTI_MAX)), 0, s, g->r_act,
                   base + part[2], g->r_idx + off, g->r_w + off, g->r_proj, H, I, cnt);
        }
        if (ok) {
            e = cudaGetLastError();
            ok = e == cudaSuccess;
        }
    }
    if (ok) {
        add_rows_kernel<<<blocks_for(n * (size_t)H, THREADS), THREADS, 0, s>>>(g->r_x, g->r_proj, n * (size_t)H);
        e = cudaGetLastError();
        ok = e == cudaSuccess;
    }

    /* The guess: of each row's guess at the next layer, the half the router
     * ranks highest, copied behind, as for a step; and every guessed expert
     * the device lacks, read from the drive where the expert stream lacks it. */
    if (ok && guess) {
        memset(g->e_mark, 0, E);
        size_t n_lacking = 0;
        for (size_t t = 0; t < n; t++) {
            for (size_t j = 0; j < k; j++) {
                const int32_t x = guess[t * k + j];
                if (g->e_mark[x]) continue;
                g->e_mark[x] = 1;
                const uint32_t place = (uint32_t)((layer + 1) * E + (size_t)x);
                if (c->where[place] < 0) g->e_list[n_lacking++] = (int32_t)place;
            }
        }
        if (c->stream && n_lacking) vitna_expert_stream_prefetch(c->stream, (const uint32_t*)g->e_list, n_lacking);
        memset(g->e_mark, 0, E);
        bool room = true;
        for (size_t t = 0; ok && room && t < n; t++) {
            for (size_t j = 0; ok && j < (k + 1) / 2; j++) {
                const int32_t x = guess[t * k + j];
                if (g->e_mark[x]) continue;
                g->e_mark[x] = 1;
                uint32_t place = (uint32_t)((layer + 1) * E + (size_t)x);
                if (c->where[place] >= 0) continue;
                vitna_expert_data_t d;
                if (c->stream) {
                    if (c->p_len >= c->p_max || !vitna_expert_stream_hold_ready(c->stream, place, &d)) continue;
                } else {
                    for (int p = 0; p < 3; p++) d.part[p] = c->parts[(size_t)place * 3 + p];
                }
                const int sl = free_slot(c);
                if (sl < 0) {
                    if (c->stream) vitna_expert_stream_release(c->stream, &place, 1);
                    room = false;
                    break;
                }
                e = fill(c, sl, place, &d, true);
                ok = e == cudaSuccess;
                c->held_list[n_held++] = sl;
                c->prefetched++;
            }
        }
    }
    for (int i = 0; i < n_held; i++) c->slots[c->held_list[i]].held = false;
    if (!ok) return fail_cuda(err, err_len, "the rows' experts could not be run on the device", e);
    return true;
}

bool vitna_cuda_moe_rows_head(struct vitna_cuda_model* g, size_t n, float* const* logits, char* err, size_t err_len) {
    if (!g->moe || n < 1 || n != (size_t)g->r_n) return fail(err, err_len, "the head of %zu rows cannot be run", n);
    size_t first = 0;
    while (first < n && !logits[first]) first++;
    if (first == n) return true;
    cudaError_t e = cudaSetDevice(g->device);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot use the CUDA device", e);
    const int H = g->hidden, V = g->vocab, G = g->rows_g;
    for (size_t r0 = first; r0 < n; r0 += LOGIT_ROWS) {
        const size_t m = n - r0 < LOGIT_ROWS ? n - r0 : LOGIT_ROWS;
        LAUNCH(g->lm_head.dtype, head_multi_kernel, dim3(grid_for(g, V), blocks_for(m, (size_t)G)), (size_t)G * H * sizeof(float), g->stream,
               g->r_x + r0 * H, g->final_norm, g->eps, g->lm_head.w, g->r_logits, H, V, G, (int)m);
        e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaMemcpyAsync(g->h_rlogits, g->r_logits, m * (size_t)V * sizeof(float), cudaMemcpyDeviceToHost, g->stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(g->stream);
        if (e != cudaSuccess) return fail_cuda(err, err_len, "the forward pass failed on the device", e);
        for (size_t i = 0; i < m; i++) {
            if (logits[r0 + i]) memcpy(logits[r0 + i], g->h_rlogits + i * (size_t)V, (size_t)V * sizeof(float));
        }
    }
    return true;
}

/* --- Widening a matrix, for checking (vitna_cuda_widen) --- */

/* Every row of a matrix widened into out, a block a row, as one of the
 * kernels above reads it: how 0 a chunk of eight at a time (load8, the
 * projections), 1 four at a time (load4, gemm_kernel), 2 one at a time
 * (widen_at, the embedding). */
template <vitna_dtype_t DT>
__global__ void widen_rows_kernel(const void* __restrict__ w, int cols, int how, float* __restrict__ out) {
    const void* row = row_at<DT>(w, blockIdx.x, cols);
    float* o = out + (size_t)blockIdx.x * cols;
    if (how == 0) {
        for (int c = threadIdx.x; c < cols / 8; c += THREADS) {
            float v[8];
            load8<DT>(row, c, v);
#pragma unroll
            for (int i = 0; i < 8; i++) o[8 * c + i] = v[i];
        }
    } else if (how == 1) {
        for (int c = threadIdx.x; c < cols / 4; c += THREADS) {
            float v[4];
            load4<DT>(row, 4 * c, v);
#pragma unroll
            for (int i = 0; i < 4; i++) o[4 * c + i] = v[i];
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += THREADS) o[i] = widen_at<DT>(row, i);
    }
}

bool vitna_cuda_widen(const vitna_matrix_t* m, float* out, char* err, size_t err_len) {
    /* Kept between calls: a model's thousands of experts are widened one after another. */
    static void* d_w = NULL;
    static float* d_out = NULL;
    static size_t w_cap = 0, out_cap = 0;
    static float* again = NULL;
    static size_t again_cap = 0;

    if (!known_dtype(m->dtype) || m->cols % 8 != 0) {
        return fail(err, err_len, "a matrix in %s, %zu wide, is not one the GPU's kernels read", vitna_dtype_name(m->dtype), m->cols);
    }
    if (m->rows > (size_t)INT_MAX || m->cols > (size_t)INT_MAX) return fail(err, err_len, "the matrix is too large");
    const size_t bytes = matrix_bytes(m);
    const size_t n = m->rows * m->cols;
    cudaError_t e = cudaSuccess;
    if (bytes > w_cap) {
        if (d_w) cudaFree(d_w);
        e = cudaMalloc(&d_w, bytes);
        w_cap = e == cudaSuccess ? bytes : 0;
    }
    if (e == cudaSuccess && n * sizeof(float) > out_cap) {
        if (d_out) cudaFree(d_out);
        e = cudaMalloc((void**)&d_out, n * sizeof(float));
        out_cap = e == cudaSuccess ? n * sizeof(float) : 0;
    }
    if (e != cudaSuccess) return fail_cuda(err, err_len, "cannot allocate device memory to widen a matrix", e);
    if (n > again_cap) {
        free(again);
        again = (float*)malloc(n * sizeof(float));
        again_cap = again ? n : 0;
        if (!again) return fail(err, err_len, "out of memory");
    }
    e = cudaMemcpy(d_w, m->data, bytes, cudaMemcpyHostToDevice);
    for (int how = 0; e == cudaSuccess && how < 3; how++) {
        /* Bits no widening gives, so a weight left unwritten cannot pass. */
        e = cudaMemset(d_out, 0xff, n * sizeof(float));
        if (e == cudaSuccess) {
            LAUNCH(m->dtype, widen_rows_kernel, (unsigned int)m->rows, 0, 0, d_w, (int)m->cols, how, d_out);
            e = cudaGetLastError();
        }
        if (e == cudaSuccess) e = cudaMemcpy(how == 0 ? out : again, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (e == cudaSuccess && how > 0 && memcmp(out, again, n * sizeof(float)) != 0) {
            static const char* ways[] = { "eight", "four", "one" };
            return fail(err, err_len, "the GPU widens a matrix in %s differently %s at a time than eight at a time", vitna_dtype_name(m->dtype),
                        ways[how]);
        }
    }
    if (e != cudaSuccess) return fail_cuda(err, err_len, "widening a matrix on the device failed", e);
    return true;
}

bool vitna_cuda_expert_check(const vitna_matrix_t* gate, const vitna_matrix_t* up, const vitna_matrix_t* down, const float* norm_w,
                             const float* x, float eps, float* xs, float* act, float* y, char* err, size_t err_len) {
    const int H = (int)gate->cols, I = (int)gate->rows;
    if (!known_dtype(gate->dtype) || up->dtype != gate->dtype || !known_dtype(down->dtype) || up->rows != gate->rows || up->cols != gate->cols ||
        down->rows != gate->cols || down->cols != gate->rows || H % 8 != 0 || I % 8 != 0) {
        return fail(err, err_len, "an expert the GPU's kernels do not take");
    }
    const size_t gb = matrix_bytes(gate), ub = matrix_bytes(up), db = matrix_bytes(down);
    const size_t floats = (size_t)H * 4 + (size_t)I;
    unsigned char* d_w = NULL;
    float* d_f = NULL;
    cudaError_t e = cudaMalloc((void**)&d_w, padded(gb) + padded(ub) + padded(db));
    if (e == cudaSuccess) e = cudaMalloc((void**)&d_f, floats * sizeof(float));
    float* d_x = d_f;
    float* d_norm = d_f + H;
    float* d_xs = d_f + 2 * (size_t)H;
    float* d_y = d_f + 3 * (size_t)H;
    float* d_act = d_f + 4 * (size_t)H;
    experts_t ex;
    memset(&ex, 0, sizeof(ex));
    ex.k = 1;
    ex.gate[0] = d_w;
    ex.up[0] = d_w + padded(gb);
    ex.down[0] = d_w + padded(gb) + padded(ub);
    ex.weight[0] = 1.0f;
    if (e == cudaSuccess) e = cudaMemcpy((void*)ex.gate[0], gate->data, gb, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy((void*)ex.up[0], up->data, ub, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy((void*)ex.down[0], down->data, db, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_x, x, (size_t)H * sizeof(float), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(d_norm, norm_w, (size_t)H * sizeof(float), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) {
        const size_t shared = (size_t)H * sizeof(float);
        norm_out_kernel<<<1, THREADS, shared>>>(d_x, d_norm, eps, d_xs, H);
        LAUNCH(gate->dtype, experts_in_kernel, blocks_for((size_t)I, WARPS), shared, 0, d_x, d_norm, eps, ex, d_act, H, I);
        LAUNCH(down->dtype, experts_parts_kernel, blocks_for((size_t)H, WARPS), 0, 0, d_act, ex, d_y, H, I);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpy(xs, d_xs, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(act, d_act, (size_t)I * sizeof(float), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(y, d_y, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost);
    if (d_w) cudaFree(d_w);
    if (d_f) cudaFree(d_f);
    if (e != cudaSuccess) return fail_cuda(err, err_len, "running an expert on the device failed", e);
    return true;
}
