/**
 * warp.c - Products summed in the GPU's order, on the CPU (warp.h).
 *
 * The AVX2 path keeps the 32 lanes' sums in four registers of eight. For
 * each group of 32 chunks (256 weights, chunk l of the group being lane l's
 * next), it loads eight lanes' chunks as an 8 x 8 block transposed, so that
 * register e holds element e of each lane's chunk, and fuses the eight
 * elements into the lanes' sums in order, against x transposed the same way
 * once a call. So each lane adds its products in the order a GPU lane does;
 * the lanes are then added in warp_sum's tree. BF16 is widened as it is
 * loaded; the other formats a row at a time first.
 */

#include "warp.h"
#include "exact.h"
#include "kernels.h"
#include "ops.h"
#include "pool.h"
#include "quant.h"
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

/* The scalar loop, with row (cols floats) as scratch. */
static void matvec_scalar(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y, float* row) {
    const size_t rb = (size_t)vitna_row_bytes(dtype, cols);
    for (size_t r = 0; r < rows; r++) {
        vitna_to_f32((const char*)w + r * rb, dtype, row, cols);
        float s[32] = { 0 };
        for (size_t c = 0; c < cols / 8; c++) {
            float* lane = &s[c % 32];
            for (size_t e = 0; e < 8; e++) *lane = fmaf(row[8 * c + e], x[8 * c + e], *lane);
        }
        y[r] = tree32(s);
    }
}

bool vitna_warp_matvec_scalar(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y) {
    float* row = (float*)malloc(cols * sizeof(float));
    if (!row) return false;
    matvec_scalar(w, dtype, rows, cols, x, y, row);
    free(row);
    return true;
}

#if defined(VITNA_WARP_X86)
/* Four registers, register i holding half of chunk i and the same half of
 * chunk i + 4 ([chunk i | chunk i + 4]), transposed within each 128-bit
 * half: m[e] then holds element e of the half of chunks 0 to 7. */
VITNA_WARP_AVX2
static void transpose_halves(const __m256* q, __m256* m) {
    const __m256 t0 = _mm256_unpacklo_ps(q[0], q[1]), t1 = _mm256_unpackhi_ps(q[0], q[1]);
    const __m256 t2 = _mm256_unpacklo_ps(q[2], q[3]), t3 = _mm256_unpackhi_ps(q[2], q[3]);
    m[0] = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    m[1] = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    m[2] = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    m[3] = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
}

/* Eight chunks of eight floats, chunk i at w + 8i, as m[e] = element e of
 * each chunk: each half of a chunk is loaded into the half of a register it
 * ends in, which leaves a 4 x 4 transpose in each half to do. */
VITNA_WARP_AVX2
static void chunks_f32(const float* w, __m256* m) {
    __m256 q[8];
    for (int i = 0; i < 4; i++) {
        q[i] = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(w + 8 * i)), _mm_loadu_ps(w + 8 * (i + 4)), 1);
        q[i + 4] = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(w + 8 * i + 4)), _mm_loadu_ps(w + 8 * (i + 4) + 4), 1);
    }
    transpose_halves(q, m);
    transpose_halves(q + 4, m + 4);
}

/* The same from eight chunks of BF16, each widened as vitna_bf16_to_f32
 * widens it, its 16 bits the top half of a float's 32: no row is widened
 * first. */
VITNA_WARP_AVX2
static void chunks_bf16(const uint16_t* w, __m256* m) {
    __m256 q[8];
    for (int i = 0; i < 4; i++) {
        const __m128i a = _mm_loadu_si128((const __m128i*)(w + 8 * i)), b = _mm_loadu_si128((const __m128i*)(w + 8 * (i + 4)));
        q[i] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_unpacklo_epi64(a, b)), 16));
        q[i + 4] = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_unpackhi_epi64(a, b)), 16));
    }
    transpose_halves(q, m);
    transpose_halves(q + 4, m + 4);
}

/* With row and xt (cols floats each) as scratch; a BF16 row is read as it is. */
VITNA_WARP_AVX2
static void matvec_avx2(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y, float* row, float* xt) {
    const size_t rb = (size_t)vitna_row_bytes(dtype, cols);
    const size_t groups = cols / 256;
    const bool bf16 = dtype == VITNA_DTYPE_BF16;
    /* xt[(j * 8 + e) * 32 + l] = element e of lane l's chunk in group j. */
    for (size_t j = 0; j < groups; j++) {
        for (size_t l = 0; l < 32; l++) {
            for (size_t e = 0; e < 8; e++) xt[(j * 8 + e) * 32 + l] = x[(32 * j + l) * 8 + e];
        }
    }
    for (size_t r = 0; r < rows; r++) {
        const char* src = (const char*)w + r * rb;
        if (!bf16) vitna_to_f32(src, dtype, row, cols);
        __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (size_t j = 0; j < groups; j++) {
            const float* xg = xt + j * 256;
            for (int g = 0; g < 4; g++) {
                /* Lanes 8g to 8g + 7's chunks of group j. */
                __m256 m[8];
                if (bf16) chunks_bf16((const uint16_t*)src + 256 * j + 64 * g, m);
                else chunks_f32(row + 256 * j + 64 * g, m);
                for (int e = 0; e < 8; e++) acc[g] = _mm256_fmadd_ps(m[e], _mm256_loadu_ps(xg + e * 32 + 8 * g), acc[g]);
            }
        }
        float s[32];
        for (int g = 0; g < 4; g++) _mm256_storeu_ps(s + 8 * g, acc[g]);
        y[r] = tree32(s);
    }
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

/* Either path, with scratch of 2 * cols floats. */
static void matvec(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y, float* scratch) {
#if defined(VITNA_WARP_X86)
    if (use_avx2(cols)) {
        matvec_avx2(w, dtype, rows, cols, x, y, scratch, scratch + cols);
        return;
    }
#endif
    matvec_scalar(w, dtype, rows, cols, x, y, scratch);
}

bool vitna_warp_matvec(const void* w, vitna_dtype_t dtype, size_t rows, size_t cols, const float* x, float* y) {
    float* scratch = (float*)malloc(2 * cols * sizeof(float));
    if (!scratch) return false;
    matvec(w, dtype, rows, cols, x, y, scratch);
    free(scratch);
    return true;
}

size_t vitna_warp_scratch_floats(size_t hidden, size_t intermediate) {
    return 2 * (hidden > intermediate ? hidden : intermediate);
}

/* The rows of gate and up from r0 to r1, and their activations. */
static void expert_in_rows(const vitna_warp_expert_t* x, vitna_dtype_t gu_dtype, size_t hidden, size_t intermediate, const float* xs,
                           float* act, size_t r0, size_t r1, float* scratch) {
    const size_t rb = (size_t)vitna_row_bytes(gu_dtype, hidden);
    float* g = act + r0;
    float* u = act + intermediate + r0;
    matvec((const char*)x->gate + r0 * rb, gu_dtype, r1 - r0, hidden, xs, g, scratch);
    matvec((const char*)x->up + r0 * rb, gu_dtype, r1 - r0, hidden, xs, u, scratch);
    vitna_silu_mul_exact(g, u, g, r1 - r0);
}

bool vitna_warp_expert(const void* gate, const void* up, const void* down, vitna_dtype_t gu_dtype, vitna_dtype_t down_dtype,
                       size_t hidden, size_t intermediate, const float* xs, float* y, float* act) {
    float* scratch = (float*)malloc(vitna_warp_scratch_floats(hidden, intermediate) * sizeof(float));
    if (!scratch) return false;
    const vitna_warp_expert_t x = { gate, up, down, y };
    expert_in_rows(&x, gu_dtype, hidden, intermediate, xs, act, 0, intermediate, scratch);
    matvec(down, down_dtype, hidden, intermediate, act, y, scratch);
    free(scratch);
    return true;
}

/* --- Several experts, their rows shared among a pool's threads --- */

/* Rows a task takes: enough that a task's work outweighs handing it out,
 * few enough that a layer's experts give every thread several. */
#define TASK_ROWS 64

typedef struct {
    const vitna_warp_expert_t* experts;
    vitna_dtype_t gu_dtype, down_dtype;
    size_t hidden, intermediate;
    const float* xs;
    float* act;
    float* scratch;
    size_t scratch_floats;
    size_t per;              /* tasks per expert in the phase under way */
} experts_job_t;

static void experts_in_task(void* arg, size_t i, size_t worker) {
    const experts_job_t* j = (const experts_job_t*)arg;
    const size_t k = i / j->per, r0 = (i % j->per) * TASK_ROWS;
    const size_t r1 = r0 + TASK_ROWS < j->intermediate ? r0 + TASK_ROWS : j->intermediate;
    expert_in_rows(&j->experts[k], j->gu_dtype, j->hidden, j->intermediate, j->xs, j->act + k * 2 * j->intermediate, r0, r1,
                   j->scratch + worker * j->scratch_floats);
}

static void experts_down_task(void* arg, size_t i, size_t worker) {
    const experts_job_t* j = (const experts_job_t*)arg;
    const size_t k = i / j->per, r0 = (i % j->per) * TASK_ROWS;
    const size_t r1 = r0 + TASK_ROWS < j->hidden ? r0 + TASK_ROWS : j->hidden;
    const size_t rb = (size_t)vitna_row_bytes(j->down_dtype, j->intermediate);
    matvec((const char*)j->experts[k].down + r0 * rb, j->down_dtype, r1 - r0, j->intermediate, j->act + k * 2 * j->intermediate,
           j->experts[k].y + r0, j->scratch + worker * j->scratch_floats);
}

void vitna_warp_experts(vitna_pool_t* pool, const vitna_warp_expert_t* experts, size_t n, vitna_dtype_t gu_dtype, vitna_dtype_t down_dtype,
                        size_t hidden, size_t intermediate, const float* xs, float* act, float* scratch) {
    experts_job_t j = { experts, gu_dtype, down_dtype, hidden, intermediate, xs, act, scratch,
                        vitna_warp_scratch_floats(hidden, intermediate), 0 };
    /* Which paths this CPU takes is decided on first use and kept, so it is
     * decided here, on this thread, before the pool's threads ask: these
     * products' path, and quant.c's for widening, which widening nothing
     * decides. */
    (void)use_avx2(0);
    vitna_dequant(NULL, gu_dtype, NULL, 0);
    j.per = (intermediate + TASK_ROWS - 1) / TASK_ROWS;
    vitna_pool_run(pool, n * j.per, experts_in_task, &j);
    j.per = (hidden + TASK_ROWS - 1) / TASK_ROWS;
    vitna_pool_run(pool, n * j.per, experts_down_task, &j);
}
