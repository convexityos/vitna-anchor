/**
 * encoder.c - A BERT-architecture text encoder in float32 on the CPU.
 *
 * See encoder.h. Each step follows transformers' BertModel with eager
 * attention (models/bert/modeling_bert.py) and sentence-transformers' Pooling
 * and Normalize modules, and tests/reference-embed.test.mjs compares the
 * result with reference/bge-small-en-v1.5/fixture.json.
 */

#include "encoder.h"
#include "compat.h"
#include "json.h"
#include "kernels.h"
#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(VITNA_OS_POSIX)
  #include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #define ENC_X86 1
  #include <immintrin.h>
#endif

#if defined(ENC_X86) && (defined(__GNUC__) || defined(__clang__))
  #define ENC_AVX2 __attribute__((target("avx2,fma")))
#else
  #define ENC_AVX2
#endif

/* The most positions one pass holds, across its texts. A text has at most
 * cfg.max_tokens, so texts are grouped into passes of up to this many rows:
 * enough rows that each weight read serves many, and few enough that a
 * pass's activations stay small (about 15 MB a thousand rows for a model
 * 384 wide). */
#define PASS_ROWS 2048

/* Outputs of a matrix product each task computes, and rows each row-wise task takes. */
#define DENSE_CHUNK 32
#define ROW_CHUNK 64

static bool fail(char* err, size_t err_len, const char* fmt, const char* a, const char* b) {
    snprintf(err, err_len, fmt, a ? a : "", b ? b : "");
    return false;
}

/* --- Which path the products take --- */

static int simd_path(void) {
    static int path = -1; /* 0 scalar, 1 avx2+fma */
    if (path < 0) {
        vitna_simd_capabilities_t caps = vitna_detect_simd_capabilities();
#if defined(ENC_X86)
        path = (caps.has_avx2 && caps.has_fma) ? 1 : 0;
#else
        (void)caps;
        path = 0;
#endif
    }
    return path;
}

const char* vitna_encoder_path(void) {
    return simd_path() == 1 ? "avx2+fma" : "scalar";
}

/* --- A pool of threads --- */

typedef void (*task_fn)(void* ctx, size_t task, size_t worker);

typedef struct {
    vitna_encoder_pool_t* pool;
    size_t index;               /* 1 to workers; the caller is 0 */
} worker_arg_t;

struct vitna_encoder_pool {
    size_t workers;             /* threads started besides the caller */
    vitna_thread_t* threads;
    worker_arg_t* args;
    vitna_mutex_t lock;
    vitna_cond_t work, done;
    task_fn fn;
    void* ctx;
    size_t tasks, next, finished;
    uint64_t round;             /* counts the runs handed out, so a worker knows a new one */
    bool stop;
};

/* Take tasks of the current run until none are left. Called with the lock
 * held, and returns with it held. */
static void take_tasks(vitna_encoder_pool_t* p, size_t worker) {
    while (p->next < p->tasks) {
        const size_t task = p->next++;
        const task_fn fn = p->fn;
        void* ctx = p->ctx;
        vitna_mutex_unlock(&p->lock);
        fn(ctx, task, worker);
        vitna_mutex_lock(&p->lock);
        if (++p->finished == p->tasks) vitna_cond_broadcast(&p->done);
    }
}

static void pool_worker(void* arg) {
    worker_arg_t* a = (worker_arg_t*)arg;
    vitna_encoder_pool_t* p = a->pool;
    uint64_t seen = 0;
    vitna_mutex_lock(&p->lock);
    for (;;) {
        while (!p->stop && p->round == seen) vitna_cond_wait(&p->work, &p->lock);
        if (p->stop) break;
        seen = p->round;
        take_tasks(p, a->index);
    }
    vitna_mutex_unlock(&p->lock);
}

static vitna_encoder_pool_t* pool_create(size_t threads) {
    vitna_encoder_pool_t* p = (vitna_encoder_pool_t*)calloc(1, sizeof(vitna_encoder_pool_t));
    if (!p) return NULL;
    vitna_mutex_init(&p->lock);
    vitna_cond_init(&p->work);
    vitna_cond_init(&p->done);
    const size_t want = threads > 1 ? threads - 1 : 0;
    p->threads = (vitna_thread_t*)calloc(want ? want : 1, sizeof(vitna_thread_t));
    p->args = (worker_arg_t*)calloc(want ? want : 1, sizeof(worker_arg_t));
    if (!p->threads || !p->args) {
        free(p->threads);
        free(p->args);
        free(p);
        return NULL;
    }
    for (size_t i = 0; i < want; i++) {
        p->args[i].pool = p;
        p->args[i].index = i + 1;
        if (!vitna_thread_start(&p->threads[i], pool_worker, &p->args[i], false)) break;
        p->workers++;
    }
    return p;
}

static void pool_free(vitna_encoder_pool_t* p) {
    if (!p) return;
    vitna_mutex_lock(&p->lock);
    p->stop = true;
    vitna_cond_broadcast(&p->work);
    vitna_mutex_unlock(&p->lock);
    for (size_t i = 0; i < p->workers; i++) vitna_thread_join(p->threads[i]);
    vitna_cond_destroy(&p->work);
    vitna_cond_destroy(&p->done);
    vitna_mutex_destroy(&p->lock);
    free(p->threads);
    free(p->args);
    free(p);
}

/* Run fn for tasks 0 to tasks - 1 on the pool's threads and the caller's,
 * and return when every one has. */
static void pool_run(vitna_encoder_pool_t* p, size_t tasks, task_fn fn, void* ctx) {
    if (!p || p->workers == 0 || tasks <= 1) {
        for (size_t t = 0; t < tasks; t++) fn(ctx, t, 0);
        return;
    }
    vitna_mutex_lock(&p->lock);
    p->fn = fn;
    p->ctx = ctx;
    p->tasks = tasks;
    p->next = 0;
    p->finished = 0;
    p->round++;
    vitna_cond_broadcast(&p->work);
    take_tasks(p, 0);
    while (p->finished < p->tasks) vitna_cond_wait(&p->done, &p->lock);
    vitna_mutex_unlock(&p->lock);
}

static size_t cpu_count(void) {
#if defined(VITNA_OS_WINDOWS)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors ? (size_t)si.dwNumberOfProcessors : 1;
#else
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (size_t)n : 1;
#endif
}

/* --- Kernels --- */

/* Every output of a product, and every attention score, is one dot product
 * summed the same way wherever it falls: eight running sums over the inputs
 * in steps of eight, those added in a fixed order, then the inputs left over
 * one at a time. So an output's bits depend on its row and its weights
 * alone, not on which rows share its tile, its pass or its thread. */

#if defined(ENC_X86)
ENC_AVX2
static inline float hsum8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

ENC_AVX2
static float dot_avx2(const float* a, const float* b, size_t n) {
    const size_t n8 = n & ~(size_t)7;
    __m256 acc = _mm256_setzero_ps();
    for (size_t k = 0; k < n8; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
    float s = hsum8(acc);
    for (size_t k = n8; k < n; k++) s += a[k] * b[k];
    return s;
}

/* y[i][j] = b[j] + x[i] . w[j], rows 0 to n - 1, outputs j0 to j1 - 1. Four
 * rows by two outputs at a time, each pair with a running sum of its own,
 * so each weight vector loaded serves four rows. */
ENC_AVX2
static void dense_avx2(const float* x, size_t n, const vitna_dense_t* d, float* y, size_t j0, size_t j1) {
    const size_t in = d->in, out = d->out, n8 = in & ~(size_t)7;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float* x0 = x + i * in;
        const float* x1 = x0 + in;
        const float* x2 = x1 + in;
        const float* x3 = x2 + in;
        size_t j = j0;
        for (; j + 2 <= j1; j += 2) {
            const float* w0 = d->w + j * in;
            const float* w1 = w0 + in;
            __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
            for (size_t k = 0; k < n8; k += 8) {
                const __m256 wa = _mm256_loadu_ps(w0 + k), wb = _mm256_loadu_ps(w1 + k);
                __m256 v = _mm256_loadu_ps(x0 + k);
                a00 = _mm256_fmadd_ps(v, wa, a00);
                a01 = _mm256_fmadd_ps(v, wb, a01);
                v = _mm256_loadu_ps(x1 + k);
                a10 = _mm256_fmadd_ps(v, wa, a10);
                a11 = _mm256_fmadd_ps(v, wb, a11);
                v = _mm256_loadu_ps(x2 + k);
                a20 = _mm256_fmadd_ps(v, wa, a20);
                a21 = _mm256_fmadd_ps(v, wb, a21);
                v = _mm256_loadu_ps(x3 + k);
                a30 = _mm256_fmadd_ps(v, wa, a30);
                a31 = _mm256_fmadd_ps(v, wb, a31);
            }
            float s[8] = { hsum8(a00), hsum8(a01), hsum8(a10), hsum8(a11), hsum8(a20), hsum8(a21), hsum8(a30), hsum8(a31) };
            for (size_t k = n8; k < in; k++) {
                s[0] += x0[k] * w0[k];
                s[1] += x0[k] * w1[k];
                s[2] += x1[k] * w0[k];
                s[3] += x1[k] * w1[k];
                s[4] += x2[k] * w0[k];
                s[5] += x2[k] * w1[k];
                s[6] += x3[k] * w0[k];
                s[7] += x3[k] * w1[k];
            }
            const float b0 = d->b ? d->b[j] : 0.0f, b1 = d->b ? d->b[j + 1] : 0.0f;
            for (size_t r = 0; r < 4; r++) {
                y[(i + r) * out + j] = s[2 * r] + b0;
                y[(i + r) * out + j + 1] = s[2 * r + 1] + b1;
            }
        }
        for (; j < j1; j++) {
            const float* w0 = d->w + j * in;
            const float b0 = d->b ? d->b[j] : 0.0f;
            for (size_t r = 0; r < 4; r++) y[(i + r) * out + j] = dot_avx2(x + (i + r) * in, w0, in) + b0;
        }
    }
    for (; i < n; i++) {
        for (size_t j = j0; j < j1; j++) y[i * out + j] = dot_avx2(x + i * in, d->w + j * in, in) + (d->b ? d->b[j] : 0.0f);
    }
}

/* acc[0..n) += p * v[0..n) */
ENC_AVX2
static void axpy_avx2(float p, const float* v, float* acc, size_t n) {
    const size_t n8 = n & ~(size_t)7;
    const __m256 pv = _mm256_set1_ps(p);
    for (size_t k = 0; k < n8; k += 8) _mm256_storeu_ps(acc + k, _mm256_fmadd_ps(pv, _mm256_loadu_ps(v + k), _mm256_loadu_ps(acc + k)));
    for (size_t k = n8; k < n; k++) acc[k] += p * v[k];
}
#endif

static float dot_scalar(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t k = 0; k < n; k++) s += a[k] * b[k];
    return s;
}

static void dense_scalar(const float* x, size_t n, const vitna_dense_t* d, float* y, size_t j0, size_t j1) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = j0; j < j1; j++) y[i * d->out + j] = dot_scalar(x + i * d->in, d->w + j * d->in, d->in) + (d->b ? d->b[j] : 0.0f);
    }
}

static float dot(const float* a, const float* b, size_t n) {
#if defined(ENC_X86)
    if (simd_path() == 1) return dot_avx2(a, b, n);
#endif
    return dot_scalar(a, b, n);
}

static void axpy(float p, const float* v, float* acc, size_t n) {
#if defined(ENC_X86)
    if (simd_path() == 1) {
        axpy_avx2(p, v, acc, n);
        return;
    }
#endif
    for (size_t k = 0; k < n; k++) acc[k] += p * v[k];
}

/* LayerNorm of one row, over n values, with the mean and variance in double. */
static void layer_norm(float* x, const float* w, const float* b, size_t n, float eps) {
    double mean = 0.0, var = 0.0;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    for (size_t i = 0; i < n; i++) {
        const double c = x[i] - mean;
        var += c * c;
    }
    var /= (double)n;
    const double rstd = 1.0 / sqrt(var + (double)eps);
    for (size_t i = 0; i < n; i++) x[i] = (float)((x[i] - mean) * rstd) * w[i] + b[i];
}

/* --- A pass: one group of texts through the model --- */

typedef struct {
    vitna_encoder_t* e;
    size_t n;                    /* texts */
    const int32_t* const* ids;
    const size_t* lens;
    const size_t* offs;          /* each text's first row */
    size_t rows;
    float *x, *q, *k, *v, *ctx, *tmp, *up;
    float* scores;               /* (workers + 1) x max_tokens */
    const vitna_encoder_layer_t* layer;
    /* the product in progress */
    const float* in;
    const vitna_dense_t* dense;
    float* out;
} pass_t;

static void dense_task(void* c, size_t task, size_t worker) {
    (void)worker;
    const pass_t* p = (const pass_t*)c;
    const size_t j0 = task * DENSE_CHUNK;
    const size_t j1 = j0 + DENSE_CHUNK < p->dense->out ? j0 + DENSE_CHUNK : p->dense->out;
#if defined(ENC_X86)
    if (simd_path() == 1) {
        dense_avx2(p->in, p->rows, p->dense, p->out, j0, j1);
        return;
    }
#endif
    dense_scalar(p->in, p->rows, p->dense, p->out, j0, j1);
}

static void dense(pass_t* p, const float* in, const vitna_dense_t* d, float* out) {
    p->in = in;
    p->dense = d;
    p->out = out;
    pool_run(p->e->pool, (d->out + DENSE_CHUNK - 1) / DENSE_CHUNK, dense_task, p);
}

/* Each text's rows: word embedding plus token type 0, plus the position's,
 * in the order BertEmbeddings adds them, then LayerNorm. */
static void embed_task(void* c, size_t t, size_t worker) {
    (void)worker;
    const pass_t* p = (const pass_t*)c;
    const vitna_encoder_t* e = p->e;
    const size_t H = e->cfg.hidden;
    for (size_t pos = 0; pos < p->lens[t]; pos++) {
        float* row = p->x + (p->offs[t] + pos) * H;
        const float* w = e->word + (size_t)p->ids[t][pos] * H;
        const float* tt = e->token_type;
        const float* pe = e->position + pos * H;
        for (size_t i = 0; i < H; i++) row[i] = (w[i] + tt[i]) + pe[i];
        layer_norm(row, e->emb_ln_w, e->emb_ln_b, H, e->cfg.ln_eps);
    }
}

/* One head of one text: each position's scores against every position of the
 * text, scaled, softmaxed, and the values weighted by them. */
static void attention_task(void* c, size_t task, size_t worker) {
    const pass_t* p = (const pass_t*)c;
    const vitna_encoder_config_t* cfg = &p->e->cfg;
    const size_t H = cfg->hidden, dh = cfg->head_dim;
    const size_t t = task / cfg->n_heads, h = task % cfg->n_heads;
    const size_t L = p->lens[t], off = p->offs[t];
    const float scale = 1.0f / sqrtf((float)dh);
    float* s = p->scores + worker * cfg->max_tokens;
    for (size_t i = 0; i < L; i++) {
        const float* qi = p->q + (off + i) * H + h * dh;
        float m = -INFINITY;
        for (size_t j = 0; j < L; j++) {
            s[j] = dot(qi, p->k + (off + j) * H + h * dh, dh) * scale;
            if (s[j] > m) m = s[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < L; j++) {
            s[j] = expf(s[j] - m);
            sum += s[j];
        }
        float* ci = p->ctx + (off + i) * H + h * dh;
        memset(ci, 0, dh * sizeof(float));
        for (size_t j = 0; j < L; j++) axpy(s[j] / sum, p->v + (off + j) * H + h * dh, ci, dh);
    }
}

/* x = LayerNorm(x + tmp), row by row, with the layer's norm after attention
 * (which 0) or after the MLP (which 1). */
static void add_norm_rows(pass_t* p, size_t r0, size_t r1, int which) {
    const size_t H = p->e->cfg.hidden;
    const float* w = which == 0 ? p->layer->attn_ln_w : p->layer->out_ln_w;
    const float* b = which == 0 ? p->layer->attn_ln_b : p->layer->out_ln_b;
    for (size_t r = r0; r < r1; r++) {
        float* x = p->x + r * H;
        const float* a = p->tmp + r * H;
        for (size_t i = 0; i < H; i++) x[i] += a[i];
        layer_norm(x, w, b, H, p->e->cfg.ln_eps);
    }
}

static void add_norm_attn_task(void* c, size_t task, size_t worker) {
    (void)worker;
    pass_t* p = (pass_t*)c;
    const size_t r0 = task * ROW_CHUNK, r1 = r0 + ROW_CHUNK < p->rows ? r0 + ROW_CHUNK : p->rows;
    add_norm_rows(p, r0, r1, 0);
}

static void add_norm_out_task(void* c, size_t task, size_t worker) {
    (void)worker;
    pass_t* p = (pass_t*)c;
    const size_t r0 = task * ROW_CHUNK, r1 = r0 + ROW_CHUNK < p->rows ? r0 + ROW_CHUNK : p->rows;
    add_norm_rows(p, r0, r1, 1);
}

/* GELU in its erf form, as torch.nn.functional.gelu computes it by default. */
static void gelu_task(void* c, size_t task, size_t worker) {
    (void)worker;
    const pass_t* p = (const pass_t*)c;
    const size_t I = p->e->cfg.intermediate;
    const size_t r0 = task * ROW_CHUNK, r1 = r0 + ROW_CHUNK < p->rows ? r0 + ROW_CHUNK : p->rows;
    for (size_t i = r0 * I; i < r1 * I; i++) {
        const float u = p->up[i];
        p->up[i] = u * 0.5f * (1.0f + erff(u * 0.70710678118654752440f));
    }
}

static bool run_pass(vitna_encoder_t* e, const int32_t* const* ids, const size_t* lens, size_t n, vitna_pooling_t pooling, bool normalize,
                     float* out, float* norms) {
    const vitna_encoder_config_t* cfg = &e->cfg;
    const size_t H = cfg->hidden, I = cfg->intermediate;
    pass_t p;
    memset(&p, 0, sizeof(p));
    p.e = e;
    p.n = n;
    p.ids = ids;
    p.lens = lens;
    size_t* offs = (size_t*)malloc(n * sizeof(size_t));
    if (!offs) return false;
    for (size_t t = 0; t < n; t++) {
        offs[t] = p.rows;
        p.rows += lens[t];
    }
    p.offs = offs;
    const size_t workers = e->pool ? e->pool->workers : 0;
    p.x = (float*)malloc(p.rows * H * sizeof(float));
    p.q = (float*)malloc(p.rows * H * sizeof(float));
    p.k = (float*)malloc(p.rows * H * sizeof(float));
    p.v = (float*)malloc(p.rows * H * sizeof(float));
    p.ctx = (float*)malloc(p.rows * H * sizeof(float));
    p.tmp = (float*)malloc(p.rows * H * sizeof(float));
    p.up = (float*)malloc(p.rows * I * sizeof(float));
    p.scores = (float*)malloc((workers + 1) * cfg->max_tokens * sizeof(float));
    const bool ok = p.x && p.q && p.k && p.v && p.ctx && p.tmp && p.up && p.scores;
    if (ok) {
        const size_t row_tasks = (p.rows + ROW_CHUNK - 1) / ROW_CHUNK;
        pool_run(e->pool, n, embed_task, &p);
        for (size_t l = 0; l < cfg->n_layers; l++) {
            p.layer = &e->layers[l];
            dense(&p, p.x, &p.layer->q, p.q);
            dense(&p, p.x, &p.layer->k, p.k);
            dense(&p, p.x, &p.layer->v, p.v);
            pool_run(e->pool, n * cfg->n_heads, attention_task, &p);
            dense(&p, p.ctx, &p.layer->o, p.tmp);
            pool_run(e->pool, row_tasks, add_norm_attn_task, &p);
            dense(&p, p.x, &p.layer->up, p.up);
            pool_run(e->pool, row_tasks, gelu_task, &p);
            dense(&p, p.up, &p.layer->down, p.tmp);
            pool_run(e->pool, row_tasks, add_norm_out_task, &p);
        }
        for (size_t t = 0; t < n; t++) {
            float* o = out + t * H;
            const float* first = p.x + offs[t] * H;
            if (pooling == VITNA_POOL_CLS) {
                memcpy(o, first, H * sizeof(float));
            } else {
                for (size_t i = 0; i < H; i++) {
                    double s = 0.0;
                    for (size_t r = 0; r < lens[t]; r++) s += first[r * H + i];
                    o[i] = (float)(s / (double)lens[t]);
                }
            }
            double ss = 0.0;
            for (size_t i = 0; i < H; i++) ss += (double)o[i] * o[i];
            const double norm = sqrt(ss);
            if (norms) norms[t] = (float)norm;
            if (normalize) {
                /* torch.nn.functional.normalize: divided by the norm, or by
                 * 1e-12 when the norm is smaller. */
                const float d = (float)(norm > 1e-12 ? norm : 1e-12);
                for (size_t i = 0; i < H; i++) o[i] /= d;
            }
        }
    }
    free(offs);
    free(p.x);
    free(p.q);
    free(p.k);
    free(p.v);
    free(p.ctx);
    free(p.tmp);
    free(p.up);
    free(p.scores);
    return ok;
}

bool vitna_encoder_embed(vitna_encoder_t* e, const int32_t* const* ids, const size_t* lens, size_t n, vitna_pooling_t pooling,
                         bool normalize, float* out, float* norms) {
    for (size_t t = 0; t < n; t++) {
        if (lens[t] == 0 || lens[t] > e->cfg.max_tokens) return false;
        for (size_t i = 0; i < lens[t]; i++) {
            if (ids[t][i] < 0 || (size_t)ids[t][i] >= e->cfg.vocab) return false;
        }
    }
    /* Texts in order, as many to a pass as fit in PASS_ROWS rows. */
    for (size_t t0 = 0; t0 < n;) {
        size_t t1 = t0, rows = 0;
        while (t1 < n && (t1 == t0 || rows + lens[t1] <= PASS_ROWS)) rows += lens[t1++];
        if (!run_pass(e, ids + t0, lens + t0, t1 - t0, pooling, normalize, out + t0 * e->cfg.hidden, norms ? norms + t0 : NULL)) return false;
        t0 = t1;
    }
    return true;
}

/* --- Loading --- */

static char* join_path(const char* dir, const char* name) {
    const size_t n = strlen(dir) + strlen(name) + 2;
    char* path = (char*)malloc(n);
    if (path) snprintf(path, n, "%s/%s", dir, name);
    return path;
}

/* A JSON file in dir, parsed; NULL if it is missing or not JSON. */
static vitna_json_doc_t* read_json(const char* dir, const char* name, char* err, size_t err_len) {
    char* path = join_path(dir, name);
    size_t len = 0;
    char* text = path ? vitna_read_file(path, &len) : NULL;
    vitna_json_doc_t* doc = NULL;
    if (text) {
        char jerr[160];
        doc = vitna_json_parse(text, len, jerr, sizeof(jerr));
        if (!doc) fail(err, err_len, "%s is not valid JSON: %s", name, jerr);
    } else {
        fail(err, err_len, "cannot read %s%s", name, NULL);
    }
    free(text);
    free(path);
    return doc;
}

static bool cfg_size(const vitna_json_value_t* cfg, const char* key, size_t* out, bool required, char* err, size_t err_len) {
    const vitna_json_value_t* v = vitna_json_get(cfg, key);
    double d;
    if (!v || vitna_json_is_null(v)) return !required || fail(err, err_len, "config.json: %s is missing%s", key, NULL);
    if (!vitna_json_as_number(v, &d) || d < 1 || d > 1e9 || d != (double)(size_t)d) return fail(err, err_len, "config.json: %s is not a positive integer%s", key, NULL);
    *out = (size_t)d;
    return true;
}

bool vitna_encoder_wanted(const char* dir) {
    char err[64];
    vitna_json_doc_t* doc = read_json(dir, "config.json", err, sizeof(err));
    if (!doc) return false;
    const char* type = vitna_json_as_string(vitna_json_get(vitna_json_root(doc), "model_type"));
    const bool bert = type && strcmp(type, "bert") == 0;
    vitna_json_free(doc);
    return bert;
}

static bool read_config(vitna_encoder_t* e, const char* dir, char* err, size_t err_len) {
    vitna_json_doc_t* doc = read_json(dir, "config.json", err, err_len);
    if (!doc) return false;
    const vitna_json_value_t* cfg = vitna_json_root(doc);
    vitna_encoder_config_t* c = &e->cfg;
    const char* type = vitna_json_as_string(vitna_json_get(cfg, "model_type"));
    const char* act = vitna_json_as_string(vitna_json_get(cfg, "hidden_act"));
    const char* pos = vitna_json_as_string(vitna_json_get(cfg, "position_embedding_type"));
    bool ok = false;
    double d;
    if (!type || strcmp(type, "bert") != 0) {
        fail(err, err_len, "config.json: model_type is %s, and only bert is supported%s", type ? type : "missing", NULL);
    } else if (!act || strcmp(act, "gelu") != 0) {
        fail(err, err_len, "config.json: hidden_act is %s, and only gelu (its erf form) is supported%s", act ? act : "missing", NULL);
    } else if (pos && strcmp(pos, "absolute") != 0) {
        fail(err, err_len, "config.json: position_embedding_type %s is not supported%s", pos, NULL);
    } else if (cfg_size(cfg, "num_hidden_layers", &c->n_layers, true, err, err_len) && cfg_size(cfg, "hidden_size", &c->hidden, true, err, err_len) &&
               cfg_size(cfg, "num_attention_heads", &c->n_heads, true, err, err_len) &&
               cfg_size(cfg, "intermediate_size", &c->intermediate, true, err, err_len) && cfg_size(cfg, "vocab_size", &c->vocab, true, err, err_len) &&
               cfg_size(cfg, "max_position_embeddings", &c->max_positions, true, err, err_len)) {
        c->type_vocab = 2;
        if (!cfg_size(cfg, "type_vocab_size", &c->type_vocab, false, err, err_len)) {
            /* err is set */
        } else if (c->hidden % c->n_heads != 0) {
            fail(err, err_len, "config.json: num_attention_heads does not divide hidden_size%s%s", NULL, NULL);
        } else {
            c->head_dim = c->hidden / c->n_heads;
            c->max_tokens = c->max_positions;
            /* BertConfig's default where config.json leaves it out. */
            c->ln_eps = 1e-12f;
            if (vitna_json_as_number(vitna_json_get(cfg, "layer_norm_eps"), &d)) c->ln_eps = (float)d;
            ok = true;
        }
    }
    vitna_json_free(doc);
    return ok;
}

static bool ends_with(const char* s, const char* suffix) {
    const size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

/* sentence-transformers' modules.json: the transformer, one pooling module
 * and optionally Normalize, nothing else; the pooling module's config; and
 * sentence_bert_config.json's max_seq_length. */
static bool read_sentence_config(vitna_encoder_t* e, const char* dir, char* err, size_t err_len) {
    vitna_json_doc_t* doc = read_json(dir, "modules.json", err, err_len);
    if (!doc) {
        return fail(err, err_len, "no modules.json: this is not a sentence-transformers model, so the pooling it was trained with is not stated%s%s", NULL, NULL);
    }
    const vitna_json_value_t* mods = vitna_json_root(doc);
    const char* pooling_path = NULL;
    bool ok = mods->type == VITNA_JSON_ARRAY && mods->u.array.count >= 2;
    if (!ok) fail(err, err_len, "modules.json: not a list of modules%s%s", NULL, NULL);
    for (size_t i = 0; ok && i < mods->u.array.count; i++) {
        const char* type = vitna_json_as_string(vitna_json_get(mods->u.array.items[i], "type"));
        const char* path = vitna_json_as_string(vitna_json_get(mods->u.array.items[i], "path"));
        if (!type) {
            ok = fail(err, err_len, "modules.json: a module has no type%s%s", NULL, NULL);
        } else if (i == 0) {
            ok = ends_with(type, "models.Transformer") || fail(err, err_len, "modules.json: the first module is %s, not a Transformer%s", type, NULL);
        } else if (ends_with(type, "models.Pooling") && !pooling_path && path) {
            pooling_path = path;
        } else if (ends_with(type, "models.Normalize") && pooling_path && !e->cfg.normalize) {
            e->cfg.normalize = true;
        } else {
            ok = fail(err, err_len, "modules.json: module %s is not supported here%s", type, NULL);
        }
    }
    if (ok && !pooling_path) ok = fail(err, err_len, "modules.json: no Pooling module%s%s", NULL, NULL);
    vitna_json_doc_t* pool_doc = NULL;
    if (ok) {
        char* name = join_path(pooling_path, "config.json");
        pool_doc = name ? read_json(dir, name, err, err_len) : NULL;
        free(name);
        ok = pool_doc != NULL;
    }
    if (ok) {
        const vitna_json_value_t* pc = vitna_json_root(pool_doc);
        bool cls = false, mean = false, other = false, b;
        vitna_json_as_bool(vitna_json_get(pc, "pooling_mode_cls_token"), &cls);
        vitna_json_as_bool(vitna_json_get(pc, "pooling_mode_mean_tokens"), &mean);
        const char* others[] = { "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken" };
        for (size_t i = 0; i < sizeof(others) / sizeof(others[0]); i++) {
            if (vitna_json_as_bool(vitna_json_get(pc, others[i]), &b) && b) other = true;
        }
        double dim = 0;
        if (other || cls == mean) {
            ok = fail(err, err_len, "the pooling module asks for pooling other than the [CLS] position or the mean alone%s%s", NULL, NULL);
        } else if (vitna_json_as_number(vitna_json_get(pc, "word_embedding_dimension"), &dim) && dim != (double)e->cfg.hidden) {
            ok = fail(err, err_len, "the pooling module's word_embedding_dimension is not the model's hidden_size%s%s", NULL, NULL);
        } else {
            e->cfg.pooling = cls ? VITNA_POOL_CLS : VITNA_POOL_MEAN;
        }
    }
    vitna_json_free(pool_doc);
    vitna_json_free(doc);
    if (!ok) return false;

    /* sentence-transformers truncates a text to max_seq_length tokens, the
     * special ones included; a longer text is refused here instead. */
    char ignore[64];
    vitna_json_doc_t* sb = read_json(dir, "sentence_bert_config.json", ignore, sizeof(ignore));
    if (sb) {
        double d;
        if (vitna_json_as_number(vitna_json_get(vitna_json_root(sb), "max_seq_length"), &d) && d >= 2 && d < (double)e->cfg.max_tokens) e->cfg.max_tokens = (size_t)d;
        vitna_json_free(sb);
    }
    return true;
}

static const vitna_tensor_desc_t* find(const vitna_encoder_t* e, const char* prefix, const char* name) {
    char full[VITNA_TENSOR_NAME_MAX];
    snprintf(full, sizeof(full), "%s%s", prefix, name);
    return vitna_safetensors_find(&e->st, full);
}

/* A tensor as float32 of the given shape (cols 0 for a vector): the mapped
 * data itself when it is float32 and aligned, else a copy widened to it. */
static const float* tensor(vitna_encoder_t* e, const char* prefix, const char* name, size_t rows, size_t cols, char* err, size_t err_len) {
    const vitna_tensor_desc_t* t = find(e, prefix, name);
    if (!t) {
        fail(err, err_len, "model.safetensors has no tensor %s%s", name, NULL);
        return NULL;
    }
    const bool shape_ok = cols ? (t->ndim == 2 && t->shape[0] == rows && t->shape[1] == cols) : (t->ndim == 1 && t->shape[0] == rows);
    if (!shape_ok) {
        fail(err, err_len, "tensor %s has a shape other than the config's%s", name, NULL);
        return NULL;
    }
    if (t->dtype != VITNA_DTYPE_F32 && t->dtype != VITNA_DTYPE_F16 && t->dtype != VITNA_DTYPE_BF16) {
        fail(err, err_len, "tensor %s is %s, and only F32, F16 and BF16 are supported", name, vitna_dtype_name(t->dtype));
        return NULL;
    }
    const size_t n = cols ? rows * cols : rows;
    if (t->dtype == VITNA_DTYPE_F32 && ((uintptr_t)t->data_ptr % sizeof(float)) == 0) return (const float*)t->data_ptr;
    float** grown = (float**)realloc(e->owned, (e->n_owned + 1) * sizeof(float*));
    float* f = (float*)malloc(n * sizeof(float));
    if (grown) e->owned = grown;
    if (!grown || !f) {
        free(f);
        fail(err, err_len, "out of memory%s%s", NULL, NULL);
        return NULL;
    }
    vitna_to_f32(t->data_ptr, t->dtype, f, n);
    e->owned[e->n_owned++] = f;
    return f;
}

bool vitna_encoder_load(vitna_encoder_t* e, const char* dir, size_t threads, char* err, size_t err_len) {
    memset(e, 0, sizeof(*e));
    if (!read_config(e, dir, err, err_len) || !read_sentence_config(e, dir, err, err_len)) return false;
    char* path = join_path(dir, "model.safetensors");
    char st_err[256];
    if (!path || !vitna_safetensors_open_ex(path, &e->st, st_err, sizeof(st_err))) {
        fail(err, err_len, "cannot read model.safetensors: %s%s", path ? st_err : "out of memory", NULL);
        free(path);
        return false;
    }
    free(path);
    e->st_open = true;

    const vitna_encoder_config_t* c = &e->cfg;
    const size_t H = c->hidden, I = c->intermediate;
    /* BertModel's own checkpoints name the embeddings at the top; a
     * checkpoint saved from a model with a head puts them under bert. */
    const char* prefix = find(e, "", "embeddings.word_embeddings.weight") ? "" : "bert.";
    bool ok = (e->word = tensor(e, prefix, "embeddings.word_embeddings.weight", c->vocab, H, err, err_len)) &&
              (e->position = tensor(e, prefix, "embeddings.position_embeddings.weight", c->max_positions, H, err, err_len)) &&
              (e->token_type = tensor(e, prefix, "embeddings.token_type_embeddings.weight", c->type_vocab, H, err, err_len)) &&
              (e->emb_ln_w = tensor(e, prefix, "embeddings.LayerNorm.weight", H, 0, err, err_len)) &&
              (e->emb_ln_b = tensor(e, prefix, "embeddings.LayerNorm.bias", H, 0, err, err_len));
    e->layers = ok ? (vitna_encoder_layer_t*)calloc(c->n_layers, sizeof(vitna_encoder_layer_t)) : NULL;
    if (ok && !e->layers) ok = fail(err, err_len, "out of memory%s%s", NULL, NULL);
    for (size_t l = 0; ok && l < c->n_layers; l++) {
        vitna_encoder_layer_t* L = &e->layers[l];
        char name[160];
#define DENSE(field, part, out_, in_)                                                                   \
        snprintf(name, sizeof(name), "encoder.layer.%zu.%s.weight", l, part);                          \
        ok = ok && (L->field.w = tensor(e, prefix, name, out_, in_, err, err_len)) != NULL;          \
        snprintf(name, sizeof(name), "encoder.layer.%zu.%s.bias", l, part);                            \
        ok = ok && (L->field.b = tensor(e, prefix, name, out_, 0, err, err_len)) != NULL;            \
        L->field.out = out_;                                                                           \
        L->field.in = in_;
#define NORM(w_, b_, part)                                                                              \
        snprintf(name, sizeof(name), "encoder.layer.%zu.%s.weight", l, part);                          \
        ok = ok && (L->w_ = tensor(e, prefix, name, H, 0, err, err_len)) != NULL;                     \
        snprintf(name, sizeof(name), "encoder.layer.%zu.%s.bias", l, part);                            \
        ok = ok && (L->b_ = tensor(e, prefix, name, H, 0, err, err_len)) != NULL;
        DENSE(q, "attention.self.query", H, H)
        DENSE(k, "attention.self.key", H, H)
        DENSE(v, "attention.self.value", H, H)
        DENSE(o, "attention.output.dense", H, H)
        NORM(attn_ln_w, attn_ln_b, "attention.output.LayerNorm")
        DENSE(up, "intermediate.dense", I, H)
        DENSE(down, "output.dense", H, I)
        NORM(out_ln_w, out_ln_b, "output.LayerNorm")
#undef DENSE
#undef NORM
    }
    if (ok) {
        e->threads = threads ? threads : cpu_count();
        if (e->threads > 256) e->threads = 256;
        e->pool = pool_create(e->threads);
        if (!e->pool) ok = fail(err, err_len, "out of memory%s%s", NULL, NULL);
    }
    if (!ok) vitna_encoder_free(e);
    return ok;
}

void vitna_encoder_free(vitna_encoder_t* e) {
    pool_free(e->pool);
    for (size_t i = 0; i < e->n_owned; i++) free(e->owned[i]);
    free(e->owned);
    free(e->layers);
    if (e->st_open) vitna_safetensors_close(&e->st);
    memset(e, 0, sizeof(*e));
}
