/**
 * model.c - A dense Llama-architecture model on the CPU, in float32, and the
 * switch that moves its forward pass to a GPU in an engine built with CUDA.
 */

#include "model.h"
#include "json.h"
#include "kernels.h"
#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(VITNA_CUDA)
  #include "model_cuda.h"
#endif

#define NO_CUDA "this engine was built without CUDA, so it cannot use a GPU. Build it with the CUDA path: make VITNA_CUDA=1, or cmake -DVITNA_CUDA=ON"

static bool fail(char* err, size_t err_len, const char* fmt, const char* a, const char* b) {
    if (err && err_len > 0) snprintf(err, err_len, fmt, a ? a : "", b ? b : "");
    return false;
}

static bool cfg_size(const vitna_json_value_t* cfg, const char* key, size_t* out, bool required, char* err, size_t err_len) {
    const vitna_json_value_t* v = vitna_json_get(cfg, key);
    double d;
    if (!v || vitna_json_is_null(v)) {
        if (required) return fail(err, err_len, "config.json: %s is missing%s", key, NULL);
        return true;
    }
    if (!vitna_json_as_number(v, &d) || d < 1 || d > 1e9 || d != (double)(size_t)d) {
        return fail(err, err_len, "config.json: %s is not a positive integer%s", key, NULL);
    }
    *out = (size_t)d;
    return true;
}

static bool cfg_false_or_missing(const vitna_json_value_t* cfg, const char* key) {
    const vitna_json_value_t* v = vitna_json_get(cfg, key);
    bool b;
    return !v || vitna_json_is_null(v) || (vitna_json_as_bool(v, &b) && !b);
}

static bool read_config(vitna_llama_config_t* c, const char* dir, char* err, size_t err_len) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    size_t len;
    char* text = vitna_read_file(path, &len);
    if (!text) return fail(err, err_len, "cannot read %s%s", path, NULL);
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(text, len, jerr, sizeof(jerr));
    free(text);
    if (!doc) return fail(err, err_len, "config.json: %s%s", jerr, NULL);
    const vitna_json_value_t* cfg = vitna_json_root(doc);
    bool ok = false;

    const char* model_type = vitna_json_as_string(vitna_json_get(cfg, "model_type"));
    const char* act = vitna_json_as_string(vitna_json_get(cfg, "hidden_act"));
    const vitna_json_value_t* scaling = vitna_json_get(cfg, "rope_scaling");
    double d;
    memset(c, 0, sizeof(*c));
    if (!model_type || strcmp(model_type, "llama") != 0) {
        fail(err, err_len, "config.json: model_type is %s, and only llama is supported%s", model_type ? model_type : "missing", NULL);
    } else if (!act || strcmp(act, "silu") != 0) {
        fail(err, err_len, "config.json: hidden_act is %s, and only silu is supported%s", act ? act : "missing", NULL);
    } else if (scaling && !vitna_json_is_null(scaling)) {
        fail(err, err_len, "config.json: rope_scaling is not supported%s%s", NULL, NULL);
    } else if (!cfg_false_or_missing(cfg, "attention_bias") || !cfg_false_or_missing(cfg, "mlp_bias")) {
        fail(err, err_len, "config.json: attention and MLP biases are not supported%s%s", NULL, NULL);
    } else if (!cfg_false_or_missing(cfg, "rope_interleaved")) {
        /* Hugging Face's Llama always rotates in half-split form. A config
         * asking for interleaved pairs means something this cannot check. */
        fail(err, err_len, "config.json: rope_interleaved true is not supported%s%s", NULL, NULL);
    } else if (cfg_size(cfg, "num_hidden_layers", &c->n_layers, true, err, err_len) &&
               cfg_size(cfg, "hidden_size", &c->hidden, true, err, err_len) &&
               cfg_size(cfg, "intermediate_size", &c->intermediate, true, err, err_len) &&
               cfg_size(cfg, "num_attention_heads", &c->n_heads, true, err, err_len) &&
               cfg_size(cfg, "vocab_size", &c->vocab, true, err, err_len) &&
               cfg_size(cfg, "max_position_embeddings", &c->max_positions, true, err, err_len)) {
        c->n_kv_heads = c->n_heads;
        c->head_dim = c->hidden / c->n_heads;
        size_t tp = 1;
        if (!cfg_size(cfg, "num_key_value_heads", &c->n_kv_heads, false, err, err_len) ||
            !cfg_size(cfg, "head_dim", &c->head_dim, false, err, err_len) ||
            !cfg_size(cfg, "pretraining_tp", &tp, false, err, err_len)) {
            /* err is set */
        } else if (tp != 1) {
            fail(err, err_len, "config.json: pretraining_tp other than 1 is not supported%s%s", NULL, NULL);
        } else if (c->n_heads % c->n_kv_heads != 0 || c->head_dim % 2 != 0) {
            fail(err, err_len, "config.json: heads do not divide into key-value groups, or head_dim is odd%s%s", NULL, NULL);
        } else {
            c->rms_eps = 1e-6f;
            c->rope_theta = 10000.0f;
            if (vitna_json_as_number(vitna_json_get(cfg, "rms_norm_eps"), &d)) c->rms_eps = (float)d;
            if (vitna_json_as_number(vitna_json_get(cfg, "rope_theta"), &d)) c->rope_theta = (float)d;
            c->tied_embeddings = false;
            vitna_json_as_bool(vitna_json_get(cfg, "tie_word_embeddings"), &c->tied_embeddings);
            ok = true;
        }
    }
    vitna_json_free(doc);
    return ok;
}

static bool bind(vitna_llama_t* m, vitna_matrix_t* out, const char* name, size_t rows, size_t cols, char* err, size_t err_len) {
    const vitna_tensor_desc_t* t = vitna_safetensors_find(&m->st, name);
    if (!t) return fail(err, err_len, "model.safetensors has no tensor %s%s", name, NULL);
    bool shape_ok = (cols == 0) ? (t->ndim == 1 && t->shape[0] == rows) : (t->ndim == 2 && t->shape[0] == rows && t->shape[1] == cols);
    if (!shape_ok) return fail(err, err_len, "tensor %s has an unexpected shape%s", name, NULL);
    if (t->dtype != VITNA_DTYPE_F32 && t->dtype != VITNA_DTYPE_BF16 && t->dtype != VITNA_DTYPE_F16) {
        return fail(err, err_len, "tensor %s is %s; F32, BF16 or F16 is needed", name, vitna_dtype_name(t->dtype));
    }
    out->data = t->data_ptr;
    out->dtype = t->dtype;
    out->rows = rows;
    out->cols = cols ? cols : 1;
    return true;
}

static float* norm_weights(vitna_llama_t* m, const char* name, char* err, size_t err_len) {
    vitna_matrix_t w;
    if (!bind(m, &w, name, m->cfg.hidden, 0, err, err_len)) return NULL;
    float* f = (float*)malloc(m->cfg.hidden * sizeof(float));
    if (!f) { fail(err, err_len, "out of memory%s%s", NULL, NULL); return NULL; }
    vitna_to_f32(w.data, w.dtype, f, m->cfg.hidden);
    return f;
}

bool vitna_llama_load(vitna_llama_t* m, const char* dir, size_t ctx, char* err, size_t err_len) {
    memset(m, 0, sizeof(*m));
    if (!read_config(&m->cfg, dir, err, err_len)) return false;
    const vitna_llama_config_t* c = &m->cfg;

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    char st_err[256];
    if (!vitna_safetensors_open_ex(path, &m->st, st_err, sizeof(st_err))) {
        return fail(err, err_len, "%s%s (a sharded checkpoint is not supported yet)", st_err, NULL);
    }

    const size_t q_dim = c->n_heads * c->head_dim;
    const size_t kv_dim = c->n_kv_heads * c->head_dim;
    char name[256];
    bool ok = bind(m, &m->embed, "model.embed_tokens.weight", c->vocab, c->hidden, err, err_len);
    if (ok) {
        if (vitna_safetensors_find(&m->st, "lm_head.weight")) {
            ok = bind(m, &m->lm_head, "lm_head.weight", c->vocab, c->hidden, err, err_len);
        } else if (c->tied_embeddings) {
            m->lm_head = m->embed;
        } else {
            ok = fail(err, err_len, "no lm_head.weight, and the config does not tie embeddings%s%s", NULL, NULL);
        }
    }
    if (ok) ok = (m->final_norm = norm_weights(m, "model.norm.weight", err, err_len)) != NULL;
    if (ok) {
        m->layers = (vitna_llama_layer_t*)calloc(c->n_layers, sizeof(vitna_llama_layer_t));
        ok = m->layers != NULL || fail(err, err_len, "out of memory%s%s", NULL, NULL);
    }
    for (size_t l = 0; ok && l < c->n_layers; l++) {
        vitna_llama_layer_t* L = &m->layers[l];
#define LAYER(field, suffix, rows, cols) \
        (snprintf(name, sizeof(name), "model.layers.%zu.%s", l, suffix), bind(m, &L->field, name, rows, cols, err, err_len))
        ok = LAYER(q, "self_attn.q_proj.weight", q_dim, c->hidden) &&
             LAYER(k, "self_attn.k_proj.weight", kv_dim, c->hidden) &&
             LAYER(v, "self_attn.v_proj.weight", kv_dim, c->hidden) &&
             LAYER(o, "self_attn.o_proj.weight", c->hidden, q_dim) &&
             LAYER(gate, "mlp.gate_proj.weight", c->intermediate, c->hidden) &&
             LAYER(up, "mlp.up_proj.weight", c->intermediate, c->hidden) &&
             LAYER(down, "mlp.down_proj.weight", c->hidden, c->intermediate);
#undef LAYER
        if (ok) {
            snprintf(name, sizeof(name), "model.layers.%zu.input_layernorm.weight", l);
            ok = (L->attn_norm = norm_weights(m, name, err, err_len)) != NULL;
        }
        if (ok) {
            snprintf(name, sizeof(name), "model.layers.%zu.post_attention_layernorm.weight", l);
            ok = (L->mlp_norm = norm_weights(m, name, err, err_len)) != NULL;
        }
    }

    if (ok) {
        m->ctx = ctx ? ctx : (c->max_positions < 4096 ? c->max_positions : 4096);
        if (m->ctx > c->max_positions) m->ctx = c->max_positions;
        size_t cache = c->n_layers * m->ctx * kv_dim;
        size_t half = c->head_dim / 2;
        m->k_cache = (float*)malloc(cache * sizeof(float));
        m->v_cache = (float*)malloc(cache * sizeof(float));
        m->inv_freq = (float*)malloc(half * sizeof(float));
        m->x = (float*)malloc(c->hidden * sizeof(float));
        m->xn = (float*)malloc(c->hidden * sizeof(float));
        m->q = (float*)malloc(q_dim * sizeof(float));
        m->k = (float*)malloc(kv_dim * sizeof(float));
        m->v = (float*)malloc(kv_dim * sizeof(float));
        m->att = (float*)malloc(q_dim * sizeof(float));
        m->proj = (float*)malloc(c->hidden * sizeof(float));
        m->gate = (float*)malloc(c->intermediate * sizeof(float));
        m->up = (float*)malloc(c->intermediate * sizeof(float));
        m->scores = (float*)malloc(m->ctx * sizeof(float));
        m->cos_t = (float*)malloc(half * sizeof(float));
        m->sin_t = (float*)malloc(half * sizeof(float));
        ok = m->k_cache && m->v_cache && m->inv_freq && m->x && m->xn && m->q && m->k && m->v && m->att &&
             m->proj && m->gate && m->up && m->scores && m->cos_t && m->sin_t;
        if (!ok) {
            fail(err, err_len, "out of memory for the key-value cache and scratch buffers%s%s", NULL, NULL);
        } else {
            /* As transformers computes it, in float32:
             * 1 / theta ** (arange(0, head_dim, 2) / head_dim) */
            for (size_t j = 0; j < half; j++) {
                float e = (float)(2 * j) / (float)c->head_dim;
                m->inv_freq[j] = 1.0f / powf(c->rope_theta, e);
            }
        }
    }
    if (!ok) vitna_llama_free(m);
    return ok;
}

void vitna_llama_free(vitna_llama_t* m) {
    if (!m) return;
    if (m->layers) {
        for (size_t l = 0; l < m->cfg.n_layers; l++) {
            free(m->layers[l].attn_norm);
            free(m->layers[l].mlp_norm);
        }
        free(m->layers);
    }
    free(m->final_norm);
    free(m->inv_freq);
    free(m->k_cache);
    free(m->v_cache);
    free(m->x); free(m->xn); free(m->q); free(m->k); free(m->v); free(m->att);
    free(m->proj); free(m->gate); free(m->up); free(m->scores); free(m->cos_t); free(m->sin_t);
#if defined(VITNA_CUDA)
    vitna_cuda_free(m->cuda);
#endif
    vitna_safetensors_close(&m->st);
    memset(m, 0, sizeof(*m));
}

void vitna_llama_reset(vitna_llama_t* m) {
    m->n_past = 0;
}

void vitna_llama_truncate(vitna_llama_t* m, size_t n) {
    if (n < m->n_past) m->n_past = n;
}

bool vitna_llama_step(vitna_llama_t* m, int32_t token, float* logits) {
    const vitna_llama_config_t* c = &m->cfg;
    if (token < 0 || (size_t)token >= c->vocab || m->n_past >= m->ctx) return false;

#if defined(VITNA_CUDA)
    if (m->cuda) {
        char err[512];
        if (!vitna_cuda_step(m->cuda, token, m->n_past, logits, err, sizeof(err))) {
            fprintf(stderr, "CUDA: %s\n", err);
            return false;
        }
        m->n_past++;
        return true;
    }
#endif

    const size_t pos = m->n_past;
    const size_t H = c->hidden, hd = c->head_dim, half = hd / 2;
    const size_t kv_dim = c->n_kv_heads * hd;
    const size_t group = c->n_heads / c->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);

    /* The token's embedding row, widened to float32. */
    const size_t esize = vitna_dtype_size(m->embed.dtype);
    vitna_to_f32((const char*)m->embed.data + (size_t)token * H * esize, m->embed.dtype, m->x, H);

    /* Rotary angles for this position: position * inv_freq, in float32. */
    for (size_t j = 0; j < half; j++) {
        float angle = (float)pos * m->inv_freq[j];
        m->cos_t[j] = cosf(angle);
        m->sin_t[j] = sinf(angle);
    }

    for (size_t l = 0; l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        float* kc = m->k_cache + l * m->ctx * kv_dim;
        float* vc = m->v_cache + l * m->ctx * kv_dim;

        /* Attention */
        vitna_rmsnorm(m->x, L->attn_norm, m->xn, H, c->rms_eps);
        vitna_matvec(L->q.data, L->q.dtype, m->xn, m->q, L->q.rows, L->q.cols);
        vitna_matvec(L->k.data, L->k.dtype, m->xn, m->k, L->k.rows, L->k.cols);
        vitna_matvec(L->v.data, L->v.dtype, m->xn, m->v, L->v.rows, L->v.cols);
        for (size_t h = 0; h < c->n_heads; h++) vitna_rope_half(m->q + h * hd, hd, m->cos_t, m->sin_t);
        for (size_t h = 0; h < c->n_kv_heads; h++) vitna_rope_half(m->k + h * hd, hd, m->cos_t, m->sin_t);
        memcpy(kc + pos * kv_dim, m->k, kv_dim * sizeof(float));
        memcpy(vc + pos * kv_dim, m->v, kv_dim * sizeof(float));

        for (size_t h = 0; h < c->n_heads; h++) {
            const float* qh = m->q + h * hd;
            const size_t kvh = h / group; /* repeat_kv: query head h reads key-value head h / group */
            float max = -INFINITY;
            for (size_t t = 0; t <= pos; t++) {
                const float* kt = kc + t * kv_dim + kvh * hd;
                float s = 0.0f;
                for (size_t d = 0; d < hd; d++) s += qh[d] * kt[d];
                s *= scale;
                m->scores[t] = s;
                if (s > max) max = s;
            }
            float sum = 0.0f;
            for (size_t t = 0; t <= pos; t++) {
                m->scores[t] = expf(m->scores[t] - max);
                sum += m->scores[t];
            }
            float inv = 1.0f / sum;
            float* out = m->att + h * hd;
            memset(out, 0, hd * sizeof(float));
            for (size_t t = 0; t <= pos; t++) {
                const float p = m->scores[t] * inv;
                const float* vt = vc + t * kv_dim + kvh * hd;
                for (size_t d = 0; d < hd; d++) out[d] += p * vt[d];
            }
        }
        vitna_matvec(L->o.data, L->o.dtype, m->att, m->proj, L->o.rows, L->o.cols);
        for (size_t i = 0; i < H; i++) m->x[i] += m->proj[i];

        /* MLP */
        vitna_rmsnorm(m->x, L->mlp_norm, m->xn, H, c->rms_eps);
        vitna_matvec(L->gate.data, L->gate.dtype, m->xn, m->gate, L->gate.rows, L->gate.cols);
        vitna_matvec(L->up.data, L->up.dtype, m->xn, m->up, L->up.rows, L->up.cols);
        vitna_silu_mul(m->gate, m->up, m->gate, c->intermediate);
        vitna_matvec(L->down.data, L->down.dtype, m->gate, m->proj, L->down.rows, L->down.cols);
        for (size_t i = 0; i < H; i++) m->x[i] += m->proj[i];
    }

    m->n_past = pos + 1;
    if (logits) {
        vitna_rmsnorm(m->x, m->final_norm, m->xn, H, c->rms_eps);
        vitna_matvec(m->lm_head.data, m->lm_head.dtype, m->xn, logits, m->lm_head.rows, m->lm_head.cols);
    }
    return true;
}

/* --- The GPU --- */

bool vitna_llama_cuda_built(void) {
#if defined(VITNA_CUDA)
    return true;
#else
    return false;
#endif
}

bool vitna_llama_cuda_probe(char* err, size_t err_len) {
#if defined(VITNA_CUDA)
    return vitna_cuda_probe(err, err_len);
#else
    return fail(err, err_len, "%s%s", NO_CUDA, NULL);
#endif
}

#if defined(VITNA_CUDA)
/* The rotary cos and sin of every position the cache holds, [ctx][head_dim / 2],
 * with the float32 formula vitna_llama_step uses for one position. */
static bool rope_tables(const vitna_llama_t* m, float** cos_out, float** sin_out) {
    const size_t half = m->cfg.head_dim / 2;
    float* ct = (float*)malloc(m->ctx * half * sizeof(float));
    float* st = (float*)malloc(m->ctx * half * sizeof(float));
    if (!ct || !st) {
        free(ct);
        free(st);
        return false;
    }
    for (size_t pos = 0; pos < m->ctx; pos++) {
        for (size_t j = 0; j < half; j++) {
            float angle = (float)pos * m->inv_freq[j];
            ct[pos * half + j] = cosf(angle);
            st[pos * half + j] = sinf(angle);
        }
    }
    *cos_out = ct;
    *sin_out = st;
    return true;
}
#endif

bool vitna_llama_use_cuda(vitna_llama_t* m, char* err, size_t err_len) {
#if defined(VITNA_CUDA)
    if (m->cuda) return true;
    float *ct = NULL, *st = NULL;
    if (!rope_tables(m, &ct, &st)) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
    m->cuda = vitna_cuda_create(m, ct, st, err, err_len);
    free(ct);
    free(st);
    if (!m->cuda) return false;
    /* The device holds the key-value cache from here on. */
    free(m->k_cache);
    free(m->v_cache);
    m->k_cache = NULL;
    m->v_cache = NULL;
    return true;
#else
    (void)m;
    return fail(err, err_len, "%s%s", NO_CUDA, NULL);
#endif
}

const char* vitna_llama_device(const vitna_llama_t* m, char* buf, size_t len) {
#if defined(VITNA_CUDA)
    if (m->cuda) {
        snprintf(buf, len, "on CUDA device 0, %s", vitna_cuda_device_name(m->cuda));
        return buf;
    }
#endif
    (void)m;
    snprintf(buf, len, "matvec path %s", vitna_matvec_path());
    return buf;
}
