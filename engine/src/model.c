/**
 * model.c - A dense Llama-architecture model on the CPU, in float32, and the
 * switch that moves its forward pass to a GPU in an engine built with CUDA.
 */

#include "model.h"
#include "expert_stream.h"
#include "gguf.h"
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
    const vitna_json_value_t* clip = vitna_json_get(cfg, "clip_qkv");
    const bool olmoe = model_type && strcmp(model_type, "olmoe") == 0;
    const bool qwen3 = model_type && strcmp(model_type, "qwen3_moe") == 0;
    const bool moe = olmoe || qwen3;
    /* Qwen3-MoE's dense layers among the experts', which its configs leave empty. */
    const vitna_json_value_t* dense = vitna_json_get(cfg, "mlp_only_layers");
    const bool some_dense = dense && !vitna_json_is_null(dense) && !(dense->type == VITNA_JSON_ARRAY && dense->u.array.count == 0);
    double d;
    memset(c, 0, sizeof(*c));
    if (!model_type || (strcmp(model_type, "llama") != 0 && !moe)) {
        fail(err, err_len, "config.json: model_type is %s, and only llama, olmoe and qwen3_moe are supported%s", model_type ? model_type : "missing",
             NULL);
    } else if (olmoe && clip && !vitna_json_is_null(clip)) {
        fail(err, err_len, "config.json: clip_qkv is not supported%s%s", NULL, NULL);
    } else if (qwen3 && !cfg_false_or_missing(cfg, "use_sliding_window")) {
        fail(err, err_len, "config.json: use_sliding_window true is not supported%s%s", NULL, NULL);
    } else if (qwen3 && (some_dense || (vitna_json_as_number(vitna_json_get(cfg, "decoder_sparse_step"), &d) && d != 1.0))) {
        fail(err, err_len, "config.json: dense layers among the experts' (mlp_only_layers, decoder_sparse_step) are not supported%s%s", NULL, NULL);
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
        } else if (moe && !(cfg_size(cfg, "num_experts", &c->n_experts, true, err, err_len) &&
                            cfg_size(cfg, "num_experts_per_tok", &c->n_experts_used, true, err, err_len))) {
            /* err is set */
        } else if (moe && (c->n_experts_used > c->n_experts || c->n_experts_used > VITNA_EXPERTS_USED_MAX)) {
            fail(err, err_len, "config.json: num_experts_per_tok is more than num_experts, or more than this engine takes%s%s", NULL, NULL);
        } else if (qwen3 && !cfg_size(cfg, "moe_intermediate_size", &c->intermediate, true, err, err_len)) {
            /* err is set. Qwen3-MoE's intermediate_size is the width of a dense layer it does not have; an expert's is this. */
        } else {
            /* Where config.json leaves it out, each config class's default:
             * LlamaConfig's and Qwen3MoeConfig's 1e-6, OlmoeConfig's 1e-5. */
            c->rms_eps = olmoe ? 1e-5f : 1e-6f;
            c->rope_theta = 10000.0f;
            if (vitna_json_as_number(vitna_json_get(cfg, "rms_norm_eps"), &d)) c->rms_eps = (float)d;
            if (vitna_json_as_number(vitna_json_get(cfg, "rope_theta"), &d)) c->rope_theta = (float)d;
            c->tied_embeddings = false;
            vitna_json_as_bool(vitna_json_get(cfg, "tie_word_embeddings"), &c->tied_embeddings);
            c->qk_norm = moe;
            c->qk_norm_per_head = qwen3;
            /* OlmoeConfig's and Qwen3MoeConfig's default is false. */
            c->renormalize = false;
            if (moe) vitna_json_as_bool(vitna_json_get(cfg, "norm_topk_prob"), &c->renormalize);
            ok = true;
        }
    }
    vitna_json_free(doc);
    return ok;
}

/* A tensor of the checkpoint, from whichever of its files holds it. */
static const vitna_tensor_desc_t* find_tensor(const vitna_llama_t* m, const char* name) {
    for (size_t i = 0; i < m->n_shards; i++) {
        const vitna_tensor_desc_t* t = vitna_safetensors_find(&m->shards[i], name);
        if (t) return t;
    }
    return NULL;
}

/* Open the checkpoint file at path as shard i, keeping its path. */
static bool open_shard(vitna_llama_t* m, size_t i, const char* path, char* err, size_t err_len) {
    char st_err[256];
    m->shard_paths[i] = (char*)malloc(strlen(path) + 1);
    if (!m->shard_paths[i]) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
    strcpy(m->shard_paths[i], path);
    if (!vitna_safetensors_open_ex(path, &m->shards[i], st_err, sizeof(st_err))) {
        free(m->shard_paths[i]);
        m->shard_paths[i] = NULL;
        return fail(err, err_len, "%s%s", st_err, NULL);
    }
    m->n_shards = i + 1;
    return true;
}

/* Where a GGUF file's metadata and config.json describe different models,
 * the first difference, else NULL. A value the file leaves out is not
 * compared: the tensors' shapes are checked as they are bound anyway. */
static const char* gguf_disagrees(const vitna_llama_config_t* c, const vitna_gguf_info_t* g) {
    const size_t ffn = g->expert_feed_forward_length ? (size_t)g->expert_feed_forward_length : (size_t)g->feed_forward_length;
    if (g->block_count && g->block_count != c->n_layers) return "the number of layers";
    if (g->embedding_length && g->embedding_length != c->hidden) return "the hidden size";
    if (ffn && ffn != c->intermediate) return "the MLP's width";
    if (g->head_count && g->head_count != c->n_heads) return "the number of attention heads";
    if (g->head_count_kv && g->head_count_kv != c->n_kv_heads) return "the number of key-value heads";
    if (g->key_length && g->key_length != c->head_dim) return "the head size";
    if (g->expert_count != c->n_experts) return "the number of experts";
    if (g->expert_used_count != c->n_experts_used) return "the experts a token goes through";
    if (g->rms_eps != 0.0 && (float)g->rms_eps != c->rms_eps) return "the RMSNorm epsilon";
    if (g->rope_freq_base != 0.0 && (float)g->rope_freq_base != c->rope_theta) return "the rotary base";
    return NULL;
}

/* Open the GGUF file at path as the checkpoint's one file. */
static bool open_gguf(vitna_llama_t* m, const char* path, char* err, size_t err_len) {
    m->shards = (vitna_safetensors_t*)calloc(1, sizeof(vitna_safetensors_t));
    m->shard_paths = (char**)calloc(1, sizeof(char*));
    if (!m->shards || !m->shard_paths) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
    m->shard_paths[0] = (char*)malloc(strlen(path) + 1);
    if (!m->shard_paths[0]) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
    strcpy(m->shard_paths[0], path);
    vitna_gguf_info_t info;
    char g_err[512];
    if (!vitna_gguf_open(path, &m->shards[0], &info, g_err, sizeof(g_err))) {
        free(m->shard_paths[0]);
        m->shard_paths[0] = NULL;
        return fail(err, err_len, "%s%s", g_err, NULL);
    }
    m->n_shards = 1;
    const char* differs = gguf_disagrees(&m->cfg, &info);
    if (differs) return fail(err, err_len, "%s and config.json disagree on %s", path, differs);
    return true;
}

/* Open dir's model.safetensors or, where there is none, every file its
 * model.safetensors.index.json names, and check that each tensor the index
 * lists is found in the file it names. With weights, open that GGUF file
 * instead. */
static bool open_checkpoint(vitna_llama_t* m, const char* dir, const char* weights, char* err, size_t err_len) {
    if (weights) {
        if (!vitna_gguf_path(weights)) return fail(err, err_len, "--weights takes a .gguf file, and %s is not one%s", weights, NULL);
        return open_gguf(m, weights, err, err_len);
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE* single = fopen(path, "rb");
    if (single) {
        fclose(single);
        m->shards = (vitna_safetensors_t*)calloc(1, sizeof(vitna_safetensors_t));
        m->shard_paths = (char**)calloc(1, sizeof(char*));
        if (!m->shards || !m->shard_paths) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
        return open_shard(m, 0, path, err, err_len);
    }

    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    size_t len;
    char* text = vitna_read_file(path, &len);
    if (!text) return fail(err, err_len, "%s holds neither model.safetensors nor model.safetensors.index.json%s", dir, NULL);
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(text, len, jerr, sizeof(jerr));
    free(text);
    if (!doc) return fail(err, err_len, "model.safetensors.index.json: %s%s", jerr, NULL);
    const vitna_json_value_t* map = vitna_json_get(vitna_json_root(doc), "weight_map");
    bool ok = map && map->type == VITNA_JSON_OBJECT && map->u.object.count > 0;
    if (!ok) fail(err, err_len, "model.safetensors.index.json has no weight_map%s%s", NULL, NULL);
    const size_t n = ok ? map->u.object.count : 0;
    /* Each entry's file, as an index into shards; the files in order of first mention. */
    size_t* file_of = ok ? (size_t*)malloc(n * sizeof(size_t)) : NULL;
    const char** names = ok ? (const char**)malloc(n * sizeof(char*)) : NULL;
    if (ok && (!file_of || !names)) ok = fail(err, err_len, "out of memory%s%s", NULL, NULL);
    size_t n_files = 0;
    for (size_t i = 0; ok && i < n; i++) {
        const char* file = vitna_json_as_string(map->u.object.members[i].value);
        /* A shard is a file in the model directory: a name with a path in it would reach outside. */
        if (!file || !*file || strchr(file, '/') || strchr(file, '\\') || strcmp(file, ".") == 0 || strcmp(file, "..") == 0) {
            ok = fail(err, err_len, "model.safetensors.index.json: the file for %s is not a file name in the model directory%s",
                      map->u.object.members[i].key, NULL);
            break;
        }
        size_t f = 0;
        while (f < n_files && strcmp(names[f], file) != 0) f++;
        if (f == n_files) names[n_files++] = file;
        file_of[i] = f;
    }
    if (ok) {
        m->shards = (vitna_safetensors_t*)calloc(n_files, sizeof(vitna_safetensors_t));
        m->shard_paths = (char**)calloc(n_files, sizeof(char*));
        if (!m->shards || !m->shard_paths) ok = fail(err, err_len, "out of memory%s%s", NULL, NULL);
    }
    for (size_t f = 0; ok && f < n_files; f++) {
        snprintf(path, sizeof(path), "%s/%s", dir, names[f]);
        ok = open_shard(m, f, path, err, err_len);
    }
    /* Every tensor listed is in its file, and a lookup by name finds that one. */
    for (size_t i = 0; ok && i < n; i++) {
        const char* name = map->u.object.members[i].key;
        const vitna_tensor_desc_t* t = vitna_safetensors_find(&m->shards[file_of[i]], name);
        if (!t || find_tensor(m, name) != t) {
            ok = fail(err, err_len, "model.safetensors.index.json puts %s in %s, and it is not found there alone", name, names[file_of[i]]);
        }
    }
    free(file_of);
    free(names);
    vitna_json_free(doc);
    return ok;
}

static bool bind(vitna_llama_t* m, vitna_matrix_t* out, const char* name, size_t rows, size_t cols, char* err, size_t err_len) {
    const vitna_tensor_desc_t* t = find_tensor(m, name);
    if (!t) return fail(err, err_len, "the checkpoint has no tensor %s%s", name, NULL);
    bool shape_ok = (cols == 0) ? (t->ndim == 1 && t->shape[0] == rows) : (t->ndim == 2 && t->shape[0] == rows && t->shape[1] == cols);
    if (!shape_ok) return fail(err, err_len, "tensor %s has an unexpected shape%s", name, NULL);
    const bool plain = t->dtype == VITNA_DTYPE_F32 || t->dtype == VITNA_DTYPE_BF16 || t->dtype == VITNA_DTYPE_F16;
    if (cols == 0 && !plain) {
        return fail(err, err_len, "tensor %s is %s; a norm's weights must be F32, BF16 or F16", name, vitna_dtype_name(t->dtype));
    }
    if (!plain && !(vitna_dtype_is_block(t->dtype) && vitna_row_bytes(t->dtype, cols))) {
        return fail(err, err_len, "tensor %s is %s; F32, BF16, F16, Q8_0, Q4_K or Q6_K is needed", name, vitna_dtype_name(t->dtype));
    }
    out->data = t->data_ptr;
    out->dtype = t->dtype;
    out->rows = rows;
    out->cols = cols ? cols : 1;
    return true;
}

/* The n weights of a norm, widened to float32. */
static float* norm_weights(vitna_llama_t* m, const char* name, size_t n, char* err, size_t err_len) {
    vitna_matrix_t w;
    if (!bind(m, &w, name, n, 0, err, err_len)) return NULL;
    float* f = (float*)malloc(n * sizeof(float));
    if (!f) { fail(err, err_len, "out of memory%s%s", NULL, NULL); return NULL; }
    vitna_to_f32(w.data, w.dtype, f, n);
    return f;
}

bool vitna_llama_load(vitna_llama_t* m, const char* dir, size_t ctx, size_t seqs, char* err, size_t err_len) {
    return vitna_llama_load_ex(m, dir, NULL, ctx, seqs, err, err_len);
}

bool vitna_llama_load_ex(vitna_llama_t* m, const char* dir, const char* weights, size_t ctx, size_t seqs, char* err,
                         size_t err_len) {
    memset(m, 0, sizeof(*m));
    m->gpu_kv_layers = -1;
    if (!read_config(&m->cfg, dir, err, err_len)) return false;
    const vitna_llama_config_t* c = &m->cfg;
    if (!open_checkpoint(m, dir, weights, err, err_len)) {
        vitna_llama_free(m);
        return false;
    }

    const size_t q_dim = c->n_heads * c->head_dim;
    const size_t kv_dim = c->n_kv_heads * c->head_dim;
    char name[256];
    bool ok = bind(m, &m->embed, "model.embed_tokens.weight", c->vocab, c->hidden, err, err_len);
    if (ok) {
        if (find_tensor(m, "lm_head.weight")) {
            ok = bind(m, &m->lm_head, "lm_head.weight", c->vocab, c->hidden, err, err_len);
        } else if (c->tied_embeddings) {
            m->lm_head = m->embed;
        } else {
            ok = fail(err, err_len, "no lm_head.weight, and the config does not tie embeddings%s%s", NULL, NULL);
        }
    }
    if (ok) ok = (m->final_norm = norm_weights(m, "model.norm.weight", c->hidden, err, err_len)) != NULL;
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
             LAYER(o, "self_attn.o_proj.weight", c->hidden, q_dim);
        if (ok && c->n_experts) {
            /* OLMoE's names: the router is mlp.gate, each expert mlp.experts.<e>. */
            ok = LAYER(router, "mlp.gate.weight", c->n_experts, c->hidden);
            if (ok) {
                L->experts = (vitna_expert_t*)calloc(c->n_experts, sizeof(vitna_expert_t));
                ok = L->experts != NULL || fail(err, err_len, "out of memory%s%s", NULL, NULL);
            }
            for (size_t e = 0; ok && e < c->n_experts; e++) {
                vitna_expert_t* x = &L->experts[e];
#define EXPERT(field, part, rows, cols) \
                (snprintf(name, sizeof(name), "model.layers.%zu.mlp.experts.%zu.%s", l, e, part), bind(m, &x->field, name, rows, cols, err, err_len))
                ok = EXPERT(gate, "gate_proj.weight", c->intermediate, c->hidden) &&
                     EXPERT(up, "up_proj.weight", c->intermediate, c->hidden) &&
                     EXPERT(down, "down_proj.weight", c->hidden, c->intermediate);
#undef EXPERT
            }
        } else if (ok) {
            ok = LAYER(gate, "mlp.gate_proj.weight", c->intermediate, c->hidden) &&
                 LAYER(up, "mlp.up_proj.weight", c->intermediate, c->hidden) &&
                 LAYER(down, "mlp.down_proj.weight", c->hidden, c->intermediate);
        }
#undef LAYER
        if (ok) {
            snprintf(name, sizeof(name), "model.layers.%zu.input_layernorm.weight", l);
            ok = (L->attn_norm = norm_weights(m, name, c->hidden, err, err_len)) != NULL;
        }
        if (ok) {
            snprintf(name, sizeof(name), "model.layers.%zu.post_attention_layernorm.weight", l);
            ok = (L->mlp_norm = norm_weights(m, name, c->hidden, err, err_len)) != NULL;
        }
        if (ok && c->qk_norm) {
            snprintf(name, sizeof(name), "model.layers.%zu.self_attn.q_norm.weight", l);
            ok = (L->q_norm = norm_weights(m, name, c->qk_norm_per_head ? c->head_dim : q_dim, err, err_len)) != NULL;
            if (ok) {
                snprintf(name, sizeof(name), "model.layers.%zu.self_attn.k_norm.weight", l);
                ok = (L->k_norm = norm_weights(m, name, c->qk_norm_per_head ? c->head_dim : kv_dim, err, err_len)) != NULL;
            }
        }
    }

    if (ok) {
        m->ctx = ctx ? ctx : (c->max_positions < 4096 ? c->max_positions : 4096);
        if (m->ctx > c->max_positions) m->ctx = c->max_positions;
        m->seqs = seqs ? seqs : 1;
        size_t cache = m->seqs * c->n_layers * m->ctx * kv_dim;
        size_t half = c->head_dim / 2;
        m->past = (size_t*)calloc(m->seqs, sizeof(size_t));
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
        if (c->n_experts) {
            m->router_logits = (float*)malloc(c->n_experts * sizeof(float));
            m->expert_out = (float*)malloc(c->hidden * sizeof(float));
        }
        ok = m->past && m->k_cache && m->v_cache && m->inv_freq && m->x && m->xn && m->q && m->k && m->v && m->att &&
             m->proj && m->gate && m->up && m->scores && m->cos_t && m->sin_t &&
             (!c->n_experts || (m->router_logits && m->expert_out));
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
            free(m->layers[l].q_norm);
            free(m->layers[l].k_norm);
            free(m->layers[l].experts);
        }
        free(m->layers);
    }
    free(m->router_logits);
    free(m->expert_out);
    free(m->final_norm);
    free(m->inv_freq);
    free(m->past);
    free(m->k_cache);
    free(m->v_cache);
    free(m->x); free(m->xn); free(m->q); free(m->k); free(m->v); free(m->att);
    free(m->proj); free(m->gate); free(m->up); free(m->scores); free(m->cos_t); free(m->sin_t);
#if defined(VITNA_CUDA)
    vitna_cuda_free(m->cuda);
#endif
    vitna_expert_stream_close(m->stream);
    free(m->pred_xn);
    free(m->pred_logits);
    free(m->rows_logits);
    free(m->rows_next);
    free(m->rows_weights);
    free(m->rows_order);
    free(m->rows_pred);
    free(m->rows_chunk);
    free(m->rows_outs);
    for (size_t i = 0; i < m->n_shards; i++) {
        vitna_safetensors_close(&m->shards[i]);
        free(m->shard_paths[i]);
    }
    free(m->shards);
    free(m->shard_paths);
    memset(m, 0, sizeof(*m));
}

void vitna_llama_reset(vitna_llama_t* m) {
    for (size_t s = 0; s < m->seqs; s++) m->past[s] = 0;
}

void vitna_llama_truncate(vitna_llama_t* m, size_t seq, size_t n) {
    if (seq < m->seqs && n < m->past[seq]) m->past[seq] = n;
}

void vitna_llama_fail_step_once(vitna_llama_t* m, size_t pos, bool lose_device) {
    m->fail_armed = true;
    m->fail_loses = lose_device;
    m->fail_at = pos;
}

/* The step at pos fails, as a test asked. On a GPU, what earlier steps queued
 * runs to its end first, so the failure is reported with nothing of the
 * model's still running there, as after a device error. A test may end the
 * server as soon as it reads the failure: on Windows, a server ended while a
 * chunk of a prompt it had queued still ran made kernels in other CUDA
 * processes fault with an illegal address. */
static void fail_step_as_asked(vitna_llama_t* m, size_t pos) {
    m->fail_armed = false;
#if defined(VITNA_CUDA)
    if (m->cuda) vitna_cuda_wait(m->cuda);
#endif
    if (m->fail_loses) {
        m->device_lost = true;
        snprintf(m->device_error, sizeof(m->device_error), "the step at position %zu failed, and the device with it, as a test asked", pos);
    }
    fprintf(stderr, "The step at position %zu failed%s, as a test asked.\n", pos, m->fail_loses ? ", and the device with it" : "");
}

#if defined(VITNA_CUDA)
/* A step on the GPU failed, for the reason in err: say so, and find whether
 * the device can run anything more in this process (device_lost). */
static void cuda_failed(vitna_llama_t* m, const char* err) {
    fprintf(stderr, "CUDA: %s\n", err);
    m->device_lost = vitna_cuda_lost(m->cuda, m->device_error, sizeof(m->device_error));
}
#endif

void vitna_llama_trace_routing(vitna_llama_t* m, float* logits, int32_t* chosen, size_t positions, const int32_t* pin,
                               size_t pin_positions) {
    m->trace_logits = logits;
    m->trace_chosen = chosen;
    m->trace_positions = (logits || chosen) ? positions : 0;
    m->pin = pin;
    m->pin_positions = pin ? pin_positions : 0;
}

/* Where a matrix of the checkpoint is: the shard whose mapping holds it,
 * and its offset and length in that file. */
static bool locate(const vitna_llama_t* m, const vitna_matrix_t* w, vitna_extent_t* out) {
    const char* p = (const char*)w->data;
    for (size_t i = 0; i < m->n_shards; i++) {
        const char* base = (const char*)m->shards[i].mmap.data;
        if (p >= base && p < base + m->shards[i].mmap.size) {
            out->file = (uint32_t)i;
            out->offset = (uint64_t)(p - base);
            out->length = (uint64_t)w->rows * vitna_row_bytes(w->dtype, w->cols);
            return true;
        }
    }
    return false;
}

/* Reads in flight at once: enough to keep an NVMe drive's queue busy with
 * the experts one layer lacks. */
#define STREAM_THREADS 4

bool vitna_llama_stream_experts(vitna_llama_t* m, size_t cache_bytes, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    if (!c->n_experts) return fail(err, err_len, "this model has no experts to read from the drive%s%s", NULL, NULL);
    if (m->stream) return true;
    /* The GPU copies its experts from wherever they were when it took the model. */
    if (m->cuda) return fail(err, err_len, "the experts must be streamed before the GPU takes the model%s%s", NULL, NULL);
    const size_t E = c->n_experts, K = c->n_experts_used, n = c->n_layers * E;
    vitna_expert_place_t* places = (vitna_expert_place_t*)calloc(n, sizeof(*places));
    if (!places) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
    bool ok = true;
    for (size_t l = 0; ok && l < c->n_layers; l++) {
        for (size_t e = 0; ok && e < E; e++) {
            const vitna_expert_t* x = &m->layers[l].experts[e];
            vitna_expert_place_t* p = &places[l * E + e];
            p->n_parts = 3; /* gate, up, down: mixture_of_experts reads them in this order */
            ok = locate(m, &x->gate, &p->part[0]) && locate(m, &x->up, &p->part[1]) && locate(m, &x->down, &p->part[2]);
        }
    }
    const size_t slot = ok ? vitna_expert_stream_slot_bytes_for(places, n, m->n_shards) : 0;
    size_t slots = slot ? cache_bytes / slot : 0;
    if (slots > n) slots = n; /* a slot for every expert holds them all */
    if (!ok || slot == 0) {
        ok = fail(err, err_len, "an expert is not where the checkpoint's files can be read%s%s", NULL, NULL);
    } else if (slots < 2 * K) {
        char need[32];
        snprintf(need, sizeof(need), "%zu", (2 * K * slot + ((size_t)1 << 20) - 1) >> 20);
        ok = fail(err, err_len, "the expert cache must hold twice the experts a token goes through: %s MiB or more%s", need, NULL);
    }
    if (ok) {
        m->pred_xn = (float*)malloc(c->hidden * sizeof(float));
        m->pred_logits = (float*)malloc(E * sizeof(float));
        ok = (m->pred_xn && m->pred_logits) || fail(err, err_len, "out of memory%s%s", NULL, NULL);
    }
    if (ok) {
        m->predicted_layer = -1;
        m->stream = vitna_expert_stream_open((const char* const*)m->shard_paths, m->n_shards, places, n, slots, STREAM_THREADS, err, err_len);
        ok = m->stream != NULL;
    }
    free(places);
    return ok;
}

bool vitna_llama_read_experts(vitna_llama_t* m, double* mib, double* ms) {
    if (!m->stream) return false;
    const size_t n = m->cfg.n_layers * m->cfg.n_experts, K = m->cfg.n_experts_used;
    uint32_t* order = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!order) return false;
    /* A fixed shuffle, so that no two reads in a row are neighbours in the files. */
    uint64_t r = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i++) order[i] = (uint32_t)i;
    for (size_t i = n - 1; i > 0; i--) {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        const size_t j = (size_t)(r % (i + 1));
        const uint32_t t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    const uint64_t before = vitna_expert_stream_stats(m->stream).bytes_read;
    vitna_expert_data_t data[VITNA_EXPERTS_USED_MAX];
    bool ok = true;
    const double t0 = vitna_time_ms();
    for (size_t i = 0; i < n && ok; i += K) {
        const size_t k = n - i < K ? n - i : K;
        vitna_expert_stream_hold(m->stream, order + i, k);
        if (i + k < n) vitna_expert_stream_prefetch(m->stream, order + i + k, n - i - k < K ? n - i - k : K);
        ok = vitna_expert_stream_wait(m->stream, order + i, k, data);
        if (ok) vitna_expert_stream_release(m->stream, order + i, k);
    }
    *ms = vitna_time_ms() - t0;
    *mib = (double)(vitna_expert_stream_stats(m->stream).bytes_read - before) / 1048576.0;
    free(order);
    return ok;
}

const char* vitna_llama_stream_report(vitna_llama_t* m, char* buf, size_t len) {
    if (len == 0) return buf;
    buf[0] = '\0';
    if (!m->stream) return buf;
    const vitna_expert_stream_stats_t s = vitna_expert_stream_stats(m->stream);
    snprintf(buf, len,
             "experts: %llu acquired: %llu already read, %llu still being read for a prefetch, %llu read when asked for. "
             "%llu prefetched, %llu of those used. %.1f MiB read in %llu reads, %.0f ms of reading summed over %d threads, "
             "%.0f ms waited for. The lookahead named %.1f%% of the experts the layers routed to",
             (unsigned long long)s.acquired, (unsigned long long)s.hits, (unsigned long long)s.in_flight,
             (unsigned long long)s.misses, (unsigned long long)s.prefetched, (unsigned long long)s.prefetch_used,
             (double)s.bytes_read / 1048576.0, (unsigned long long)s.reads, s.read_ms, STREAM_THREADS, s.wait_ms,
             m->predicted_total ? 100.0 * (double)m->predicted_right / (double)m->predicted_total : 0.0);
    return buf;
}

/* The k largest of n logits, largest first, the lower index first between equals. */
static void top_k(const float* logits, size_t n, size_t k, int32_t* out) {
    for (size_t i = 0; i < k; i++) {
        int32_t best = -1;
        for (size_t e = 0; e < n; e++) {
            bool taken = false;
            for (size_t j = 0; j < i && !taken; j++) taken = out[j] == (int32_t)e;
            if (!taken && (best < 0 || logits[e] > logits[best])) best = (int32_t)e;
        }
        out[i] = best;
    }
}

/* A guess at the experts layer will route to, and reads started for those
 * the cache lacks: the residual stream as it stands, normalized with that
 * layer's own norm and scored by its router. The layer still routes for
 * itself when it comes; the guess only decides what is read early. */
static void prefetch_layer(vitna_llama_t* m, size_t layer) {
    const vitna_llama_config_t* c = &m->cfg;
    const vitna_llama_layer_t* L = &m->layers[layer];
    const size_t K = c->n_experts_used;
    vitna_rmsnorm(m->x, L->mlp_norm, m->pred_xn, c->hidden, c->rms_eps);
    vitna_matvec(L->router.data, L->router.dtype, m->pred_xn, m->pred_logits, L->router.rows, L->router.cols);
    top_k(m->pred_logits, c->n_experts, K, m->predicted);
    m->predicted_layer = (int64_t)layer;
    uint32_t places[VITNA_EXPERTS_USED_MAX];
    for (size_t k = 0; k < K; k++) places[k] = (uint32_t)(layer * c->n_experts + (size_t)m->predicted[k]);
    vitna_expert_stream_prefetch(m->stream, places, K);
}

/* Layer l's routing for the token at pos, from the router's logits over
 * every expert, as transformers' OlmoeSparseMoeBlock routes: their softmax,
 * in float32; the experts with the n_experts_used largest weights, which are
 * the largest logits, largest first and the lower expert first between
 * equals, into chosen; and each weighted by its own share of the softmax,
 * or with renormalize (norm_topk_prob, as Qwen3MoeSparseMoeBlock routes)
 * by that share over the shares of the experts used, added largest first as
 * torch.topk returns them. The experts the token goes through are
 * written to order, in order of expert, with their weights in weights: the
 * router's choice, or where a test pins them the pinned experts. The
 * router's logits and choice are traced for tests. The CPU's
 * mixture_of_experts and the GPU's step both route here. */
static void route(vitna_llama_t* m, size_t l, size_t pos, const float* logits, int32_t* chosen, int32_t* order, float* weights) {
    const vitna_llama_config_t* c = &m->cfg;
    const size_t E = c->n_experts, K = c->n_experts_used;
    top_k(logits, E, K, chosen);
    const size_t at = pos * c->n_layers + l;
    if (pos < m->trace_positions) {
        if (m->trace_logits) memcpy(m->trace_logits + at * E, logits, E * sizeof(float));
        if (m->trace_chosen) memcpy(m->trace_chosen + at * K, chosen, K * sizeof(int32_t));
    }
    memcpy(order, pos < m->pin_positions ? m->pin + at * K : chosen, K * sizeof(int32_t));
    for (size_t i = 1; i < K; i++) {
        const int32_t e = order[i];
        size_t j = i;
        for (; j > 0 && order[j - 1] > e; j--) order[j] = order[j - 1];
        order[j] = e;
    }
    float max = logits[0];
    for (size_t e = 1; e < E; e++) if (logits[e] > max) max = logits[e];
    float sum = 0.0f;
    for (size_t e = 0; e < E; e++) sum += expf(logits[e] - max);
    for (size_t k = 0; k < K; k++) weights[k] = expf(logits[order[k]] - max) / sum;
    if (c->renormalize) {
        /* The shares of the experts used, added largest first, as torch.topk
         * orders the weights that Qwen3MoeSparseMoeBlock sums. */
        float share[VITNA_EXPERTS_USED_MAX];
        memcpy(share, weights, K * sizeof(float));
        for (size_t i = 1; i < K; i++) {
            const float s = share[i];
            size_t j = i;
            for (; j > 0 && share[j - 1] < s; j--) share[j] = share[j - 1];
            share[j] = s;
        }
        float total = 0.0f;
        for (size_t k = 0; k < K; k++) total += share[k];
        for (size_t k = 0; k < K; k++) weights[k] /= total;
    }
}

/* How many of layer l's chosen experts the lookahead had named, when it
 * guessed for this layer. */
static void count_lookahead(vitna_llama_t* m, size_t l, const int32_t* chosen) {
    const size_t K = m->cfg.n_experts_used;
    if (m->predicted_layer != (int64_t)l) return;
    for (size_t k = 0; k < K; k++) {
        for (size_t j = 0; j < K; j++) m->predicted_right += m->predicted[j] == chosen[k];
    }
    m->predicted_total += K;
}

/* Layer l's mixture of experts for the token at pos: m->xn in, m->proj out.
 * This is transformers' OlmoeSparseMoeBlock with its eager experts: the
 * router's logits over every expert, routed as route() says, and the
 * outputs of the experts it names, each scaled by its weight, added to
 * zeros one expert at a time in order of expert, as the loop over experts
 * and index_add_ add them. With the experts streamed, the same arithmetic
 * runs on the bytes read from the drive; false if a read fails. */
static bool mixture_of_experts(vitna_llama_t* m, const vitna_llama_layer_t* L, size_t l, size_t pos) {
    const vitna_llama_config_t* c = &m->cfg;
    const size_t E = c->n_experts, K = c->n_experts_used, H = c->hidden;
    float* logits = m->router_logits;
    vitna_matvec(L->router.data, L->router.dtype, m->xn, logits, L->router.rows, L->router.cols);
    int32_t chosen[VITNA_EXPERTS_USED_MAX], order[VITNA_EXPERTS_USED_MAX];
    float weights[VITNA_EXPERTS_USED_MAX];
    route(m, l, pos, logits, chosen, order, weights);

    /* Streamed: hold this layer's experts, reading those the cache lacks,
     * then start reading what the next layer is guessed to want while
     * these are read and run. Held experts cannot be given up for the guess. */
    uint32_t places[VITNA_EXPERTS_USED_MAX];
    vitna_expert_data_t data[VITNA_EXPERTS_USED_MAX];
    if (m->stream) {
        for (size_t k = 0; k < K; k++) places[k] = (uint32_t)(l * E + (size_t)order[k]);
        vitna_expert_stream_hold(m->stream, places, K);
        count_lookahead(m, l, chosen);
        if (l + 1 < c->n_layers) prefetch_layer(m, l + 1);
        if (!vitna_expert_stream_wait(m->stream, places, K, data)) return false;
    }

    memset(m->proj, 0, H * sizeof(float));
    for (size_t k = 0; k < K; k++) {
        const vitna_expert_t* x = &L->experts[order[k]];
        /* The streamed parts are gate, up and down, as vitna_llama_stream_experts placed them. */
        const void* gate = m->stream ? data[k].part[0] : x->gate.data;
        const void* up = m->stream ? data[k].part[1] : x->up.data;
        const void* down = m->stream ? data[k].part[2] : x->down.data;
        const float w = weights[k];
        vitna_matvec(gate, x->gate.dtype, m->xn, m->gate, x->gate.rows, x->gate.cols);
        vitna_matvec(up, x->up.dtype, m->xn, m->up, x->up.rows, x->up.cols);
        vitna_silu_mul(m->gate, m->up, m->gate, c->intermediate);
        vitna_matvec(down, x->down.dtype, m->gate, m->expert_out, x->down.rows, x->down.cols);
        for (size_t i = 0; i < H; i++) {
            const float scaled = m->expert_out[i] * w;
            m->proj[i] += scaled;
        }
    }
    if (m->stream) vitna_expert_stream_release(m->stream, places, K);
    return true;
}

#if defined(VITNA_CUDA)
/* A step of a mixture of experts on the GPU, a layer at a time: the device
 * runs each layer up to its router, this routes as mixture_of_experts does
 * on the CPU, and the device runs the experts named. With the router's
 * logits come the next layer's router's on the same residual, the guess the
 * CPU's lookahead makes, and the device starts copying the experts it names. */
static bool moe_step_on_gpu(vitna_llama_t* m, size_t seq, int32_t token, float* logits) {
    const vitna_llama_config_t* c = &m->cfg;
    const size_t pos = m->past[seq], K = c->n_experts_used;
    char err[512];
    bool ok = true;
    for (size_t l = 0; ok && l < c->n_layers; l++) {
        const bool guess = l + 1 < c->n_layers;
        ok = vitna_cuda_moe_route(m->cuda, seq, token, pos, l, m->router_logits, guess ? m->pred_logits : NULL, err, sizeof(err));
        if (!ok) break;
        int32_t chosen[VITNA_EXPERTS_USED_MAX], order[VITNA_EXPERTS_USED_MAX];
        float weights[VITNA_EXPERTS_USED_MAX];
        route(m, l, pos, m->router_logits, chosen, order, weights);
        count_lookahead(m, l, chosen);
        if (guess) {
            top_k(m->pred_logits, c->n_experts, K, m->predicted);
            m->predicted_layer = (int64_t)(l + 1);
        }
        ok = vitna_cuda_moe_experts(m->cuda, l, order, weights, K, guess ? m->predicted : NULL, err, sizeof(err));
    }
    if (ok && logits) ok = vitna_cuda_moe_head(m->cuda, logits, err, sizeof(err));
    if (!ok) {
        /* As a dense step's failure is: said, and the device probed. An
         * expert the drive failed to give leaves the device usable, and the
         * probe says so. */
        cuda_failed(m, err);
        return false;
    }
    m->past[seq] = pos + 1;
    return true;
}

/* n rows of a mixture of experts on the GPU, each a token at its position in
 * its sequence, a layer at a time: the device runs every row up to the
 * layer's router, each row routes here as moe_step_on_gpu routes one, and
 * the device runs each expert once for all the rows routed to it. Every
 * value is the one n steps would give. logits[i], where logits and it are not
 * NULL, receives row i's. Advances no sequence: the caller does, once this
 * has returned true; on false err says why. */
static bool moe_rows_on_gpu(vitna_llama_t* m, const vitna_cuda_row_t* rows, size_t n, float* const* logits, char* err, size_t err_len) {
    const vitna_llama_config_t* c = &m->cfg;
    const size_t E = c->n_experts, K = c->n_experts_used;
    bool ok = true;
    for (size_t l = 0; ok && l < c->n_layers; l++) {
        const bool guess = l + 1 < c->n_layers;
        ok = vitna_cuda_moe_rows_route(m->cuda, rows, n, l, m->rows_logits, guess ? m->rows_next : NULL, err, err_len);
        if (!ok) break;
        for (size_t t = 0; t < n; t++) {
            int32_t chosen[VITNA_EXPERTS_USED_MAX];
            route(m, l, (size_t)rows[t].pos, m->rows_logits + t * E, chosen, m->rows_order + t * K, m->rows_weights + t * K);
            if (l > 0) { /* the row's guess at this layer, made at the one before */
                for (size_t a = 0; a < K; a++) {
                    for (size_t b = 0; b < K; b++) m->predicted_right += m->rows_pred[t * K + b] == chosen[a];
                }
                m->predicted_total += K;
            }
            if (guess) top_k(m->rows_next + t * E, E, K, m->rows_pred + t * K);
        }
        ok = vitna_cuda_moe_rows_experts(m->cuda, l, n, m->rows_order, m->rows_weights, K, guess ? m->rows_pred : NULL, err, err_len);
    }
    if (ok && logits) ok = vitna_cuda_moe_rows_head(m->cuda, n, logits, err, err_len);
    return ok;
}
#endif

void vitna_llama_no_rows(vitna_llama_t* m) {
    m->no_rows = true;
}

void vitna_llama_test_long_from(vitna_llama_t* m, size_t n) {
#if defined(VITNA_CUDA)
    if (m->cuda) vitna_cuda_test_long_from(m->cuda, n);
#else
    (void)m;
    (void)n;
#endif
}

size_t vitna_llama_gpu_kv_layers(const vitna_llama_t* m) {
#if defined(VITNA_CUDA)
    if (m->cuda) return (size_t)vitna_cuda_kv_layers(m->cuda);
#endif
    (void)m;
    return 0;
}

bool vitna_llama_step(vitna_llama_t* m, size_t seq, int32_t token, float* logits) {
    const vitna_llama_config_t* c = &m->cfg;
    if (seq >= m->seqs || token < 0 || (size_t)token >= c->vocab || m->past[seq] >= m->ctx || m->device_lost) return false;
    if (m->fail_armed && m->past[seq] == m->fail_at) {
        fail_step_as_asked(m, m->fail_at);
        return false;
    }

#if defined(VITNA_CUDA)
    if (m->cuda && c->n_experts) return moe_step_on_gpu(m, seq, token, logits);
    if (m->cuda) {
        char err[512];
        if (!vitna_cuda_step(m->cuda, seq, token, m->past[seq], logits, err, sizeof(err))) {
            cuda_failed(m, err);
            return false;
        }
        m->past[seq]++;
        return true;
    }
#endif

    const size_t pos = m->past[seq];
    const size_t H = c->hidden, hd = c->head_dim, half = hd / 2;
    const size_t kv_dim = c->n_kv_heads * hd;
    const size_t group = c->n_heads / c->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);

    /* The token's embedding row, widened to float32. */
    const size_t row_bytes = (size_t)vitna_row_bytes(m->embed.dtype, H);
    vitna_to_f32((const char*)m->embed.data + (size_t)token * row_bytes, m->embed.dtype, m->x, H);
    /* Streamed experts: the first layer's guess, read while its attention runs. */
    if (m->stream) prefetch_layer(m, 0);

    /* Rotary angles for this position: position * inv_freq, in float32. */
    for (size_t j = 0; j < half; j++) {
        float angle = (float)pos * m->inv_freq[j];
        m->cos_t[j] = cosf(angle);
        m->sin_t[j] = sinf(angle);
    }

    for (size_t l = 0; l < c->n_layers; l++) {
        const vitna_llama_layer_t* L = &m->layers[l];
        float* kc = m->k_cache + (seq * c->n_layers + l) * m->ctx * kv_dim;
        float* vc = m->v_cache + (seq * c->n_layers + l) * m->ctx * kv_dim;

        /* Attention */
        vitna_rmsnorm(m->x, L->attn_norm, m->xn, H, c->rms_eps);
        vitna_matvec(L->q.data, L->q.dtype, m->xn, m->q, L->q.rows, L->q.cols);
        vitna_matvec(L->k.data, L->k.dtype, m->xn, m->k, L->k.rows, L->k.cols);
        vitna_matvec(L->v.data, L->v.dtype, m->xn, m->v, L->v.rows, L->v.cols);
        if (c->qk_norm && c->qk_norm_per_head) {
            /* Qwen3: each head of q and k normalized alone, every head with
             * the same weights, as Qwen3MoeAttention normalizes q and k
             * viewed as [heads][head_dim]. */
            for (size_t h = 0; h < c->n_heads; h++) vitna_rmsnorm(m->q + h * hd, L->q_norm, m->q + h * hd, hd, c->rms_eps);
            for (size_t h = 0; h < c->n_kv_heads; h++) vitna_rmsnorm(m->k + h * hd, L->k_norm, m->k + h * hd, hd, c->rms_eps);
        } else if (c->qk_norm) {
            /* OLMoE: q and k each normalized whole, across their heads. */
            vitna_rmsnorm(m->q, L->q_norm, m->q, L->q.rows, c->rms_eps);
            vitna_rmsnorm(m->k, L->k_norm, m->k, L->k.rows, c->rms_eps);
        }
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
        if (c->n_experts) {
            if (!mixture_of_experts(m, L, l, pos)) return false;
        } else {
            vitna_matvec(L->gate.data, L->gate.dtype, m->xn, m->gate, L->gate.rows, L->gate.cols);
            vitna_matvec(L->up.data, L->up.dtype, m->xn, m->up, L->up.rows, L->up.cols);
            vitna_silu_mul(m->gate, m->up, m->gate, c->intermediate);
            vitna_matvec(L->down.data, L->down.dtype, m->gate, m->proj, L->down.rows, L->down.cols);
        }
        for (size_t i = 0; i < H; i++) m->x[i] += m->proj[i];
    }

    m->past[seq] = pos + 1;
    if (logits) {
        vitna_rmsnorm(m->x, m->final_norm, m->xn, H, c->rms_eps);
        vitna_matvec(m->lm_head.data, m->lm_head.dtype, m->xn, logits, m->lm_head.rows, m->lm_head.cols);
    }
    return true;
}

size_t vitna_llama_steps(vitna_llama_t* m, size_t seq, const int32_t* tokens, size_t count, float* logits, size_t rows) {
    const vitna_llama_config_t* c = &m->cfg;
    if (seq >= m->seqs || m->device_lost) return 0;
    const size_t start = m->past[seq];
    if (rows > count) rows = count;
    /* The tokens that can run: those before the first that is out of range,
     * past the cache or failed by a test. */
    size_t run = 0;
    while (run < count && tokens[run] >= 0 && (size_t)tokens[run] < c->vocab && start + run < m->ctx &&
           !(m->fail_armed && start + run == m->fail_at)) {
        run++;
    }
    /* Logits come only from a batch that runs to its end. */
    float* out = run == count ? logits : NULL;
    const size_t first = count - rows; /* the first position whose logits are wanted */

#if defined(VITNA_CUDA)
    if (m->cuda && c->n_experts && !m->no_rows && run >= 2) {
        /* A mixture of experts: the tokens as rows of one sequence, a chunk
         * of up to rows_max at a time, each chunk confirmed before the next. */
        char err[512];
        vitna_cuda_row_t* chunk = m->rows_chunk;
        float** outs = m->rows_outs;
        for (size_t done = 0; done < run;) {
            const size_t n = run - done < m->rows_max ? run - done : m->rows_max;
            for (size_t i = 0; i < n; i++) {
                chunk[i].token = tokens[done + i];
                chunk[i].pos = (int32_t)(start + done + i);
                chunk[i].seq = (int32_t)seq;
                outs[i] = (out && done + i >= first) ? out + (done + i - first) * c->vocab : NULL;
            }
            if (!moe_rows_on_gpu(m, chunk, n, outs, err, sizeof(err))) {
                cuda_failed(m, err);
                m->past[seq] = start + done;
                return done;
            }
            done += n;
            m->past[seq] = start + done;
        }
    } else if (m->cuda && run >= vitna_cuda_prompt_min(m->cuda)) {
        char err[512];
        if (!vitna_cuda_steps(m->cuda, seq, tokens, run, start, out, out ? rows : 0, err, sizeof(err))) {
            cuda_failed(m, err);
            return 0;
        }
        m->past[seq] = start + run;
    } else
#endif
    {
        for (size_t i = 0; i < run; i++) {
            float* row = (out && i >= first) ? out + (i - first) * c->vocab : NULL;
            if (!vitna_llama_step(m, seq, tokens[i], row)) return i;
        }
    }
    /* A token a test failed stops the batch as a step there would, with the
     * same word on stderr: vitna_llama_step gives it, and fails. */
    if (run < count && m->fail_armed && start + run == m->fail_at) (void)vitna_llama_step(m, seq, tokens[run], NULL);
    return run;
}

size_t vitna_llama_prompt_piece_min(const vitna_llama_t* m) {
#if defined(VITNA_CUDA)
    if (m->cuda) return vitna_cuda_prompt_piece_min(m->cuda);
#endif
    (void)m;
    return 1;
}

/* The most rows a pass of vitna_llama_step_rows takes on a GPU. A prompt's
 * chunks for a mixture of experts (vitna_llama_steps) are rows_max. */
#define PASS_MAX 64

size_t vitna_llama_exact_max(const vitna_llama_t* m) {
#if defined(VITNA_CUDA)
    if (m->cuda && m->cfg.n_experts) return m->no_rows || m->rows_max < 2 ? 1 : (m->rows_max < PASS_MAX ? m->rows_max : PASS_MAX);
    if (m->cuda) {
        const size_t k = vitna_cuda_exact_max(m->cuda);
        return k > 1 ? k : 1;
    }
#endif
    (void)m;
    return 1;
}

/* Whether seq is one of the n in list. */
static bool listed(const size_t* list, size_t n, size_t seq) {
    for (size_t i = 0; i < n; i++) {
        if (list[i] == seq) return true;
    }
    return false;
}

#if defined(VITNA_CUDA)

/* One GPU pass over k rows, which are rows at[0..k-1] of the caller's. A
 * pass of one row runs as a step, but a mixture of experts runs every pass
 * as rows (moe_rows_on_gpu). On a device error every row fails, and their
 * sequences stop. */
static size_t gpu_pass(vitna_llama_t* m, const vitna_cuda_row_t* pass, const size_t* at, size_t k, float* logits, bool* ran,
                       size_t* stopped, size_t* n_stopped) {
    const size_t V = m->cfg.vocab;
    char err[512];
    bool ok;
    if (m->cfg.n_experts) {
        float* out[PASS_MAX];
        for (size_t j = 0; j < k; j++) out[j] = logits ? logits + at[j] * V : NULL;
        ok = moe_rows_on_gpu(m, pass, k, logits ? out : NULL, err, sizeof(err));
    } else if (k == 1) {
        ok = vitna_cuda_step(m->cuda, (size_t)pass[0].seq, pass[0].token, (size_t)pass[0].pos, logits ? logits + at[0] * V : NULL, err,
                             sizeof(err));
    } else {
        float* out[PASS_MAX];
        for (size_t j = 0; j < k; j++) out[j] = logits ? logits + at[j] * V : NULL;
        ok = vitna_cuda_rows(m->cuda, pass, k, out, err, sizeof(err));
    }
    if (!ok) cuda_failed(m, err);
    for (size_t j = 0; j < k; j++) {
        const size_t seq = (size_t)pass[j].seq;
        if (ok) {
            ran[at[j]] = true;
            m->past[seq]++;
        } else if (!listed(stopped, *n_stopped, seq)) {
            stopped[(*n_stopped)++] = seq;
        }
    }
    return ok ? k : 0;
}

/* The rows in passes of up to vitna_cuda_exact_max, in order. A row that
 * cannot run (out of range, past its sequence's cache, failed by a test)
 * fails before its pass, as its step would, and the rows ahead of it run.
 * Once the device is lost, no pass runs, the one being filled included. */
static size_t rows_on_gpu(vitna_llama_t* m, const vitna_llama_row_t* rows, size_t n, float* logits, bool* ran, size_t* stopped) {
    const vitna_llama_config_t* c = &m->cfg;
    size_t cap = vitna_llama_exact_max(m);
    if (cap > PASS_MAX) cap = PASS_MAX;
    vitna_cuda_row_t pass[PASS_MAX];
    size_t at[PASS_MAX];
    size_t k = 0, done = 0, n_stopped = 0;
    for (size_t i = 0; i < n && !m->device_lost; i++) {
        const size_t seq = rows[i].seq;
        if (listed(stopped, n_stopped, seq)) continue;
        /* The sequence's next position, after its rows already in this pass. */
        size_t pos = seq < m->seqs ? m->past[seq] : 0;
        for (size_t j = 0; j < k; j++) pos += (size_t)pass[j].seq == seq;
        const int32_t t = rows[i].token;
        if (seq >= m->seqs || t < 0 || (size_t)t >= c->vocab || pos >= m->ctx || (m->fail_armed && pos == m->fail_at)) {
            if (seq < m->seqs && m->fail_armed && pos == m->fail_at && t >= 0 && (size_t)t < c->vocab && pos < m->ctx) {
                fail_step_as_asked(m, pos);
            }
            stopped[n_stopped++] = seq;
            continue;
        }
        pass[k].token = t;
        pass[k].pos = (int32_t)pos;
        pass[k].seq = (int32_t)seq;
        at[k++] = i;
        if (k == cap) {
            done += gpu_pass(m, pass, at, k, logits, ran, stopped, &n_stopped);
            k = 0;
        }
    }
    if (k > 0 && !m->device_lost) done += gpu_pass(m, pass, at, k, logits, ran, stopped, &n_stopped);
    return done;
}
#endif

size_t vitna_llama_step_rows(vitna_llama_t* m, const vitna_llama_row_t* rows, size_t n, float* logits, bool* ran) {
    for (size_t i = 0; i < n; i++) ran[i] = false;
    /* The sequences stopped by a row that failed. */
    size_t few[16];
    size_t* stopped = n <= 16 ? few : (size_t*)malloc(n * sizeof(size_t));
    if (!stopped) return 0;
    size_t done = 0;
#if defined(VITNA_CUDA)
    if (m->cuda && vitna_llama_exact_max(m) >= 2) {
        done = rows_on_gpu(m, rows, n, logits, ran, stopped);
        if (stopped != few) free(stopped);
        return done;
    }
#endif
    size_t n_stopped = 0;
    for (size_t i = 0; i < n; i++) {
        if (listed(stopped, n_stopped, rows[i].seq)) continue;
        ran[i] = vitna_llama_step(m, rows[i].seq, rows[i].token, logits ? logits + i * m->cfg.vocab : NULL);
        if (ran[i]) done++;
        else stopped[n_stopped++] = rows[i].seq;
    }
    if (stopped != few) free(stopped);
    return done;
}

size_t vitna_llama_steps_exact(vitna_llama_t* m, size_t seq, const int32_t* tokens, size_t count, float* logits) {
    vitna_llama_row_t few[16];
    bool few_ran[16];
    vitna_llama_row_t* rows = count <= 16 ? few : (vitna_llama_row_t*)malloc(count * sizeof(vitna_llama_row_t));
    bool* ran = count <= 16 ? few_ran : (bool*)malloc(count * sizeof(bool));
    size_t done = 0;
    if (rows && ran) {
        for (size_t i = 0; i < count; i++) {
            rows[i].seq = seq;
            rows[i].token = tokens[i];
        }
        done = vitna_llama_step_rows(m, rows, count, logits, ran);
    }
    if (rows != few) free(rows);
    if (ran != few_ran) free(ran);
    return done;
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

bool vitna_llama_use_cuda(vitna_llama_t* m, size_t expert_cache_bytes, char* err, size_t err_len) {
#if defined(VITNA_CUDA)
    if (m->cuda) return true;
    if (m->cfg.n_experts && !m->pred_logits) {
        /* The lookahead's logits, which the GPU computes with the router's. */
        m->pred_logits = (float*)malloc(m->cfg.n_experts * sizeof(float));
        if (!m->pred_logits) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
        m->predicted_layer = -1;
    }
    float *ct = NULL, *st = NULL;
    if (!rope_tables(m, &ct, &st)) return fail(err, err_len, "out of memory%s%s", NULL, NULL);
    m->cuda = vitna_cuda_create(m, ct, st, expert_cache_bytes, err, err_len);
    free(ct);
    free(st);
    if (!m->cuda) return false;
    if (m->cfg.n_experts) {
        /* Room to route a pass of rows. */
        const size_t R = vitna_cuda_moe_rows_max(m->cuda), E = m->cfg.n_experts, K = m->cfg.n_experts_used;
        m->rows_logits = (float*)malloc(R * E * sizeof(float));
        m->rows_next = (float*)malloc(R * E * sizeof(float));
        m->rows_weights = (float*)malloc(R * K * sizeof(float));
        m->rows_order = (int32_t*)malloc(R * K * sizeof(int32_t));
        m->rows_pred = (int32_t*)malloc(R * K * sizeof(int32_t));
        m->rows_chunk = (vitna_cuda_row_t*)malloc(R * sizeof(vitna_cuda_row_t));
        m->rows_outs = (float**)malloc(R * sizeof(float*));
        if (!m->rows_logits || !m->rows_next || !m->rows_weights || !m->rows_order || !m->rows_pred || !m->rows_chunk || !m->rows_outs) {
            vitna_cuda_free(m->cuda);
            m->cuda = NULL;
            return fail(err, err_len, "out of memory%s%s", NULL, NULL);
        }
        m->rows_max = R;
    }
    /* The device holds the key-value cache from here on. */
    free(m->k_cache);
    free(m->v_cache);
    m->k_cache = NULL;
    m->v_cache = NULL;
    return true;
#else
    (void)m;
    (void)expert_cache_bytes;
    return fail(err, err_len, "%s%s", NO_CUDA, NULL);
#endif
}

bool vitna_llama_widen_on_gpu(const vitna_matrix_t* w, float* out, char* err, size_t err_len) {
#if defined(VITNA_CUDA)
    return vitna_cuda_widen(w, out, err, err_len);
#else
    (void)w;
    (void)out;
    return fail(err, err_len, "%s%s", NO_CUDA, NULL);
#endif
}

bool vitna_llama_cpu_experts(vitna_llama_t* m, size_t threads, char* err, size_t err_len) {
#if defined(VITNA_CUDA)
    if (!m->cuda) return fail(err, err_len, "the experts run on the CPU beside the GPU, and this model is not on the GPU%s%s", NULL, NULL);
    return vitna_cuda_cpu_experts(m->cuda, threads, err, err_len);
#else
    (void)m;
    (void)threads;
    return fail(err, err_len, "%s%s", NO_CUDA, NULL);
#endif
}

bool vitna_llama_expert_on_gpu(const vitna_llama_t* m, size_t l, size_t e, const float* x, float* xs, float* act, float* y, char* err,
                               size_t err_len) {
    if (!m->cfg.n_experts || l >= m->cfg.n_layers || e >= m->cfg.n_experts) {
        return fail(err, err_len, "there is no such expert in this model%s%s", NULL, NULL);
    }
#if defined(VITNA_CUDA)
    const vitna_llama_layer_t* L = &m->layers[l];
    const vitna_expert_t* x0 = &L->experts[e];
    return vitna_cuda_expert_check(&x0->gate, &x0->up, &x0->down, L->mlp_norm, x, m->cfg.rms_eps, xs, act, y, err, err_len);
#else
    (void)x;
    (void)xs;
    (void)act;
    (void)y;
    return fail(err, err_len, "%s%s", NO_CUDA, NULL);
#endif
}

const char* vitna_llama_gpu_report(vitna_llama_t* m, char* buf, size_t len) {
    if (len == 0) return buf;
    buf[0] = '\0';
#if defined(VITNA_CUDA)
    if (m->cuda && m->cfg.n_experts) {
        vitna_cuda_moe_report(m->cuda, buf, len);
        const size_t at = strlen(buf);
        snprintf(buf + at, len - at, ". The lookahead named %.1f%% of the experts the layers routed to",
                 m->predicted_total ? 100.0 * (double)m->predicted_right / (double)m->predicted_total : 0.0);
    }
#else
    (void)m;
#endif
    return buf;
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
