/**
 * gguf.c - A GGUF file read as a checkpoint (gguf.h).
 *
 * The layout, little-endian throughout: "GGUF", a version, the number of
 * tensors and of metadata entries; each entry a key, a type and a value;
 * each tensor a name, its dimensions (the fastest first), a ggml type and
 * an offset; then, at the next multiple of the file's alignment, the data
 * section the offsets count from. Every length is checked against what is
 * left of the file before it is used.
 */

#include "gguf.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" */
#define DEFAULT_ALIGNMENT 32
#define NAME_MAX_GGUF 255

/* Metadata value types. */
enum { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

/* ggml's tensor types the engine computes with; any other is refused, by its name where known. */
static vitna_dtype_t ggml_dtype(uint32_t t) {
    switch (t) {
        case 0: return VITNA_DTYPE_F32;
        case 1: return VITNA_DTYPE_F16;
        case 8: return VITNA_DTYPE_Q8_0;
        case 12: return VITNA_DTYPE_Q4_K;
        case 14: return VITNA_DTYPE_Q6_K;
        case 30: return VITNA_DTYPE_BF16;
        default: return VITNA_DTYPE_UNKNOWN;
    }
}

static const char* ggml_type_name(uint32_t t) {
    static const char* names[] = {
        "F32", "F16", "Q4_0", "Q4_1", "type 4", "type 5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K",
        "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
        "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16",
    };
    return t < sizeof(names) / sizeof(names[0]) ? names[t] : "an unknown type";
}

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    bool ok;
} cursor_t;

static bool need(cursor_t* c, uint64_t n) {
    if (!c->ok || (uint64_t)(c->end - c->p) < n) c->ok = false;
    return c->ok;
}

static uint64_t rd_le(cursor_t* c, int bytes) {
    if (!need(c, (uint64_t)bytes)) return 0;
    uint64_t v = 0;
    for (int i = bytes - 1; i >= 0; i--) v = (v << 8) | c->p[i];
    c->p += bytes;
    return v;
}

static bool rd_str(cursor_t* c, const char** s, uint64_t* len) {
    uint64_t n = rd_le(c, 8);
    if (!need(c, n)) return false;
    *s = (const char*)c->p;
    *len = n;
    c->p += n;
    return true;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default: return 0;
    }
}

/* A scalar of type t as a double. */
static double rd_number(cursor_t* c, uint32_t t) {
    const size_t n = scalar_size(t);
    uint64_t u = rd_le(c, (int)n);
    switch (t) {
        case T_I8: return (double)(int8_t)u;
        case T_I16: return (double)(int16_t)u;
        case T_I32: return (double)(int32_t)u;
        case T_I64: return (double)(int64_t)u;
        case T_F32: { uint32_t b = (uint32_t)u; float f; memcpy(&f, &b, 4); return (double)f; }
        case T_F64: { double d; memcpy(&d, &u, 8); return d; }
        default: return (double)u;
    }
}

/* Step over a value of type t, which may be an array (of arrays, to a depth). */
static bool skip_value(cursor_t* c, uint32_t t, int depth) {
    if (scalar_size(t)) return need(c, scalar_size(t)) && (c->p += scalar_size(t), true);
    if (t == T_STR) {
        const char* s;
        uint64_t n;
        return rd_str(c, &s, &n);
    }
    if (t == T_ARR && depth < 4) {
        const uint32_t et = (uint32_t)rd_le(c, 4);
        const uint64_t count = rd_le(c, 8);
        if (!c->ok) return false;
        if (scalar_size(et)) {
            if (count > (uint64_t)(c->end - c->p) / scalar_size(et)) return c->ok = false;
            c->p += count * scalar_size(et);
            return true;
        }
        for (uint64_t i = 0; i < count && c->ok; i++) skip_value(c, et, depth + 1);
        return c->ok;
    }
    return c->ok = false;
}

/* A numeric metadata entry, kept by its key. */
typedef struct {
    const char* key;
    size_t len;
    double value;
} number_t;

static bool key_is(const number_t* n, const char* arch, const char* suffix) {
    const size_t a = strlen(arch), s = strlen(suffix);
    return n->len == a + 1 + s && memcmp(n->key, arch, a) == 0 && n->key[a] == '.' && memcmp(n->key + a + 1, suffix, s) == 0;
}

static uint64_t meta_u64(const number_t* nums, size_t n, const char* arch, const char* suffix) {
    for (size_t i = 0; i < n; i++) {
        if (key_is(&nums[i], arch, suffix) && nums[i].value >= 0) return (uint64_t)nums[i].value;
    }
    return 0;
}

static double meta_f64(const number_t* nums, size_t n, const char* arch, const char* suffix) {
    for (size_t i = 0; i < n; i++) if (key_is(&nums[i], arch, suffix)) return nums[i].value;
    return 0.0;
}

/* One tensor as the file lists it. */
typedef struct {
    char name[NAME_MAX_GGUF + 1];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
} file_tensor_t;

/* A layer tensor's part of a llama.cpp name, and the Hugging Face name it
   takes; experts marks the stacks that become one tensor per expert. */
static const struct { const char* part; const char* hf; bool experts; } LAYER_PARTS[] = {
    { "attn_norm", "input_layernorm", false },
    { "attn_q", "self_attn.q_proj", false },
    { "attn_k", "self_attn.k_proj", false },
    { "attn_v", "self_attn.v_proj", false },
    { "attn_output", "self_attn.o_proj", false },
    { "attn_q_norm", "self_attn.q_norm", false },
    { "attn_k_norm", "self_attn.k_norm", false },
    { "ffn_norm", "post_attention_layernorm", false },
    { "ffn_gate_inp", "mlp.gate", false },
    { "ffn_gate", "mlp.gate_proj", false },
    { "ffn_up", "mlp.up_proj", false },
    { "ffn_down", "mlp.down_proj", false },
    { "ffn_gate_exps", "gate_proj", true },
    { "ffn_up_exps", "up_proj", true },
    { "ffn_down_exps", "down_proj", true },
};

static const struct { const char* gguf; const char* hf; } MODEL_PARTS[] = {
    { "token_embd.weight", "model.embed_tokens.weight" },
    { "output_norm.weight", "model.norm.weight" },
    { "output.weight", "lm_head.weight" },
};

/* The Hugging Face name of a tensor the file names, or of the stack's
   expert e. Returns false for a name this does not know. layer and
   experts are set for a layer's tensor. */
static bool hf_name(const char* name, char* out, size_t out_len, bool* experts, size_t* layer, const char** hf_part) {
    *experts = false;
    for (size_t i = 0; i < sizeof(MODEL_PARTS) / sizeof(MODEL_PARTS[0]); i++) {
        if (strcmp(name, MODEL_PARTS[i].gguf) == 0) {
            snprintf(out, out_len, "%s", MODEL_PARTS[i].hf);
            return true;
        }
    }
    if (strncmp(name, "blk.", 4) != 0 || !isdigit((unsigned char)name[4])) return false;
    char* end;
    const unsigned long l = strtoul(name + 4, &end, 10);
    if (*end != '.' || l > 100000) return false;
    const char* part = end + 1;
    const char* dot = strrchr(part, '.');
    if (!dot || strcmp(dot, ".weight") != 0) return false;
    const size_t part_len = (size_t)(dot - part);
    for (size_t i = 0; i < sizeof(LAYER_PARTS) / sizeof(LAYER_PARTS[0]); i++) {
        if (strlen(LAYER_PARTS[i].part) == part_len && strncmp(part, LAYER_PARTS[i].part, part_len) == 0) {
            *layer = (size_t)l;
            *experts = LAYER_PARTS[i].experts;
            *hf_part = LAYER_PARTS[i].hf;
            if (!*experts) snprintf(out, out_len, "model.layers.%lu.%s.weight", l, LAYER_PARTS[i].hf);
            return true;
        }
    }
    return false;
}

static bool set_err(char* err, size_t err_len, const char* fmt, const char* a, const char* b) {
    if (err && err_len > 0) snprintf(err, err_len, fmt, a ? a : "", b ? b : "");
    return false;
}

bool vitna_gguf_path(const char* path) {
    const size_t n = path ? strlen(path) : 0;
    if (n < 5) return false;
    const char* ext = path + n - 5;
    return ext[0] == '.' && tolower((unsigned char)ext[1]) == 'g' && tolower((unsigned char)ext[2]) == 'g' &&
           tolower((unsigned char)ext[3]) == 'u' && tolower((unsigned char)ext[4]) == 'f';
}

bool vitna_gguf_open(const char* path, vitna_safetensors_t* st, vitna_gguf_info_t* info, char* err, size_t err_len) {
    memset(st, 0, sizeof(*st));
    memset(info, 0, sizeof(*info));
    if (err && err_len) err[0] = '\0';
    if (!vitna_mmap_open(path, &st->mmap)) return set_err(err, err_len, "cannot open or map %s%s", path, NULL);

    cursor_t c = { (const uint8_t*)st->mmap.data, (const uint8_t*)st->mmap.data + st->mmap.size, true };
    number_t* nums = NULL;
    file_tensor_t* ft = NULL;
    bool ok = false;
    char msg[512];
    msg[0] = '\0';

    const uint32_t magic = (uint32_t)rd_le(&c, 4);
    info->version = (uint32_t)rd_le(&c, 4);
    const uint64_t n_tensors = rd_le(&c, 8);
    const uint64_t n_kv = rd_le(&c, 8);
    if (!c.ok || magic != GGUF_MAGIC) {
        snprintf(msg, sizeof(msg), "%s is not a GGUF file", path);
        goto done;
    }
    if (info->version != 2 && info->version != 3) {
        snprintf(msg, sizeof(msg), "%s is GGUF version %u; versions 2 and 3 are read", path, info->version);
        goto done;
    }
    /* Each entry and tensor takes at least 16 bytes, which bounds both counts by the file. */
    if (n_kv > (uint64_t)st->mmap.size / 16 || n_tensors > (uint64_t)st->mmap.size / 16) {
        snprintf(msg, sizeof(msg), "%s: more metadata or tensors than the file could hold", path);
        goto done;
    }

    nums = (number_t*)calloc(n_kv ? n_kv : 1, sizeof(number_t));
    size_t n_nums = 0;
    info->alignment = DEFAULT_ALIGNMENT;
    if (!nums) {
        snprintf(msg, sizeof(msg), "out of memory");
        goto done;
    }
    for (uint64_t i = 0; i < n_kv && c.ok; i++) {
        const char* key;
        uint64_t key_len;
        if (!rd_str(&c, &key, &key_len)) break;
        const uint32_t t = (uint32_t)rd_le(&c, 4);
        if (!c.ok) break;
        if (t == T_STR && key_len == 20 && memcmp(key, "general.architecture", 20) == 0) {
            const char* s;
            uint64_t n;
            if (!rd_str(&c, &s, &n)) break;
            if (n >= sizeof(info->arch)) {
                snprintf(msg, sizeof(msg), "%s: general.architecture is too long", path);
                goto done;
            }
            memcpy(info->arch, s, (size_t)n);
            info->arch[n] = '\0';
        } else if (scalar_size(t)) {
            nums[n_nums].key = key;
            nums[n_nums].len = (size_t)key_len;
            nums[n_nums].value = rd_number(&c, t);
            if (key_len == 17 && memcmp(key, "general.alignment", 17) == 0) info->alignment = (uint64_t)nums[n_nums].value;
            n_nums++;
        } else {
            skip_value(&c, t, 0);
        }
    }
    if (!c.ok) {
        snprintf(msg, sizeof(msg), "%s: its metadata runs past the end of the file, or holds a type GGUF does not have", path);
        goto done;
    }
    {
        const double raw_no = meta_f64(nums, n_nums, "split", "no");
        const double raw_cnt = meta_f64(nums, n_nums, "split", "count");
        const double raw_tc = meta_f64(nums, n_nums, "split", "tensors.count");
        if (!(raw_no >= 0.0 && raw_no <= 4096.0 && (double)(uint16_t)raw_no == raw_no) ||
            !(raw_cnt >= 0.0 && raw_cnt <= 4096.0 && (double)(uint16_t)raw_cnt == raw_cnt) ||
            !(raw_tc >= 0.0 && raw_tc <= 1e12 && (double)(uint64_t)raw_tc == raw_tc)) {
            snprintf(msg, sizeof(msg), "%s: invalid split metadata", path);
            goto done;
        }
        info->split_no = (uint16_t)raw_no;
        info->split_count = (uint16_t)raw_cnt;
        info->split_tensors_count = (uint64_t)raw_tc;
        if (info->split_count > 1) {
            if (info->split_no >= info->split_count) {
                snprintf(msg, sizeof(msg), "%s: split.no (%u) must be less than split.count (%u)", path,
                         (unsigned)info->split_no, (unsigned)info->split_count);
                goto done;
            }
        } else if (info->split_no != 0) {
            snprintf(msg, sizeof(msg), "%s: split.no (%u) set without split.count > 1", path, (unsigned)info->split_no);
            goto done;
        }
    }
    if (!info->arch[0]) {
        if (info->split_count <= 1 || info->split_no == 0) {
            snprintf(msg, sizeof(msg), "%s: no general.architecture", path);
            goto done;
        }
    } else if (strcmp(info->arch, "llama") == 0) {
        snprintf(msg, sizeof(msg), "%s: llama.cpp writes a llama model's query and key rows permuted, and this engine does not put them back", path);
        goto done;
    }
    if (info->alignment == 0 || info->alignment > (1u << 20) || (info->alignment & (info->alignment - 1))) {
        snprintf(msg, sizeof(msg), "%s: general.alignment is not a power of two up to 1 MiB", path);
        goto done;
    }
    if (info->arch[0]) {
        const char* a = info->arch;
        info->block_count = meta_u64(nums, n_nums, a, "block_count");
        info->embedding_length = meta_u64(nums, n_nums, a, "embedding_length");
        info->feed_forward_length = meta_u64(nums, n_nums, a, "feed_forward_length");
        info->expert_feed_forward_length = meta_u64(nums, n_nums, a, "expert_feed_forward_length");
        info->head_count = meta_u64(nums, n_nums, a, "attention.head_count");
        info->head_count_kv = meta_u64(nums, n_nums, a, "attention.head_count_kv");
        info->key_length = meta_u64(nums, n_nums, a, "attention.key_length");
        info->context_length = meta_u64(nums, n_nums, a, "context_length");
        info->expert_count = meta_u64(nums, n_nums, a, "expert_count");
        info->expert_used_count = meta_u64(nums, n_nums, a, "expert_used_count");
        info->rms_eps = meta_f64(nums, n_nums, a, "attention.layer_norm_rms_epsilon");
        info->rope_freq_base = meta_f64(nums, n_nums, a, "rope.freq_base");
    }

    ft = (file_tensor_t*)calloc(n_tensors ? n_tensors : 1, sizeof(file_tensor_t));
    if (!ft) {
        snprintf(msg, sizeof(msg), "out of memory");
        goto done;
    }
    size_t total = 0; /* tensors listed, each expert of a stack its own */
    for (uint64_t i = 0; i < n_tensors; i++) {
        const char* name;
        uint64_t name_len;
        if (!rd_str(&c, &name, &name_len)) break;
        if (name_len > NAME_MAX_GGUF) {
            snprintf(msg, sizeof(msg), "%s: a tensor name is longer than %d bytes", path, NAME_MAX_GGUF);
            goto done;
        }
        memcpy(ft[i].name, name, (size_t)name_len);
        ft[i].name[name_len] = '\0';
        ft[i].n_dims = (uint32_t)rd_le(&c, 4);
        if (!c.ok) break;
        if (ft[i].n_dims < 1 || ft[i].n_dims > 3) {
            snprintf(msg, sizeof(msg), "%s: tensor %s has %u dimensions; 1 to 3 are read", path, ft[i].name, ft[i].n_dims);
            goto done;
        }
        for (uint32_t d = 0; d < ft[i].n_dims; d++) ft[i].ne[d] = rd_le(&c, 8);
        ft[i].type = (uint32_t)rd_le(&c, 4);
        ft[i].offset = rd_le(&c, 8);
        if (!c.ok) break;
        for (uint64_t j = 0; j < i; j++) {
            if (strcmp(ft[j].name, ft[i].name) == 0) {
                snprintf(msg, sizeof(msg), "%s: tensor %s is listed twice", path, ft[i].name);
                goto done;
            }
        }
        total += ft[i].n_dims == 3 ? (size_t)(ft[i].ne[2] <= 65536 ? ft[i].ne[2] : 0) : 1;
    }
    if (!c.ok) {
        snprintf(msg, sizeof(msg), "%s: its tensor directory runs past the end of the file", path);
        goto done;
    }
    info->file_tensors = (size_t)n_tensors;

    /* The data section starts at the next multiple of the alignment. */
    const uint64_t pos = (uint64_t)(c.p - (const uint8_t*)st->mmap.data);
    const uint64_t data_start = (pos + info->alignment - 1) / info->alignment * info->alignment;
    if (data_start > (uint64_t)st->mmap.size) {
        snprintf(msg, sizeof(msg), "%s: no data section", path);
        goto done;
    }
    const uint64_t data_size = (uint64_t)st->mmap.size - data_start;
    const uint8_t* data = (const uint8_t*)st->mmap.data + data_start;
    st->data_base_offset = data_start;

    st->tensor_capacity = total ? total : 1;
    st->tensors = (vitna_tensor_desc_t*)calloc(st->tensor_capacity, sizeof(vitna_tensor_desc_t));
    if (!st->tensors) {
        snprintf(msg, sizeof(msg), "out of memory");
        goto done;
    }
    for (uint64_t i = 0; i < n_tensors; i++) {
        const file_tensor_t* t = &ft[i];
        char hf[VITNA_TENSOR_NAME_MAX];
        bool experts;
        size_t layer = 0;
        const char* part = NULL;
        if (!hf_name(t->name, hf, sizeof(hf), &experts, &layer, &part)) {
            snprintf(msg, sizeof(msg), "%s: tensor %s is not one this engine knows where to put", path, t->name);
            goto done;
        }
        const vitna_dtype_t dtype = ggml_dtype(t->type);
        if (dtype == VITNA_DTYPE_UNKNOWN) {
            snprintf(msg, sizeof(msg), "%s: tensor %s is %s; F32, F16, BF16, Q8_0, Q4_K and Q6_K are read", path, t->name,
                     ggml_type_name(t->type));
            goto done;
        }
        if (experts != (t->n_dims == 3)) {
            snprintf(msg, sizeof(msg), "%s: tensor %s has %u dimensions, where %s", path, t->name, t->n_dims,
                     experts ? "a stack of experts has 3" : "only a stack of experts has 3");
            goto done;
        }
        const uint64_t cols = t->ne[0];
        const uint64_t rows = t->n_dims >= 2 ? t->ne[1] : 1;
        const uint64_t count = t->n_dims == 3 ? t->ne[2] : 1;
        const uint64_t row_bytes = cols <= SIZE_MAX ? vitna_row_bytes(dtype, (size_t)cols) : 0;
        if (!row_bytes || rows == 0 || count == 0 || count > 65536) {
            snprintf(msg, sizeof(msg), "%s: tensor %s has rows that are not whole %s blocks, or no rows", path, t->name,
                     vitna_dtype_name(dtype));
            goto done;
        }
        if (rows > UINT64_MAX / row_bytes || rows * row_bytes > UINT64_MAX / count) {
            snprintf(msg, sizeof(msg), "%s: tensor %s overflows", path, t->name);
            goto done;
        }
        const uint64_t slice = rows * row_bytes;
        const uint64_t bytes = slice * count;
        if (t->offset % info->alignment != 0 || t->offset > data_size || bytes > data_size - t->offset) {
            snprintf(msg, sizeof(msg), "%s: tensor %s lies outside the data section, or off its alignment", path, t->name);
            goto done;
        }
        for (uint64_t e = 0; e < count; e++) {
            vitna_tensor_desc_t* d = &st->tensors[st->tensor_count++];
            if (experts) {
                snprintf(d->name, sizeof(d->name), "model.layers.%zu.mlp.experts.%llu.%s.weight", layer, (unsigned long long)e, part);
            } else {
                snprintf(d->name, sizeof(d->name), "%s", hf);
            }
            d->dtype = dtype;
            if (t->n_dims == 1) {
                d->ndim = 1;
                d->shape[0] = (size_t)cols;
            } else {
                d->ndim = 2;
                d->shape[0] = (size_t)rows;
                d->shape[1] = (size_t)cols;
            }
            d->offset_begin = t->offset + e * slice;
            d->offset_end = d->offset_begin + slice;
            d->data_ptr = data + d->offset_begin;
        }
    }
    ok = true;

done:
    free(nums);
    free(ft);
    if (!ok) {
        vitna_safetensors_close(st);
        set_err(err, err_len, "%s%s", msg[0] ? msg : "cannot read the GGUF file", NULL);
    }
    return ok;
}

static bool parse_5digits(const char* p, uint32_t* out) {
    uint32_t v = 0;
    for (int i = 0; i < 5; i++) {
        if (!isdigit((unsigned char)p[i])) return false;
        v = v * 10u + (uint32_t)(p[i] - '0');
    }
    *out = v;
    return true;
}

bool vitna_gguf_split_path(const char* path, uint16_t index, uint16_t count, char* out, size_t out_len) {
    if (!path || !out || out_len == 0 || count == 0 || index >= count || !vitna_gguf_path(path)) return false;
    /* Suffix is "-NNNNN-of-MMMMM.gguf" (20 bytes). */
    const size_t n = strlen(path);
    if (n <= 20) return false;
    const char* suf = path + n - 20;
    if (suf[0] != '-' || memcmp(suf + 6, "-of-", 4) != 0) return false;
    uint32_t cur_no = 0, cur_cnt = 0;
    if (!parse_5digits(suf + 1, &cur_no) || !parse_5digits(suf + 10, &cur_cnt)) return false;
    if (cur_no == 0 || cur_no > cur_cnt || cur_cnt != (uint32_t)count) return false;
    const size_t prefix_len = n - 20;
    const char* ext = path + n - 5; /* preserve original .gguf / .GGUF casing */
    const int wrote = snprintf(out, out_len, "%.*s-%05u-of-%05u%s", (int)prefix_len, path, (unsigned)(index + 1),
                               (unsigned)count, ext);
    return wrote > 0 && (size_t)wrote < out_len;
}

static char* dup_str(const char* s) {
    const size_t n = strlen(s) + 1;
    char* copy = (char*)malloc(n);
    if (copy) memcpy(copy, s, n);
    return copy;
}

bool vitna_gguf_open_shards(const char* path, vitna_safetensors_t** shards_out, char*** paths_out, size_t* n_shards_out,
                            vitna_gguf_info_t* info_out, char* err, size_t err_len) {
    *shards_out = NULL;
    *paths_out = NULL;
    *n_shards_out = 0;
    memset(info_out, 0, sizeof(*info_out));
    if (err && err_len) err[0] = '\0';

    vitna_safetensors_t first_st;
    vitna_gguf_info_t first_info;
    char g_err[512];
    if (!vitna_gguf_open(path, &first_st, &first_info, g_err, sizeof(g_err))) {
        return set_err(err, err_len, "%s%s", g_err, NULL);
    }

    if (first_info.split_count <= 1) {
        vitna_safetensors_t* shards = (vitna_safetensors_t*)calloc(1, sizeof(vitna_safetensors_t));
        char** paths = (char**)calloc(1, sizeof(char*));
        char* p0 = dup_str(path);
        if (!shards || !paths || !p0) {
            vitna_safetensors_close(&first_st);
            free(shards);
            free(paths);
            free(p0);
            return set_err(err, err_len, "out of memory%s%s", NULL, NULL);
        }
        shards[0] = first_st;
        paths[0] = p0;
        *shards_out = shards;
        *paths_out = paths;
        *n_shards_out = 1;
        *info_out = first_info;
        return true;
    }

    const uint16_t count = first_info.split_count;
    vitna_safetensors_t* shards = (vitna_safetensors_t*)calloc(count, sizeof(vitna_safetensors_t));
    char** paths = (char**)calloc(count, sizeof(char*));
    vitna_gguf_info_t* infos = (vitna_gguf_info_t*)calloc(count, sizeof(vitna_gguf_info_t));
    bool* opened = (bool*)calloc(count, sizeof(bool));
    bool ok = false;
    char msg[1536];
    msg[0] = '\0';

    if (!shards || !paths || !infos || !opened) {
        vitna_safetensors_close(&first_st);
        snprintf(msg, sizeof(msg), "out of memory");
        goto done;
    }

    /* Place the already-opened shard at its declared index so we never double-map it. */
    shards[first_info.split_no] = first_st;
    infos[first_info.split_no] = first_info;
    opened[first_info.split_no] = true;
    paths[first_info.split_no] = dup_str(path);
    if (!paths[first_info.split_no]) {
        snprintf(msg, sizeof(msg), "out of memory");
        goto done;
    }

    /* Validate that the supplied filename matches its declared split.no and split.count. */
    {
        char expected_self[1024];
        if (!vitna_gguf_split_path(path, first_info.split_no, count, expected_self, sizeof(expected_self))) {
            snprintf(msg, sizeof(msg), "%.400s: split.count is %u, but the filename does not match <prefix>-NNNNN-of-%05u.gguf",
                     path, (unsigned)count, (unsigned)count);
            goto done;
        }
        if (strcmp(path, expected_self) != 0) {
            snprintf(msg, sizeof(msg), "%.400s: split.no is %u, expected filename %.400s", path,
                     (unsigned)first_info.split_no, expected_self);
            goto done;
        }
    }

    for (uint16_t i = 0; i < count; i++) {
        if (opened[i]) continue;
        char spath[1024];
        if (!vitna_gguf_split_path(path, i, count, spath, sizeof(spath))) {
            snprintf(msg, sizeof(msg), "%s: cannot form path for split shard %u of %u", path, (unsigned)(i + 1),
                     (unsigned)count);
            goto done;
        }
        paths[i] = dup_str(spath);
        if (!paths[i]) {
            snprintf(msg, sizeof(msg), "out of memory");
            goto done;
        }
        if (!vitna_gguf_open(spath, &shards[i], &infos[i], g_err, sizeof(g_err))) {
            snprintf(msg, sizeof(msg), "%s", g_err);
            goto done;
        }
        opened[i] = true;
        if (infos[i].split_count != count || infos[i].split_no != i) {
            snprintf(msg, sizeof(msg), "%s: expected split %u of %u, found split %u of %u", spath, (unsigned)i,
                     (unsigned)count, (unsigned)infos[i].split_no, (unsigned)infos[i].split_count);
            goto done;
        }
    }

    /* Cross-shard consistency checks: architecture, split.tensors.count, and unique tensor names. */
    const vitna_gguf_info_t* meta0 = &infos[0];
    uint64_t total_file_tensors = 0;
    uint64_t expected_total = meta0->split_tensors_count;
    for (uint16_t i = 0; i < count; i++) {
        if (infos[i].arch[0] && strcmp(infos[i].arch, meta0->arch) != 0) {
            snprintf(msg, sizeof(msg), "%s: architecture %s disagrees with shard 1 (%s)", paths[i], infos[i].arch,
                     meta0->arch);
            goto done;
        }
        if (infos[i].split_tensors_count) {
            if (expected_total && infos[i].split_tensors_count != expected_total) {
                snprintf(msg, sizeof(msg), "%s: split.tensors.count (%llu) disagrees with shard 1 (%llu)", paths[i],
                         (unsigned long long)infos[i].split_tensors_count, (unsigned long long)expected_total);
                goto done;
            }
            expected_total = infos[i].split_tensors_count;
        }
        total_file_tensors += (uint64_t)infos[i].file_tensors;
    }
    if (expected_total && total_file_tensors != expected_total) {
        snprintf(msg, sizeof(msg), "%s: split shards hold %llu tensors, but split.tensors.count is %llu", paths[0],
                 (unsigned long long)total_file_tensors, (unsigned long long)expected_total);
        goto done;
    }

    for (uint16_t i = 0; i < count; i++) {
        for (size_t t = 0; t < shards[i].tensor_count; t++) {
            const char* tname = shards[i].tensors[t].name;
            for (uint16_t j = 0; j < i; j++) {
                if (vitna_safetensors_find(&shards[j], tname) != NULL) {
                    snprintf(msg, sizeof(msg), "%s: tensor %s also appears in %s", paths[i], tname, paths[j]);
                    goto done;
                }
            }
        }
    }

    *shards_out = shards;
    *paths_out = paths;
    *n_shards_out = count;
    *info_out = *meta0;
    info_out->file_tensors = (size_t)total_file_tensors;
    ok = true;

done:
    free(infos);
    if (!ok) {
        if (shards && opened) {
            for (uint16_t i = 0; i < count; i++) {
                if (opened[i]) vitna_safetensors_close(&shards[i]);
            }
        }
        if (paths) {
            for (uint16_t i = 0; i < count; i++) free(paths[i]);
        }
        free(shards);
        free(paths);
        set_err(err, err_len, "%s%s", msg[0] ? msg : "cannot open split GGUF checkpoint", NULL);
    }
    free(opened);
    return ok;
}
