/**
 * safetensors.c - Zero-copy SafeTensors reader.
 *
 * The header is parsed with json.c. The parser this replaced compared the
 * first 10 characters of a 12-character key ("__metadata__"), so the metadata
 * block was read as a tensor, and it never checked that a tensor's byte range
 * lay inside the file.
 */

#include "safetensors.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A header larger than this is refused rather than parsed. */
#define VITNA_SAFETENSORS_MAX_HEADER (100ull * 1024 * 1024)

static const struct { const char* name; vitna_dtype_t dtype; size_t size; } DTYPES[] = {
    { "F32", VITNA_DTYPE_F32, 4 },
    { "F16", VITNA_DTYPE_F16, 2 },
    { "BF16", VITNA_DTYPE_BF16, 2 },
    { "I32", VITNA_DTYPE_I32, 4 },
    { "I16", VITNA_DTYPE_I16, 2 },
    { "I8", VITNA_DTYPE_I8, 1 },
    { "U8", VITNA_DTYPE_U8, 1 },
    { "BOOL", VITNA_DTYPE_BOOL, 1 },
    { "F64", VITNA_DTYPE_F64, 8 },
    { "I64", VITNA_DTYPE_I64, 8 },
    { "U16", VITNA_DTYPE_U16, 2 },
    { "U32", VITNA_DTYPE_U32, 4 },
    { "U64", VITNA_DTYPE_U64, 8 },
};
#define N_DTYPES (sizeof(DTYPES) / sizeof(DTYPES[0]))

/* The block formats, which only a GGUF file holds, so parse_dtype never
   sees them: SafeTensors has no name for them. */
static const struct { const char* name; vitna_dtype_t dtype; size_t elems, bytes; } BLOCKS[] = {
    { "Q8_0", VITNA_DTYPE_Q8_0, 32, 34 },
    { "Q4_K", VITNA_DTYPE_Q4_K, 256, 144 },
    { "Q6_K", VITNA_DTYPE_Q6_K, 256, 210 },
};
#define N_BLOCKS (sizeof(BLOCKS) / sizeof(BLOCKS[0]))

size_t vitna_dtype_size(vitna_dtype_t dtype) {
    for (size_t i = 0; i < N_DTYPES; i++) if (DTYPES[i].dtype == dtype) return DTYPES[i].size;
    return 0;
}

bool vitna_dtype_is_block(vitna_dtype_t dtype) {
    for (size_t i = 0; i < N_BLOCKS; i++) if (BLOCKS[i].dtype == dtype) return true;
    return false;
}

size_t vitna_dtype_block_elems(vitna_dtype_t dtype) {
    for (size_t i = 0; i < N_BLOCKS; i++) if (BLOCKS[i].dtype == dtype) return BLOCKS[i].elems;
    return vitna_dtype_size(dtype) ? 1 : 0;
}

size_t vitna_dtype_block_bytes(vitna_dtype_t dtype) {
    for (size_t i = 0; i < N_BLOCKS; i++) if (BLOCKS[i].dtype == dtype) return BLOCKS[i].bytes;
    return vitna_dtype_size(dtype);
}

uint64_t vitna_row_bytes(vitna_dtype_t dtype, size_t cols) {
    const size_t elems = vitna_dtype_block_elems(dtype);
    if (!elems || cols % elems) return 0;
    return (uint64_t)(cols / elems) * vitna_dtype_block_bytes(dtype);
}

const char* vitna_dtype_name(vitna_dtype_t dtype) {
    for (size_t i = 0; i < N_DTYPES; i++) if (DTYPES[i].dtype == dtype) return DTYPES[i].name;
    for (size_t i = 0; i < N_BLOCKS; i++) if (BLOCKS[i].dtype == dtype) return BLOCKS[i].name;
    return "UNKNOWN";
}

static vitna_dtype_t parse_dtype(const char* s) {
    for (size_t i = 0; i < N_DTYPES; i++) if (strcmp(DTYPES[i].name, s) == 0) return DTYPES[i].dtype;
    return VITNA_DTYPE_UNKNOWN;
}

static bool set_err(char* err, size_t err_len, const char* fmt, const char* a) {
    if (err && err_len > 0) snprintf(err, err_len, fmt, a);
    return false;
}

/* A non-negative integer that a JSON number holds exactly. */
static bool as_u64(const vitna_json_value_t* v, uint64_t* out) {
    double d;
    if (!vitna_json_as_number(v, &d)) return false;
    if (!(d >= 0.0) || d > 9007199254740992.0 || d != (double)(uint64_t)d) return false;
    *out = (uint64_t)d;
    return true;
}

uint64_t vitna_tensor_numel(const vitna_tensor_desc_t* t) {
    uint64_t n = 1;
    for (size_t d = 0; d < t->ndim; d++) n *= (uint64_t)t->shape[d];
    return n;
}

static bool parse_directory(vitna_safetensors_t* st, char* err, size_t err_len) {
    char jerr[160];
    vitna_json_doc_t* doc = vitna_json_parse(st->header_json, (size_t)st->header_len, jerr, sizeof(jerr));
    if (!doc) return set_err(err, err_len, "header: %s", jerr);

    bool ok = false;
    const vitna_json_value_t* root = vitna_json_root(doc);
    const uint64_t data_size = (uint64_t)st->mmap.size - st->data_base_offset;
    const uint8_t* data_base = (const uint8_t*)st->mmap.data + st->data_base_offset;

    if (root->type != VITNA_JSON_OBJECT) {
        set_err(err, err_len, "%s", "header is not a JSON object");
        goto done;
    }
    st->tensor_capacity = root->u.object.count > 0 ? root->u.object.count : 1;
    st->tensors = (vitna_tensor_desc_t*)calloc(st->tensor_capacity, sizeof(vitna_tensor_desc_t));
    if (!st->tensors) {
        set_err(err, err_len, "%s", "out of memory");
        goto done;
    }

    for (size_t i = 0; i < root->u.object.count; i++) {
        const vitna_json_member_t* m = &root->u.object.members[i];
        if (strcmp(m->key, "__metadata__") == 0) {
            if (m->value->type != VITNA_JSON_OBJECT) {
                set_err(err, err_len, "%s", "__metadata__ is not an object");
                goto done;
            }
            continue;
        }
        if (m->key_len >= VITNA_TENSOR_NAME_MAX) {
            set_err(err, err_len, "tensor name too long: %.64s...", m->key);
            goto done;
        }
        const vitna_json_value_t* entry = m->value;
        const char* dtype_s = vitna_json_as_string(vitna_json_get(entry, "dtype"));
        const vitna_json_value_t* shape = vitna_json_get(entry, "shape");
        const vitna_json_value_t* offsets = vitna_json_get(entry, "data_offsets");
        if (entry->type != VITNA_JSON_OBJECT || !dtype_s || !shape || shape->type != VITNA_JSON_ARRAY ||
            !offsets || offsets->type != VITNA_JSON_ARRAY || offsets->u.array.count != 2) {
            set_err(err, err_len, "tensor %s: needs dtype, shape and two data_offsets", m->key);
            goto done;
        }
        vitna_tensor_desc_t* t = &st->tensors[st->tensor_count];
        memcpy(t->name, m->key, m->key_len + 1);
        t->dtype = parse_dtype(dtype_s);
        if (t->dtype == VITNA_DTYPE_UNKNOWN) {
            set_err(err, err_len, "tensor %s: unknown dtype", m->key);
            goto done;
        }
        if (shape->u.array.count > 8) {
            set_err(err, err_len, "tensor %s: more than 8 dimensions", m->key);
            goto done;
        }
        t->ndim = shape->u.array.count;
        uint64_t numel = 1;
        for (size_t d = 0; d < t->ndim; d++) {
            uint64_t dim;
            if (!as_u64(shape->u.array.items[d], &dim) || dim > SIZE_MAX) {
                set_err(err, err_len, "tensor %s: bad shape", m->key);
                goto done;
            }
            t->shape[d] = (size_t)dim;
            if (dim != 0 && numel > UINT64_MAX / dim) {
                set_err(err, err_len, "tensor %s: shape overflows", m->key);
                goto done;
            }
            numel *= dim;
        }
        if (!as_u64(offsets->u.array.items[0], &t->offset_begin) || !as_u64(offsets->u.array.items[1], &t->offset_end) ||
            t->offset_begin > t->offset_end || t->offset_end > data_size) {
            set_err(err, err_len, "tensor %s: data_offsets outside the data section", m->key);
            goto done;
        }
        if (t->offset_end - t->offset_begin != numel * vitna_dtype_size(t->dtype)) {
            set_err(err, err_len, "tensor %s: byte range does not match shape x dtype size", m->key);
            goto done;
        }
        t->data_ptr = data_base + t->offset_begin;
        st->tensor_count++;
    }
    ok = true;

done:
    vitna_json_free(doc);
    return ok;
}

bool vitna_safetensors_open_ex(const char* filepath, vitna_safetensors_t* st, char* err, size_t err_len) {
    if (!filepath || !st) return false;
    memset(st, 0, sizeof(*st));
    if (err && err_len > 0) err[0] = '\0';

    if (!vitna_mmap_open(filepath, &st->mmap)) {
        return set_err(err, err_len, "cannot open or map %s", filepath);
    }
    if (st->mmap.size < 8) {
        vitna_safetensors_close(st);
        return set_err(err, err_len, "%s is shorter than a header length", filepath);
    }

    /* 8-byte little-endian header length */
    const uint8_t* raw = (const uint8_t*)st->mmap.data;
    st->header_len = 0;
    for (int i = 7; i >= 0; i--) st->header_len = (st->header_len << 8) | raw[i];

    if (st->header_len > VITNA_SAFETENSORS_MAX_HEADER || 8 + st->header_len > (uint64_t)st->mmap.size) {
        vitna_safetensors_close(st);
        return set_err(err, err_len, "%s: header length is larger than the file allows", filepath);
    }

    st->header_json = (const char*)(raw + 8);
    st->data_base_offset = 8 + st->header_len;

    if (!parse_directory(st, err, err_len)) {
        vitna_safetensors_close(st);
        return false;
    }
    return true;
}

bool vitna_safetensors_open(const char* filepath, vitna_safetensors_t* st) {
    return vitna_safetensors_open_ex(filepath, st, NULL, 0);
}

const vitna_tensor_desc_t* vitna_safetensors_find(const vitna_safetensors_t* st, const char* name) {
    if (!st || !name) return NULL;
    for (size_t i = 0; i < st->tensor_count; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            return &st->tensors[i];
        }
    }
    return NULL;
}

void vitna_safetensors_close(vitna_safetensors_t* st) {
    if (!st) return;
    free(st->tensors);
    st->tensors = NULL;
    st->tensor_count = 0;
    st->tensor_capacity = 0;
    vitna_mmap_close(&st->mmap);
}
