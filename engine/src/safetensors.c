/**
 * safetensors.c - Lean, zero-dependency Safetensors parser.
 */

#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static vitna_dtype_t parse_dtype(const char* str, size_t len) {
    if (len == 3 && strncmp(str, "F32", 3) == 0) return VITNA_DTYPE_F32;
    if (len == 3 && strncmp(str, "F16", 3) == 0) return VITNA_DTYPE_F16;
    if (len == 4 && strncmp(str, "BF16", 4) == 0) return VITNA_DTYPE_BF16;
    if (len == 3 && strncmp(str, "I32", 3) == 0) return VITNA_DTYPE_I32;
    if (len == 3 && strncmp(str, "I16", 3) == 0) return VITNA_DTYPE_I16;
    if (len == 2 && strncmp(str, "I8", 2) == 0)  return VITNA_DTYPE_I8;
    if (len == 2 && strncmp(str, "U8", 2) == 0)  return VITNA_DTYPE_U8;
    if (len == 4 && strncmp(str, "BOOL", 4) == 0) return VITNA_DTYPE_BOOL;
    return VITNA_DTYPE_UNKNOWN;
}

static const char* skip_ws(const char* p, const char* end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    return p;
}

static bool parse_json_directory(vitna_safetensors_t* st) {
    const char* p = st->header_json;
    const char* end = st->header_json + st->header_len;

    p = skip_ws(p, end);
    if (p >= end || *p != '{') return false;
    p++;

    st->tensor_capacity = 64;
    st->tensor_count = 0;
    st->tensors = (vitna_tensor_desc_t*)malloc(st->tensor_capacity * sizeof(vitna_tensor_desc_t));
    if (!st->tensors) return false;

    const uint8_t* raw_base = (const uint8_t*)st->mmap.data + st->data_base_offset;

    while (p < end) {
        p = skip_ws(p, end);
        if (p >= end || *p == '}') break;
        if (*p == ',') { p++; continue; }

        if (*p != '"') break;
        p++;
        const char* key_start = p;
        while (p < end && *p != '"') p++;
        if (p >= end) break;
        size_t key_len = (size_t)(p - key_start);
        p++; /* skip closing quote */

        p = skip_ws(p, end);
        if (p >= end || *p != ':') break;
        p++;
        p = skip_ws(p, end);

        if (key_len == 10 && strncmp(key_start, "__metadata__", 10) == 0) {
            /* Skip metadata block */
            int brace_depth = 0;
            while (p < end) {
                if (*p == '{') brace_depth++;
                else if (*p == '}') {
                    brace_depth--;
                    if (brace_depth <= 0) { p++; break; }
                }
                p++;
            }
            continue;
        }

        /* Tensor entry */
        if (*p != '{') break;
        p++;

        if (st->tensor_count >= st->tensor_capacity) {
            size_t new_cap = st->tensor_capacity * 2;
            vitna_tensor_desc_t* resized = (vitna_tensor_desc_t*)realloc(st->tensors, new_cap * sizeof(vitna_tensor_desc_t));
            if (!resized) return false;
            st->tensors = resized;
            st->tensor_capacity = new_cap;
        }

        vitna_tensor_desc_t* t = &st->tensors[st->tensor_count];
        memset(t, 0, sizeof(*t));
        size_t copy_len = key_len < sizeof(t->name) - 1 ? key_len : sizeof(t->name) - 1;
        memcpy(t->name, key_start, copy_len);
        t->name[copy_len] = '\0';

        /* Parse fields inside tensor definition */
        while (p < end && *p != '}') {
            p = skip_ws(p, end);
            if (*p == ',') { p++; continue; }
            if (*p != '"') { p++; continue; }
            p++;
            const char* field_start = p;
            while (p < end && *p != '"') p++;
            size_t field_len = (size_t)(p - field_start);
            if (p < end) p++; /* skip quote */
            p = skip_ws(p, end);
            if (p < end && *p == ':') p++;
            p = skip_ws(p, end);

            if (field_len == 5 && strncmp(field_start, "dtype", 5) == 0) {
                if (*p == '"') {
                    p++;
                    const char* d_start = p;
                    while (p < end && *p != '"') p++;
                    t->dtype = parse_dtype(d_start, (size_t)(p - d_start));
                    if (p < end) p++;
                }
            } else if (field_len == 5 && strncmp(field_start, "shape", 5) == 0) {
                if (*p == '[') {
                    p++;
                    t->ndim = 0;
                    while (p < end && *p != ']') {
                        p = skip_ws(p, end);
                        if (*p == ',') { p++; continue; }
                        if (isdigit((unsigned char)*p)) {
                            char* next_p;
                            unsigned long val = strtoul(p, &next_p, 10);
                            if (t->ndim < 8) {
                                t->shape[t->ndim++] = (size_t)val;
                            }
                            p = next_p;
                        } else {
                            p++;
                        }
                    }
                    if (p < end && *p == ']') p++;
                }
            } else if (field_len == 12 && strncmp(field_start, "data_offsets", 12) == 0) {
                if (*p == '[') {
                    p++;
                    p = skip_ws(p, end);
                    char* next_p;
                    t->offset_begin = strtoull(p, &next_p, 10);
                    p = next_p;
                    p = skip_ws(p, end);
                    if (*p == ',') p++;
                    p = skip_ws(p, end);
                    t->offset_end = strtoull(p, &next_p, 10);
                    p = next_p;
                    while (p < end && *p != ']') p++;
                    if (p < end && *p == ']') p++;
                }
            } else {
                /* Skip unhandled field value */
                while (p < end && *p != ',' && *p != '}') p++;
            }
        }
        if (p < end && *p == '}') p++;

        t->data_ptr = raw_base + t->offset_begin;
        st->tensor_count++;
    }

    return true;
}

bool vitna_safetensors_open(const char* filepath, vitna_safetensors_t* st) {
    if (!filepath || !st) return false;
    memset(st, 0, sizeof(*st));

    if (!vitna_mmap_open(filepath, &st->mmap)) {
        return false;
    }

    if (st->mmap.size < 8) {
        vitna_safetensors_close(st);
        return false;
    }

    /* Read 8-byte little-endian header size */
    const uint8_t* raw = (const uint8_t*)st->mmap.data;
    st->header_len = (uint64_t)raw[0] |
                    ((uint64_t)raw[1] << 8) |
                    ((uint64_t)raw[2] << 16) |
                    ((uint64_t)raw[3] << 24) |
                    ((uint64_t)raw[4] << 32) |
                    ((uint64_t)raw[5] << 40) |
                    ((uint64_t)raw[6] << 48) |
                    ((uint64_t)raw[7] << 56);

    if (8 + st->header_len > st->mmap.size) {
        vitna_safetensors_close(st);
        return false;
    }

    st->header_json = (const char*)(raw + 8);
    st->data_base_offset = 8 + st->header_len;

    if (!parse_json_directory(st)) {
        vitna_safetensors_close(st);
        return false;
    }

    return true;
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
    if (st->tensors) {
        free(st->tensors);
        st->tensors = NULL;
    }
    st->tensor_count = 0;
    st->tensor_capacity = 0;
    vitna_mmap_close(&st->mmap);
}
