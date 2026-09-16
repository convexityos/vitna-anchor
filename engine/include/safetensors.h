/**
 * safetensors.h - Zero-copy Safetensors header parser and tensor locator.
 */

#ifndef VITNA_SAFETENSORS_H
#define VITNA_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "compat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VITNA_DTYPE_UNKNOWN = 0,
    VITNA_DTYPE_F32,
    VITNA_DTYPE_F16,
    VITNA_DTYPE_BF16,
    VITNA_DTYPE_I32,
    VITNA_DTYPE_I16,
    VITNA_DTYPE_I8,
    VITNA_DTYPE_U8,
    VITNA_DTYPE_BOOL
} vitna_dtype_t;

typedef struct {
    char name[128];
    vitna_dtype_t dtype;
    size_t shape[8];
    size_t ndim;
    uint64_t offset_begin;
    uint64_t offset_end;
    const void* data_ptr; /* non-NULL if mmap is active */
} vitna_tensor_desc_t;

typedef struct {
    vitna_mmap_t mmap;
    uint64_t header_len;
    const char* header_json;
    vitna_tensor_desc_t* tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    uint64_t data_base_offset;
} vitna_safetensors_t;

/**
 * Open a safetensors file via memory-mapping and parse its tensor directory.
 */
bool vitna_safetensors_open(const char* filepath, vitna_safetensors_t* st);

/**
 * Locate a tensor descriptor by name in the parsed directory.
 * Returns NULL if not found.
 */
const vitna_tensor_desc_t* vitna_safetensors_find(const vitna_safetensors_t* st, const char* name);

/**
 * Close and release all memory-mapped resources.
 */
void vitna_safetensors_close(vitna_safetensors_t* st);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_SAFETENSORS_H */
