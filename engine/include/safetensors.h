/**
 * safetensors.h - Zero-copy SafeTensors reader.
 *
 * Maps the file and parses its JSON header into a tensor directory. Every
 * tensor is checked on open: its dtype is known, its shape is at most 8
 * dimensions, its byte range lies inside the file's data section, and that
 * range holds exactly shape x dtype-size bytes. The __metadata__ entry is
 * skipped. Tensor data is read in place from the mapping.
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
    VITNA_DTYPE_BOOL,
    VITNA_DTYPE_F64,
    VITNA_DTYPE_I64,
    VITNA_DTYPE_U16,
    VITNA_DTYPE_U32,
    VITNA_DTYPE_U64
} vitna_dtype_t;

#define VITNA_TENSOR_NAME_MAX 256

typedef struct {
    char name[VITNA_TENSOR_NAME_MAX];
    vitna_dtype_t dtype;
    size_t shape[8];
    size_t ndim;
    uint64_t offset_begin;   /* relative to the start of the data section */
    uint64_t offset_end;
    const void* data_ptr;    /* into the mapping */
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

/** Bytes per element of a dtype, or 0 for VITNA_DTYPE_UNKNOWN. */
size_t vitna_dtype_size(vitna_dtype_t dtype);

/** The SafeTensors name of a dtype ("BF16", ...). */
const char* vitna_dtype_name(vitna_dtype_t dtype);

/**
 * Open a SafeTensors file, map it, and parse and check its tensor directory.
 * On failure returns false, and writes a reason to err when err is not NULL.
 */
bool vitna_safetensors_open_ex(const char* filepath, vitna_safetensors_t* st, char* err, size_t err_len);

/** vitna_safetensors_open_ex without the reason. */
bool vitna_safetensors_open(const char* filepath, vitna_safetensors_t* st);

/**
 * Locate a tensor descriptor by name in the parsed directory.
 * Returns NULL if not found.
 */
const vitna_tensor_desc_t* vitna_safetensors_find(const vitna_safetensors_t* st, const char* name);

/** Number of elements in a tensor: the product of its shape. */
uint64_t vitna_tensor_numel(const vitna_tensor_desc_t* t);

/**
 * Close and release all memory-mapped resources.
 */
void vitna_safetensors_close(vitna_safetensors_t* st);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_SAFETENSORS_H */
