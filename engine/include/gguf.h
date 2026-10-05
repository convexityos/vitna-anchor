/**
 * gguf.h - A GGUF file (versions 2 and 3) read as a checkpoint.
 *
 * Maps the file, reads its metadata and its tensor directory, and checks
 * every tensor on open: its type is one the engine computes with (F32, F16,
 * BF16, Q8_0, Q4_K or Q6_K), its rows are whole blocks, its data offset is
 * aligned as the file says, and its bytes lie inside the file.
 *
 * The tensors are then listed under the names a Hugging Face checkpoint of
 * the same model gives them, in that checkpoint's shapes (rows, then
 * columns), so the loader finds them as it finds a SafeTensors file's. llama.cpp
 * stacks a layer's experts into one tensor for each of gate, up and down;
 * each expert is listed as a tensor of its own, a view of its slice of the
 * stack, which is contiguous. A tensor whose name this does not know is
 * refused rather than skipped, so nothing in the file goes unused unseen.
 *
 * llama.cpp permutes the query and key rows of a Llama-architecture model
 * for its interleaved rotary embedding; a file of that architecture is
 * refused, since its rows would need putting back. OLMoE's and Qwen's MoE
 * files are written unpermuted.
 */

#ifndef VITNA_GGUF_H
#define VITNA_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What the file's metadata says about the model, for checking against
   config.json. A field the file leaves out stays 0. */
typedef struct {
    char arch[64];                /* general.architecture */
    uint32_t version;             /* of the GGUF format */
    uint64_t alignment;           /* of tensor data, general.alignment or 32 */
    uint64_t block_count;
    uint64_t embedding_length;
    uint64_t feed_forward_length;
    uint64_t expert_feed_forward_length;
    uint64_t head_count;
    uint64_t head_count_kv;
    uint64_t key_length;
    uint64_t context_length;
    uint64_t expert_count;
    uint64_t expert_used_count;
    double rms_eps;               /* attention.layer_norm_rms_epsilon */
    double rope_freq_base;        /* rope.freq_base */
    size_t file_tensors;          /* tensors in the file, before experts are split out */
} vitna_gguf_info_t;

/**
 * Open path as a checkpoint, filling st (its mapping and its tensors under
 * Hugging Face names) and info. On failure returns false and writes a
 * reason to err. Close st with vitna_safetensors_close.
 */
bool vitna_gguf_open(const char* path, vitna_safetensors_t* st, vitna_gguf_info_t* info, char* err, size_t err_len);

/** True when path ends in .gguf. */
bool vitna_gguf_path(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_GGUF_H */
