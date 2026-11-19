/**
 * gpu_plan.h - Multi-GPU device selection and pipeline-parallel layer splitter.
 *
 * Pure C11, platform-independent planning for Gate A13 (splitting a dense or
 * mixture-of-experts model across multiple NVIDIA or AMD GPUs with equal or
 * heterogeneous VRAM). Computes contiguous [layer_begin, layer_end) assignments
 * proportional to each GPU's usable memory (or explicit caller ratios),
 * accounting for embedding weights on the first GPU, output head weights on
 * the last GPU, per-GPU VRAM reserve headroom, and remaining per-GPU MoE
 * expert cache capacity.
 */

#ifndef VITNA_GPU_PLAN_H
#define VITNA_GPU_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VITNA_MAX_GPUS 16

/** Default VRAM headroom left free per GPU (512 MiB). */
#define VITNA_GPU_DEFAULT_RESERVE ((size_t)512 << 20)

typedef struct {
    int index;
    char name[128];
    size_t total_bytes;
    size_t free_bytes;
} vitna_gpu_device_spec_t;

typedef struct {
    size_t n_layers;
    size_t embed_bytes;           /* token embedding table (placed on first shard) */
    size_t head_bytes;            /* final norm + lm_head (placed on last shard) */
    size_t layer_weights_bytes;   /* resident non-expert weights per layer */
    size_t layer_kv_bytes;        /* KV cache bytes per layer at target ctx * seqs */
    size_t expert_bytes_per_layer;/* total MoE expert weight bytes per layer (0 for dense) */
    size_t reserve_bytes;         /* headroom kept free per GPU (0 uses default 512 MiB) */
} vitna_gpu_model_budget_t;

typedef struct {
    int device_index;
    size_t layer_begin;           /* inclusive */
    size_t layer_end;             /* exclusive */
    size_t resident_bytes;        /* weights + KV + embed/head placed on this GPU */
    size_t expert_cache_bytes;    /* remaining usable VRAM for MoE experts on this GPU */
    bool holds_embed;
    bool holds_head;
} vitna_gpu_shard_t;

typedef struct {
    vitna_gpu_shard_t shards[VITNA_MAX_GPUS];
    size_t n_shards;
    size_t total_resident_bytes;
    size_t total_expert_cache_bytes;
    bool fits;
    char reason[256];
} vitna_gpu_split_plan_t;

/**
 * Parse a comma-separated GPU device list such as "0", "0,1", "cuda:0,1", or
 * "hip:1,2". Rejects empty input, non-numeric tokens, negative or >=
 * VITNA_MAX_GPUS ids, and duplicate ids.
 */
bool vitna_gpu_parse_device_list(const char* spec, int* out_ids, size_t max_ids, size_t* out_count, char* err,
                                 size_t err_len);

/**
 * Parse comma-separated positive split weights (e.g. "0.4,0.6" or "1,3") for
 * expected_count GPUs, normalizing them to sum to 1.0 in out_ratios. Rejects
 * wrong count, non-positive, non-finite, or malformed entries.
 */
bool vitna_gpu_parse_ratios(const char* spec, size_t expected_count, double* out_ratios, char* err, size_t err_len);

/**
 * Compute a contiguous layer-split plan across devs[0..n_devs-1].
 * When explicit_ratios is non-NULL, layers are apportioned by those weights;
 * otherwise each GPU receives layers proportional to its usable VRAM after
 * fixed overhead (reserve_bytes + embed/head). Always populates plan->fits
 * and plan->reason (on failure).
 */
bool vitna_gpu_plan_split(const vitna_gpu_device_spec_t* devs, size_t n_devs, const vitna_gpu_model_budget_t* model,
                          const double* explicit_ratios, vitna_gpu_split_plan_t* plan);

/** Return the device_index assigned to layer, or -1 if out of range / invalid plan. */
int vitna_gpu_layer_device(const vitna_gpu_split_plan_t* plan, size_t layer);

/** Format a concise human-readable summary of plan into buf. Returns buf. */
const char* vitna_gpu_plan_format(const vitna_gpu_split_plan_t* plan, char* buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_GPU_PLAN_H */
