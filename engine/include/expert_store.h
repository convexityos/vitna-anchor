/**
 * expert_store.h - Multitier memory hierarchy, multi-precision, and expert streaming cache.
 *
 * Implements:
 * - Resident dense tier (RAM/VRAM)
 * - Multi-precision expert tiers (INT8, INT4, INT3, INT2)
 * - LRU cache for dynamically staged routed experts
 * - Pinned hot-store for empirically hot experts
 * - Semantic prefetching for lookahead expert queuing
 * - Telemetry tracking (hits, misses, read volume)
 */

#ifndef VITNA_EXPERT_STORE_H
#define VITNA_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "compat.h"
#include "kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t layer_idx;
    uint32_t expert_idx;
    size_t size_bytes;
    uint64_t file_offset;
    int file_shard_idx;
    void* buffer;         /* Pointer to resident weight data in RAM */
    uint64_t last_used_turn;
    bool is_pinned;
    vitna_quant_type_t quant_type;
} vitna_expert_slot_t;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t bytes_streamed;
    double time_io_ms;
} vitna_expert_stats_t;

typedef struct {
    vitna_file_t* shards;
    size_t shard_count;

    vitna_expert_slot_t* slots;
    size_t slot_count;
    size_t slot_capacity;

    size_t ram_budget_bytes;
    size_t ram_used_bytes;

    uint64_t current_turn;
    vitna_expert_stats_t stats;
} vitna_expert_store_t;

/**
 * Initialize an expert store with a specified RAM cache budget in bytes.
 */
bool vitna_expert_store_init(
    vitna_expert_store_t* store,
    size_t ram_budget_bytes,
    const char** shard_paths,
    size_t shard_count
);

/**
 * Register an expert's location on NVMe storage.
 */
bool vitna_expert_store_register(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    uint32_t expert_idx,
    int shard_idx,
    uint64_t file_offset,
    size_t size_bytes,
    bool pin_hot
);

/**
 * Register an expert with an explicit precision tier (INT8, INT4, INT3, INT2).
 */
bool vitna_expert_store_register_quant(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    uint32_t expert_idx,
    int shard_idx,
    uint64_t file_offset,
    size_t size_bytes,
    bool pin_hot,
    vitna_quant_type_t quant_type
);

/**
 * Acquire expert weights for computation.
 * If in RAM/VRAM, returns immediately (cache hit).
 * If on NVMe, streams via positional read into the LRU cache (cache miss).
 */
const void* vitna_expert_store_acquire(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    uint32_t expert_idx
);

/**
 * Asynchronously prefetch predicted downstream experts into the LRU RAM cache.
 */
void vitna_expert_store_prefetch(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    const uint32_t* predicted_expert_ids,
    size_t count
);

/**
 * Pin anchor experts identified by prefix affinity caching.
 */
void vitna_expert_store_pin_prefix(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    const uint32_t* expert_ids,
    size_t count
);

/**
 * Advance the current turn counter (used for LRU eviction order).
 */
void vitna_expert_store_step_turn(vitna_expert_store_t* store);

/**
 * Snapshot current I/O and cache telemetry.
 */
vitna_expert_stats_t vitna_expert_store_get_stats(const vitna_expert_store_t* store);

/**
 * Release all resources in the expert store.
 */
void vitna_expert_store_destroy(vitna_expert_store_t* store);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_EXPERT_STORE_H */
