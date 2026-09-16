/**
 * expert_store.c - Multitier expert cache, multi-precision, and NVMe streaming implementation.
 */

#include "expert_store.h"
#include <stdlib.h>
#include <string.h>

bool vitna_expert_store_init(
    vitna_expert_store_t* store,
    size_t ram_budget_bytes,
    const char** shard_paths,
    size_t shard_count
) {
    if (!store) return false;
    memset(store, 0, sizeof(*store));

    store->ram_budget_bytes = ram_budget_bytes;
    store->shard_count = shard_count;

    if (shard_count > 0 && shard_paths) {
        store->shards = (vitna_file_t*)malloc(shard_count * sizeof(vitna_file_t));
        if (!store->shards) return false;

        for (size_t i = 0; i < shard_count; i++) {
            if (!vitna_file_open_read(shard_paths[i], false, &store->shards[i])) {
                for (size_t j = 0; j < i; j++) {
                    vitna_file_close(&store->shards[j]);
                }
                free(store->shards);
                store->shards = NULL;
                return false;
            }
        }
    }

    store->slot_capacity = 256;
    store->slots = (vitna_expert_slot_t*)malloc(store->slot_capacity * sizeof(vitna_expert_slot_t));
    if (!store->slots) {
        vitna_expert_store_destroy(store);
        return false;
    }

    return true;
}

bool vitna_expert_store_register_quant(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    uint32_t expert_idx,
    int shard_idx,
    uint64_t file_offset,
    size_t size_bytes,
    bool pin_hot,
    vitna_quant_type_t quant_type
) {
    if (!store) return false;

    if (store->slot_count >= store->slot_capacity) {
        size_t new_cap = store->slot_capacity * 2;
        vitna_expert_slot_t* resized = (vitna_expert_slot_t*)realloc(store->slots, new_cap * sizeof(vitna_expert_slot_t));
        if (!resized) return false;
        store->slots = resized;
        store->slot_capacity = new_cap;
    }

    vitna_expert_slot_t* slot = &store->slots[store->slot_count++];
    slot->layer_idx = layer_idx;
    slot->expert_idx = expert_idx;
    slot->size_bytes = size_bytes;
    slot->file_offset = file_offset;
    slot->file_shard_idx = shard_idx;
    slot->buffer = NULL;
    slot->last_used_turn = 0;
    slot->is_pinned = pin_hot;
    slot->quant_type = quant_type;

    return true;
}

bool vitna_expert_store_register(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    uint32_t expert_idx,
    int shard_idx,
    uint64_t file_offset,
    size_t size_bytes,
    bool pin_hot
) {
    return vitna_expert_store_register_quant(
        store, layer_idx, expert_idx, shard_idx, file_offset, size_bytes, pin_hot, VITNA_QUANT_INT4
    );
}

static vitna_expert_slot_t* find_slot(vitna_expert_store_t* store, uint32_t layer_idx, uint32_t expert_idx) {
    for (size_t i = 0; i < store->slot_count; i++) {
        if (store->slots[i].layer_idx == layer_idx && store->slots[i].expert_idx == expert_idx) {
            return &store->slots[i];
        }
    }
    return NULL;
}

static bool evict_lru_candidate(vitna_expert_store_t* store, size_t needed_bytes) {
    while (store->ram_used_bytes + needed_bytes > store->ram_budget_bytes) {
        vitna_expert_slot_t* oldest = NULL;
        uint64_t min_turn = UINT64_MAX;

        for (size_t i = 0; i < store->slot_count; i++) {
            vitna_expert_slot_t* s = &store->slots[i];
            if (s->buffer && !s->is_pinned) {
                if (s->last_used_turn < min_turn) {
                    min_turn = s->last_used_turn;
                    oldest = s;
                }
            }
        }

        if (!oldest) {
            /* All resident slots are pinned or nothing left to evict */
            return false;
        }

        vitna_aligned_free(oldest->buffer);
        oldest->buffer = NULL;
        store->ram_used_bytes -= oldest->size_bytes;
    }
    return true;
}

const void* vitna_expert_store_acquire(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    uint32_t expert_idx
) {
    if (!store) return NULL;

    vitna_expert_slot_t* slot = find_slot(store, layer_idx, expert_idx);
    if (!slot) return NULL;

    slot->last_used_turn = store->current_turn;

    if (slot->buffer) {
        /* Cache Hit */
        store->stats.hits++;
        return slot->buffer;
    }

    /* Cache Miss - Stream from NVMe */
    store->stats.misses++;

    if (slot->file_shard_idx < 0 || (size_t)slot->file_shard_idx >= store->shard_count) {
        return NULL;
    }

    /* Make room in RAM budget */
    evict_lru_candidate(store, slot->size_bytes);

    void* buf = vitna_aligned_alloc(64, slot->size_bytes);
    if (!buf) return NULL;

    double t0 = vitna_time_ms();
    int64_t bytes_read = vitna_file_pread(
        &store->shards[slot->file_shard_idx],
        buf,
        slot->size_bytes,
        slot->file_offset
    );
    double t1 = vitna_time_ms();

    if (bytes_read < 0 || (size_t)bytes_read != slot->size_bytes) {
        vitna_aligned_free(buf);
        return NULL;
    }

    slot->buffer = buf;
    store->ram_used_bytes += slot->size_bytes;
    store->stats.bytes_streamed += (uint64_t)bytes_read;
    store->stats.time_io_ms += (t1 - t0);

    return slot->buffer;
}

void vitna_expert_store_prefetch(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    const uint32_t* predicted_expert_ids,
    size_t count
) {
    if (!store || !predicted_expert_ids || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        vitna_expert_slot_t* slot = find_slot(store, layer_idx, predicted_expert_ids[i]);
        if (slot && !slot->buffer) {
            /* Warm into cache ahead of time */
            vitna_expert_store_acquire(store, layer_idx, predicted_expert_ids[i]);
        }
    }
}

void vitna_expert_store_pin_prefix(
    vitna_expert_store_t* store,
    uint32_t layer_idx,
    const uint32_t* expert_ids,
    size_t count
) {
    if (!store || !expert_ids || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        vitna_expert_slot_t* slot = find_slot(store, layer_idx, expert_ids[i]);
        if (slot) {
            slot->is_pinned = true;
            if (!slot->buffer) {
                vitna_expert_store_acquire(store, layer_idx, expert_ids[i]);
            }
        }
    }
}

void vitna_expert_store_step_turn(vitna_expert_store_t* store) {
    if (store) store->current_turn++;
}

vitna_expert_stats_t vitna_expert_store_get_stats(const vitna_expert_store_t* store) {
    if (!store) {
        vitna_expert_stats_t empty = {0};
        return empty;
    }
    return store->stats;
}

void vitna_expert_store_destroy(vitna_expert_store_t* store) {
    if (!store) return;

    if (store->slots) {
        for (size_t i = 0; i < store->slot_count; i++) {
            if (store->slots[i].buffer) {
                vitna_aligned_free(store->slots[i].buffer);
                store->slots[i].buffer = NULL;
            }
        }
        free(store->slots);
        store->slots = NULL;
    }

    if (store->shards) {
        for (size_t i = 0; i < store->shard_count; i++) {
            vitna_file_close(&store->shards[i]);
        }
        free(store->shards);
        store->shards = NULL;
    }

    store->slot_count = 0;
    store->slot_capacity = 0;
    store->shard_count = 0;
    store->ram_used_bytes = 0;
}
