/**
 * expert_stream.h - A mixture of experts' experts, read from the drive as
 * the forward pass needs them, into a cache in memory of a fixed size.
 *
 * Each expert is read into a slot of the cache with direct I/O: the
 * operating system's file cache is bypassed (FILE_FLAG_NO_BUFFERING on
 * Windows, O_DIRECT on Linux, F_NOCACHE on macOS), so a read is a read from
 * the drive, and the experts in memory are the ones the cache holds, however
 * large the checkpoint. Direct I/O reads whole sectors into aligned memory,
 * so each piece of an expert is read from the 4096-byte boundary before it to
 * the one after it, and its data begins inside that.
 *
 * Reads run on threads of their own, each with its own handles to the files,
 * so the experts a layer lacks are read at the same time, not one after
 * another. The forward pass acquires the experts a layer routes to: those
 * the cache holds come back at once, the rest when read. An acquired expert
 * keeps its slot until it is released. When a slot is wanted, the one given
 * up is the least used of those nobody holds, the least recently used
 * between equals; every 65,536 acquisitions halve the counts, so that what
 * was used long ago weighs less than what is used now. Least used beats
 * least recently used here because a token visits every layer in turn: with
 * fewer slots than a token's experts, least recently used gives up each
 * expert just before its layer comes round again.
 *
 * prefetch starts reading experts the cache lacks and returns without
 * waiting: a guess at what a later layer will want, which costs a read and a
 * slot when it is wrong, and changes nothing that is computed. A slot being
 * read is never given up, so a prefetch keeps its slot at least until its
 * read is done.
 */

#ifndef VITNA_EXPERT_STREAM_H
#define VITNA_EXPERT_STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** A run of bytes in one of the stream's files. */
typedef struct {
    uint32_t file;
    uint64_t offset;
    uint64_t length;
} vitna_extent_t;

#define VITNA_EXPERT_PARTS_MAX 3

/** Where one expert is: up to three parts, such as its gate, up and down matrices. */
typedef struct {
    vitna_extent_t part[VITNA_EXPERT_PARTS_MAX];
    size_t n_parts;
} vitna_expert_place_t;

/** Where an acquired expert's parts are in memory, in the order of its place's parts. */
typedef struct {
    const void* part[VITNA_EXPERT_PARTS_MAX];
} vitna_expert_data_t;

typedef struct {
    uint64_t acquired;       /* experts acquired */
    uint64_t hits;           /* of those, already read when acquired */
    uint64_t in_flight;      /* still being read, for a prefetch, when acquired */
    uint64_t misses;         /* not in the cache: read for the acquisition */
    uint64_t prefetched;     /* reads a prefetch started */
    uint64_t prefetch_used;  /* of those, acquired before their slot was given up */
    uint64_t reads;          /* reads done, each one sector-aligned run */
    uint64_t bytes_read;
    double read_ms;          /* time the reads took, summed over the threads */
    double wait_ms;          /* time acquire waited for reads */
} vitna_expert_stream_stats_t;

typedef struct vitna_expert_stream vitna_expert_stream_t;

/**
 * Open the n_files files at paths for direct I/O, on threads threads, with a
 * cache of n_slots slots, each large enough for any of the n_places places.
 * A place's parts that follow one another in a file are read as one run.
 * Returns NULL, with the reason in err, if a file cannot be opened for
 * direct I/O, the memory cannot be had, or a part is empty.
 */
vitna_expert_stream_t* vitna_expert_stream_open(const char* const* paths, size_t n_files, const vitna_expert_place_t* places,
                                                size_t n_places, size_t n_slots, size_t threads, char* err, size_t err_len);

/** The bytes a slot takes: a place's runs, each widened to whole 4096-byte sectors. */
size_t vitna_expert_stream_slot_bytes(const vitna_expert_stream_t* s);

/** The bytes a slot would take for these places, before opening a stream; 0 if a place is not valid. */
size_t vitna_expert_stream_slot_bytes_for(const vitna_expert_place_t* places, size_t n_places, size_t n_files);

/**
 * Make the k places resident, read them where needed, and set out[i] to
 * where place ids[i]'s parts are. They stay where they are until released.
 * Blocks until every one is read. Returns false, with the reason on stderr,
 * if a read fails, which acquires none of them. k must be at most half the
 * slots, so that what one caller holds never starves its own next read. It
 * is hold, then wait.
 */
bool vitna_expert_stream_acquire(vitna_expert_stream_t* s, const uint32_t* ids, size_t k, vitna_expert_data_t* out);

/**
 * The first half of acquire: hold the k places, so no prefetch can give
 * their slots up, and queue reads for those the cache lacks, without
 * waiting for them. Waits only when every slot is held or being read.
 */
void vitna_expert_stream_hold(vitna_expert_stream_t* s, const uint32_t* ids, size_t k);

/** The second half: wait for k held places to be read, as acquire does. */
bool vitna_expert_stream_wait(vitna_expert_stream_t* s, const uint32_t* ids, size_t k, vitna_expert_data_t* out);

/** Release the k places acquire returned. */
void vitna_expert_stream_release(vitna_expert_stream_t* s, const uint32_t* ids, size_t k);

/**
 * Hold place id only if the cache has it read already, setting out to where
 * its parts are, and return true; otherwise return false, starting no read
 * and waiting for nothing. A place held so is released as one acquired is,
 * but it is not counted as acquired: it is for copying an expert onward (to
 * a GPU) before anyone asks for it.
 */
bool vitna_expert_stream_hold_ready(vitna_expert_stream_t* s, uint32_t id, vitna_expert_data_t* out);

/** The cache's slots. */
size_t vitna_expert_stream_slots(const vitna_expert_stream_t* s);

/**
 * The memory the slots are in, one block of *bytes, for registering it with
 * a device that copies from it.
 */
void* vitna_expert_stream_memory(const vitna_expert_stream_t* s, size_t* bytes);

/**
 * Start reading those of the k places the cache lacks, into slots nobody
 * holds, and return without waiting. Stops early, dropping the rest, when no
 * slot can be given up.
 */
void vitna_expert_stream_prefetch(vitna_expert_stream_t* s, const uint32_t* ids, size_t k);

/** The counts so far. */
vitna_expert_stream_stats_t vitna_expert_stream_stats(vitna_expert_stream_t* s);

/** Stop the threads, close the files, free the cache. Nothing may be held. */
void vitna_expert_stream_close(vitna_expert_stream_t* s);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_EXPERT_STREAM_H */
