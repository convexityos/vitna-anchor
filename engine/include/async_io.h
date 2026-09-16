/**
 * async_io.h - High-throughput asynchronous NVMe streaming engine.
 *
 * Implements:
 * - Windows I/O Completion Ports (IOCP) with non-buffered page-aligned I/O
 * - POSIX multi-threaded pread / io_uring event completion queue
 * - Concurrent multi-drive striped dispatch
 */

#ifndef VITNA_ASYNC_IO_H
#define VITNA_ASYNC_IO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "compat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* buffer;
    size_t bytes_requested;
    int64_t bytes_transferred;
    uint64_t file_offset;
    void* user_tag;
    bool success;
} vitna_io_completion_t;

typedef struct {
#if defined(VITNA_OS_WINDOWS)
    HANDLE iocp_port;
#else
    int epoll_or_ring_fd;
#endif
    size_t in_flight_count;
    size_t max_queue_depth;
} vitna_async_pool_t;

/**
 * Initialize the asynchronous I/O completion pool.
 */
bool vitna_async_io_init(vitna_async_pool_t* pool, size_t max_queue_depth);

/**
 * Enqueue an asynchronous positional read operation from a file shard.
 */
bool vitna_async_io_submit(
    vitna_async_pool_t* pool,
    vitna_file_t* file,
    void* aligned_buffer,
    size_t bytes,
    uint64_t offset,
    void* user_tag
);

/**
 * Wait for one or more I/O operations to complete, up to timeout_ms.
 * Returns the number of completed operations written into out_completions.
 */
size_t vitna_async_io_poll(
    vitna_async_pool_t* pool,
    uint32_t timeout_ms,
    vitna_io_completion_t* out_completions,
    size_t max_completions
);

/**
 * Destroy the pool and release all OS handles.
 */
void vitna_async_io_destroy(vitna_async_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_ASYNC_IO_H */
