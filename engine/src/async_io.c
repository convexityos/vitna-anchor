/**
 * async_io.c - High-throughput asynchronous NVMe streaming engine.
 */

#include "async_io.h"
#include <stdlib.h>
#include <string.h>

#if defined(VITNA_OS_WINDOWS)

typedef struct {
    OVERLAPPED ov;
    vitna_io_completion_t info;
} vitna_overlapped_ext_t;

bool vitna_async_io_init(vitna_async_pool_t* pool, size_t max_queue_depth) {
    if (!pool) return false;
    memset(pool, 0, sizeof(*pool));
    pool->max_queue_depth = max_queue_depth > 0 ? max_queue_depth : 64;

    pool->iocp_port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    return pool->iocp_port != NULL;
}

bool vitna_async_io_submit(
    vitna_async_pool_t* pool,
    vitna_file_t* file,
    void* aligned_buffer,
    size_t bytes,
    uint64_t offset,
    void* user_tag
) {
    if (!pool || !file || file->handle == INVALID_HANDLE_VALUE || !aligned_buffer || bytes == 0) {
        return false;
    }

    /* Bind file handle to IOCP port (safe to call multiple times with same key) */
    HANDLE bound = CreateIoCompletionPort(file->handle, pool->iocp_port, (ULONG_PTR)file, 0);
    if (!bound) {
        DWORD err = GetLastError();
        if (err != ERROR_INVALID_PARAMETER) { /* ERROR_INVALID_PARAMETER if already associated */
            return false;
        }
    }

    vitna_overlapped_ext_t* ext = (vitna_overlapped_ext_t*)calloc(1, sizeof(vitna_overlapped_ext_t));
    if (!ext) return false;

    ext->ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
    ext->ov.OffsetHigh = (DWORD)(offset >> 32);

    ext->info.buffer = aligned_buffer;
    ext->info.bytes_requested = bytes;
    ext->info.file_offset = offset;
    ext->info.user_tag = user_tag;

    DWORD bytes_read = 0;
    BOOL ok = ReadFile(file->handle, aligned_buffer, (DWORD)bytes, &bytes_read, &ext->ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            free(ext);
            return false;
        }
    }

    pool->in_flight_count++;
    return true;
}

size_t vitna_async_io_poll(
    vitna_async_pool_t* pool,
    uint32_t timeout_ms,
    vitna_io_completion_t* out_completions,
    size_t max_completions
) {
    if (!pool || !out_completions || max_completions == 0 || pool->in_flight_count == 0) {
        return 0;
    }

    size_t completed = 0;
    DWORD start_t = GetTickCount();

    while (completed < max_completions && pool->in_flight_count > 0) {
        DWORD bytes_transferred = 0;
        ULONG_PTR completion_key = 0;
        LPOVERLAPPED lp_ov = NULL;

        DWORD remaining_ms = timeout_ms;
        if (timeout_ms != INFINITE) {
            DWORD elapsed = GetTickCount() - start_t;
            if (elapsed >= timeout_ms) break;
            remaining_ms = timeout_ms - elapsed;
        }

        BOOL ok = GetQueuedCompletionStatus(
            pool->iocp_port,
            &bytes_transferred,
            &completion_key,
            &lp_ov,
            completed == 0 ? remaining_ms : 0
        );

        if (!lp_ov) {
            break; /* Timeout or no events */
        }

        vitna_overlapped_ext_t* ext = (vitna_overlapped_ext_t*)lp_ov;
        ext->info.bytes_transferred = (int64_t)bytes_transferred;
        ext->info.success = ok ? true : false;

        out_completions[completed++] = ext->info;
        free(ext);
        pool->in_flight_count--;
    }

    return completed;
}

void vitna_async_io_destroy(vitna_async_pool_t* pool) {
    if (!pool) return;
    if (pool->iocp_port) {
        CloseHandle(pool->iocp_port);
        pool->iocp_port = NULL;
    }
    pool->in_flight_count = 0;
}

#else

/* POSIX synchronous pread fallback implementation */

bool vitna_async_io_init(vitna_async_pool_t* pool, size_t max_queue_depth) {
    if (!pool) return false;
    memset(pool, 0, sizeof(*pool));
    pool->max_queue_depth = max_queue_depth;
    pool->epoll_or_ring_fd = 0;
    return true;
}

bool vitna_async_io_submit(
    vitna_async_pool_t* pool,
    vitna_file_t* file,
    void* aligned_buffer,
    size_t bytes,
    uint64_t offset,
    void* user_tag
) {
    if (!pool || !file || file->fd < 0 || !aligned_buffer) return false;
    int64_t rd = vitna_file_pread(file, aligned_buffer, bytes, offset);

    /* Immediately complete in synchronous fallback mode */
    pool->in_flight_count++;
    return rd >= 0;
}

size_t vitna_async_io_poll(
    vitna_async_pool_t* pool,
    uint32_t timeout_ms,
    vitna_io_completion_t* out_completions,
    size_t max_completions
) {
    (void)timeout_ms;
    if (!pool || !out_completions || max_completions == 0 || pool->in_flight_count == 0) return 0;

    /* Drain in-flight count */
    size_t completed = pool->in_flight_count < max_completions ? pool->in_flight_count : max_completions;
    pool->in_flight_count -= completed;
    return completed;
}

void vitna_async_io_destroy(vitna_async_pool_t* pool) {
    if (!pool) return;
    pool->in_flight_count = 0;
}

#endif
