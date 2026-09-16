/**
 * compat.h - Cross-platform OS abstraction for vitna-engine.
 *
 * Provides unified primitives for:
 * - High-resolution monotonic timers
 * - Memory-mapped file I/O (mmap on POSIX, MapViewOfFile on Windows)
 * - Positional file reads (pread on POSIX, overlapped ReadFile on Windows)
 * - Cache-line aligned dynamic memory allocation
 * - Lightweight mutual exclusion and thread pools
 */

#ifndef VITNA_COMPAT_H
#define VITNA_COMPAT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
  #define VITNA_OS_WINDOWS 1
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <io.h>
#else
  #define VITNA_OS_POSIX 1
  #define _GNU_SOURCE
  #include <stdlib.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <pthread.h>
  #include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- Monotonic Timer --- */

static inline uint64_t vitna_time_nanos(void) {
#if defined(VITNA_OS_WINDOWS)
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static inline double vitna_time_ms(void) {
    return (double)vitna_time_nanos() / 1000000.0;
}

/* --- Aligned Memory Allocation (e.g. 64-byte AVX-512 boundaries) --- */

static inline void* vitna_aligned_alloc(size_t alignment, size_t size) {
#if defined(VITNA_OS_WINDOWS)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static inline void vitna_aligned_free(void* ptr) {
    if (!ptr) return;
#if defined(VITNA_OS_WINDOWS)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* --- Memory-Mapped File Handle --- */

typedef struct {
    void* data;
    size_t size;
#if defined(VITNA_OS_WINDOWS)
    HANDLE file_handle;
    HANDLE map_handle;
#else
    int fd;
#endif
} vitna_mmap_t;

static inline bool vitna_mmap_open(const char* path, vitna_mmap_t* out) {
    if (!path || !out) return false;
    out->data = NULL;
    out->size = 0;

#if defined(VITNA_OS_WINDOWS)
    out->file_handle = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (out->file_handle == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(out->file_handle, &file_size)) {
        CloseHandle(out->file_handle);
        return false;
    }
    out->size = (size_t)file_size.QuadPart;

    out->map_handle = CreateFileMappingA(
        out->file_handle,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );
    if (!out->map_handle) {
        CloseHandle(out->file_handle);
        return false;
    }

    out->data = MapViewOfFile(
        out->map_handle,
        FILE_MAP_READ,
        0,
        0,
        out->size
    );
    if (!out->data) {
        CloseHandle(out->map_handle);
        CloseHandle(out->file_handle);
        return false;
    }
    return true;

#else
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) return false;

    struct stat st;
    if (fstat(out->fd, &st) < 0) {
        close(out->fd);
        return false;
    }
    out->size = (size_t)st.st_size;

    out->data = mmap(NULL, out->size, PROT_READ, MAP_SHARED, out->fd, 0);
    if (out->data == MAP_FAILED) {
        out->data = NULL;
        close(out->fd);
        return false;
    }
    return true;
#endif
}

static inline void vitna_mmap_close(vitna_mmap_t* m) {
    if (!m) return;
#if defined(VITNA_OS_WINDOWS)
    if (m->data) UnmapViewOfFile(m->data);
    if (m->map_handle) CloseHandle(m->map_handle);
    if (m->file_handle != INVALID_HANDLE_VALUE) CloseHandle(m->file_handle);
    m->data = NULL;
    m->map_handle = NULL;
    m->file_handle = INVALID_HANDLE_VALUE;
#else
    if (m->data && m->size > 0) munmap(m->data, m->size);
    if (m->fd >= 0) close(m->fd);
    m->data = NULL;
    m->fd = -1;
#endif
    m->size = 0;
}

/* --- Positional File Read (for async NVMe expert streaming) --- */

typedef struct {
#if defined(VITNA_OS_WINDOWS)
    HANDLE handle;
#else
    int fd;
#endif
} vitna_file_t;

static inline bool vitna_file_open_read(const char* path, bool direct_io, vitna_file_t* out) {
    if (!path || !out) return false;
#if defined(VITNA_OS_WINDOWS)
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (direct_io) {
        flags |= FILE_FLAG_NO_BUFFERING;
    }
    out->handle = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        flags,
        NULL
    );
    return out->handle != INVALID_HANDLE_VALUE;
#else
    int flags = O_RDONLY;
  #ifdef O_DIRECT
    if (direct_io) flags |= O_DIRECT;
  #endif
    out->fd = open(path, flags);
    return out->fd >= 0;
#endif
}

static inline int64_t vitna_file_pread(vitna_file_t* f, void* buffer, size_t bytes, uint64_t offset) {
#if defined(VITNA_OS_WINDOWS)
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(offset >> 32);

    DWORD bytes_read = 0;
    if (!ReadFile(f->handle, buffer, (DWORD)bytes, &bytes_read, &ov)) {
        DWORD err = GetLastError();
        if (err != ERROR_HANDLE_EOF) return -1;
    }
    return (int64_t)bytes_read;
#else
    return (int64_t)pread(f->fd, buffer, bytes, (off_t)offset);
#endif
}

static inline void vitna_file_close(vitna_file_t* f) {
    if (!f) return;
#if defined(VITNA_OS_WINDOWS)
    if (f->handle != INVALID_HANDLE_VALUE) {
        CloseHandle(f->handle);
        f->handle = INVALID_HANDLE_VALUE;
    }
#else
    if (f->fd >= 0) {
        close(f->fd);
        f->fd = -1;
    }
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* VITNA_COMPAT_H */
