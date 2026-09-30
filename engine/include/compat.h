/**
 * compat.h - Cross-platform OS abstraction for vitna-engine.
 *
 * Provides unified primitives for:
 * - High-resolution monotonic timers
 * - Memory-mapped file I/O (mmap on POSIX, MapViewOfFile on Windows)
 * - Positional file reads (pread on POSIX, overlapped ReadFile on Windows)
 * - Cache-line aligned dynamic memory allocation
 * - Threads, locks and condition variables
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
  #include <stdlib.h>
  #include <malloc.h>
#else
  #define VITNA_OS_POSIX 1
  /* g++, which nvcc hands model_cuda.cu to, defines it already. */
  #ifndef _GNU_SOURCE
    #define _GNU_SOURCE
  #endif
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

/* Nanoseconds in ticks of a counter running at freq ticks a second. Whole
 * seconds and the remainder are taken separately: ticks * 1e9 overflows 64
 * bits once ticks passes 2^64 / 1e9, which for Windows' usual 10 MHz
 * performance counter is 31 minutes after boot, and the clock then wrapped
 * every 31 minutes. The remainder is below freq, so remainder * 1e9 fits for
 * any freq up to about 18 GHz. */
static inline uint64_t vitna_ticks_to_nanos(uint64_t ticks, uint64_t freq) {
    return (ticks / freq) * 1000000000ULL + ((ticks % freq) * 1000000000ULL) / freq;
}

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
    return vitna_ticks_to_nanos((uint64_t)counter.QuadPart, (uint64_t)freq.QuadPart);
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
  #else
    (void)direct_io;
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

/* --- Threads, locks and condition variables --- */

#if defined(VITNA_OS_WINDOWS)
typedef SRWLOCK vitna_mutex_t;
typedef CONDITION_VARIABLE vitna_cond_t;
typedef HANDLE vitna_thread_t;
static inline void vitna_mutex_init(vitna_mutex_t* m) { InitializeSRWLock(m); }
static inline void vitna_mutex_destroy(vitna_mutex_t* m) { (void)m; }
static inline void vitna_mutex_lock(vitna_mutex_t* m) { AcquireSRWLockExclusive(m); }
static inline void vitna_mutex_unlock(vitna_mutex_t* m) { ReleaseSRWLockExclusive(m); }
static inline void vitna_cond_init(vitna_cond_t* c) { InitializeConditionVariable(c); }
static inline void vitna_cond_destroy(vitna_cond_t* c) { (void)c; }
static inline void vitna_cond_wait(vitna_cond_t* c, vitna_mutex_t* m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static inline void vitna_cond_signal(vitna_cond_t* c) { WakeConditionVariable(c); }
static inline void vitna_cond_broadcast(vitna_cond_t* c) { WakeAllConditionVariable(c); }
#else
typedef pthread_mutex_t vitna_mutex_t;
typedef pthread_cond_t vitna_cond_t;
typedef pthread_t vitna_thread_t;
static inline void vitna_mutex_init(vitna_mutex_t* m) { pthread_mutex_init(m, NULL); }
static inline void vitna_mutex_destroy(vitna_mutex_t* m) { pthread_mutex_destroy(m); }
static inline void vitna_mutex_lock(vitna_mutex_t* m) { pthread_mutex_lock(m); }
static inline void vitna_mutex_unlock(vitna_mutex_t* m) { pthread_mutex_unlock(m); }
static inline void vitna_cond_init(vitna_cond_t* c) { pthread_cond_init(c, NULL); }
static inline void vitna_cond_destroy(vitna_cond_t* c) { pthread_cond_destroy(c); }
static inline void vitna_cond_wait(vitna_cond_t* c, vitna_mutex_t* m) { pthread_cond_wait(c, m); }
static inline void vitna_cond_signal(vitna_cond_t* c) { pthread_cond_signal(c); }
static inline void vitna_cond_broadcast(vitna_cond_t* c) { pthread_cond_broadcast(c); }
#endif

typedef struct {
    void (*fn)(void*);
    void* arg;
} vitna_thread_call_t;

#if defined(VITNA_OS_WINDOWS)
static inline DWORD WINAPI vitna_thread_entry(LPVOID p) {
#else
static inline void* vitna_thread_entry(void* p) {
#endif
    vitna_thread_call_t call = *(vitna_thread_call_t*)p;
    free(p);
    call.fn(call.arg);
    return 0;
}

/**
 * Run fn(arg) on a thread of its own. A detached thread is never joined: it
 * frees what it holds when fn returns. Returns false, with fn not run, if no
 * thread could start.
 */
static inline bool vitna_thread_start(vitna_thread_t* t, void (*fn)(void*), void* arg, bool detached) {
    vitna_thread_call_t* call = (vitna_thread_call_t*)malloc(sizeof(vitna_thread_call_t));
    if (!call) return false;
    call->fn = fn;
    call->arg = arg;
#if defined(VITNA_OS_WINDOWS)
    HANDLE h = CreateThread(NULL, 0, vitna_thread_entry, call, 0, NULL);
    if (!h) {
        free(call);
        return false;
    }
    if (detached) CloseHandle(h);
    else *t = h;
#else
    pthread_t h;
    if (pthread_create(&h, NULL, vitna_thread_entry, call) != 0) {
        free(call);
        return false;
    }
    if (detached) pthread_detach(h);
    else *t = h;
#endif
    return true;
}

/** Wait for a thread vitna_thread_start started undetached to return. */
static inline void vitna_thread_join(vitna_thread_t t) {
#if defined(VITNA_OS_WINDOWS)
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
#else
    pthread_join(t, NULL);
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* VITNA_COMPAT_H */
