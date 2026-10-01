/**
 * expert_stream.c - Experts read from the drive with direct I/O into a cache
 * in memory. See expert_stream.h.
 */

/* glibc declares O_DIRECT only for _GNU_SOURCE, which must come before any
 * header. Without it a direct open would quietly read through the cache. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
  #define _GNU_SOURCE
#endif

#include "expert_stream.h"
#include "compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(VITNA_OS_WINDOWS)
  #include <errno.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

/* Direct I/O reads whole sectors into sector-aligned memory. 4096 is a
 * multiple of every sector size in use, 512 included. */
#define SECTOR ((uint64_t)4096)

/* --- Files opened for direct I/O --- */

typedef struct {
#if defined(VITNA_OS_WINDOWS)
    HANDLE h;
#else
    int fd;
#endif
} dfile_t;

static void dfile_init(dfile_t* f) {
#if defined(VITNA_OS_WINDOWS)
    f->h = INVALID_HANDLE_VALUE;
#else
    f->fd = -1;
#endif
}

static bool dfile_open(dfile_t* f, const char* path, char* why, size_t why_len) {
#if defined(VITNA_OS_WINDOWS)
    f->h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, NULL);
    if (f->h == INVALID_HANDLE_VALUE) {
        snprintf(why, why_len, "Windows error %lu", (unsigned long)GetLastError());
        return false;
    }
    return true;
#elif defined(O_DIRECT)
    f->fd = open(path, O_RDONLY | O_DIRECT);
    if (f->fd < 0) {
        snprintf(why, why_len, "%s", strerror(errno));
        return false;
    }
    return true;
#elif defined(F_NOCACHE)
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0 || fcntl(f->fd, F_NOCACHE, 1) != 0) {
        snprintf(why, why_len, "%s", strerror(errno));
        if (f->fd >= 0) close(f->fd);
        f->fd = -1;
        return false;
    }
    return true;
#else
    (void)path;
    snprintf(why, why_len, "this system has no direct I/O");
    return false;
#endif
}

static void dfile_close(dfile_t* f) {
#if defined(VITNA_OS_WINDOWS)
    if (f->h != INVALID_HANDLE_VALUE) CloseHandle(f->h);
    f->h = INVALID_HANDLE_VALUE;
#else
    if (f->fd >= 0) close(f->fd);
    f->fd = -1;
#endif
}

/* Read n bytes at off, both sector multiples, into sector-aligned buf.
 * Returns the bytes read, fewer at the end of the file, or -1. */
static int64_t dfile_read(dfile_t* f, void* buf, uint64_t n, uint64_t off) {
#if defined(VITNA_OS_WINDOWS)
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(off & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)(off >> 32);
    DWORD got = 0;
    if (!ReadFile(f->h, buf, (DWORD)n, &got, &ov) && GetLastError() != ERROR_HANDLE_EOF) return -1;
    return (int64_t)got;
#else
    uint64_t total = 0;
    while (total < n) {
        ssize_t r = pread(f->fd, (char*)buf + total, (size_t)(n - total), (off_t)(off + total));
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return -1;
        total += (uint64_t)r;
        /* Past a short read the next offset may not be a sector multiple. */
        if (r == 0 || total % SECTOR != 0) break;
    }
    return (int64_t)total;
#endif
}

/* --- The stream --- */

/* One run of whole sectors read from a file into a slot at dst. */
typedef struct {
    uint32_t file;
    uint64_t start;    /* a sector multiple */
    uint64_t length;   /* a sector multiple */
    uint64_t needed;   /* the bytes from start the place's parts reach to */
    size_t dst;        /* a sector multiple */
} run_t;

/* How a place is read, and where its parts land in a slot. */
typedef struct {
    run_t run[VITNA_EXPERT_PARTS_MAX];
    size_t n_runs;
    size_t part_at[VITNA_EXPERT_PARTS_MAX];
    size_t n_parts;
    uint64_t bytes;
} plan_t;

enum { EMPTY, READING, READY, FAILED };

typedef struct {
    int64_t place;     /* -1 when empty */
    int state;
    uint32_t users;
    uint64_t last_use;
    bool prefetched;   /* read for a prefetch, and not acquired since */
    unsigned char* buf;
} slot_t;

typedef struct {
    vitna_expert_stream_t* s;
    dfile_t* files;    /* this thread's own handles: one synchronous handle serializes its reads */
    vitna_thread_t thread;
    bool started;
} worker_t;

struct vitna_expert_stream {
    size_t n_files;
    plan_t* plans;
    size_t n_places;
    int64_t* where;    /* each place's slot, or -1 */
    uint32_t* uses;    /* each place's acquisitions */
    slot_t* slots;
    size_t n_slots;
    size_t slot_bytes;
    unsigned char* memory;
    size_t* queue;     /* slots waiting to be read, a ring of n_slots */
    size_t q_head, q_len;
    worker_t* workers;
    size_t n_workers;
    vitna_mutex_t lock;
    vitna_cond_t work; /* a read was queued, or the stream is closing */
    vitna_cond_t done; /* a read finished */
    bool closing;
    uint64_t clock;
    vitna_expert_stream_stats_t stats;
};

static void worker_main(void* arg) {
    worker_t* w = (worker_t*)arg;
    vitna_expert_stream_t* s = w->s;
    vitna_mutex_lock(&s->lock);
    for (;;) {
        while (s->q_len == 0 && !s->closing) vitna_cond_wait(&s->work, &s->lock);
        if (s->closing) break;
        const size_t i = s->queue[s->q_head];
        s->q_head = (s->q_head + 1) % s->n_slots;
        s->q_len--;
        /* A slot being read is never given up, so its place and plan stay put. */
        const plan_t* plan = &s->plans[s->slots[i].place];
        unsigned char* buf = s->slots[i].buf;
        vitna_mutex_unlock(&s->lock);

        const double t0 = vitna_time_ms();
        bool ok = true;
        uint64_t bytes = 0;
        for (size_t r = 0; r < plan->n_runs && ok; r++) {
            const run_t* run = &plan->run[r];
            const int64_t got = dfile_read(&w->files[run->file], buf + run->dst, run->length, run->start);
            ok = got >= 0 && (uint64_t)got >= run->needed;
            if (got > 0) bytes += (uint64_t)got;
        }
        const double t1 = vitna_time_ms();

        vitna_mutex_lock(&s->lock);
        s->slots[i].state = ok ? READY : FAILED;
        s->stats.reads += plan->n_runs;
        s->stats.bytes_read += bytes;
        s->stats.read_ms += t1 - t0;
        vitna_cond_broadcast(&s->done);
    }
    vitna_mutex_unlock(&s->lock);
}

static bool fail(char* err, size_t err_len, const char* msg) {
    if (err && err_len > 0) snprintf(err, err_len, "%s", msg);
    return false;
}

/* Sort a place's parts by file and offset, merge those that follow one
 * another into runs of whole sectors, and note where each part lands. */
static bool make_plan(const vitna_expert_place_t* place, size_t n_files, plan_t* plan, char* err, size_t err_len) {
    memset(plan, 0, sizeof(*plan));
    const size_t n = place->n_parts;
    if (n == 0 || n > VITNA_EXPERT_PARTS_MAX) return fail(err, err_len, "a place has no parts, or too many");
    size_t order[VITNA_EXPERT_PARTS_MAX];
    for (size_t i = 0; i < n; i++) {
        const vitna_extent_t* p = &place->part[i];
        if (p->file >= n_files || p->length == 0) return fail(err, err_len, "a part is empty, or in a file the stream lacks");
        size_t j = i;
        for (; j > 0; j--) {
            const vitna_extent_t* q = &place->part[order[j - 1]];
            if (q->file < p->file || (q->file == p->file && q->offset <= p->offset)) break;
            order[j] = order[j - 1];
        }
        order[j] = i;
    }
    uint64_t dst = 0;
    for (size_t k = 0; k < n;) {
        const vitna_extent_t* first = &place->part[order[k]];
        uint64_t end = first->offset + first->length;
        size_t m = k + 1;
        while (m < n && place->part[order[m]].file == first->file && place->part[order[m]].offset == end) {
            end += place->part[order[m]].length;
            m++;
        }
        run_t* run = &plan->run[plan->n_runs++];
        run->file = first->file;
        run->start = first->offset / SECTOR * SECTOR;
        run->length = (end + SECTOR - 1) / SECTOR * SECTOR - run->start;
        run->needed = end - run->start;
        run->dst = (size_t)dst;
        for (size_t q = k; q < m; q++) {
            plan->part_at[order[q]] = (size_t)(dst + (place->part[order[q]].offset - run->start));
        }
        dst += run->length;
        k = m;
    }
    plan->n_parts = n;
    plan->bytes = dst;
    return true;
}

vitna_expert_stream_t* vitna_expert_stream_open(const char* const* paths, size_t n_files, const vitna_expert_place_t* places,
                                                size_t n_places, size_t n_slots, size_t threads, char* err, size_t err_len) {
    if (n_files == 0 || n_places == 0 || threads == 0 || n_slots < 2) {
        fail(err, err_len, "a stream needs files, places, two slots or more, and a thread");
        return NULL;
    }
    vitna_expert_stream_t* s = (vitna_expert_stream_t*)calloc(1, sizeof(*s));
    if (!s) {
        fail(err, err_len, "out of memory");
        return NULL;
    }
    vitna_mutex_init(&s->lock);
    vitna_cond_init(&s->work);
    vitna_cond_init(&s->done);
    s->n_files = n_files;
    s->n_places = n_places;
    s->n_slots = n_slots;
    s->plans = (plan_t*)calloc(n_places, sizeof(plan_t));
    s->where = (int64_t*)malloc(n_places * sizeof(int64_t));
    s->uses = (uint32_t*)calloc(n_places, sizeof(uint32_t));
    s->slots = (slot_t*)calloc(n_slots, sizeof(slot_t));
    s->queue = (size_t*)malloc(n_slots * sizeof(size_t));
    s->workers = (worker_t*)calloc(threads, sizeof(worker_t));
    bool ok = s->plans && s->where && s->uses && s->slots && s->queue && s->workers;
    if (!ok) fail(err, err_len, "out of memory");
    for (size_t p = 0; ok && p < n_places; p++) {
        ok = make_plan(&places[p], n_files, &s->plans[p], err, err_len);
        if (ok && s->plans[p].bytes > s->slot_bytes) s->slot_bytes = (size_t)s->plans[p].bytes;
        s->where[p] = -1;
    }
    if (ok && s->slot_bytes > SIZE_MAX / n_slots) ok = fail(err, err_len, "the cache is larger than memory can address");
    if (ok) {
        s->memory = (unsigned char*)vitna_aligned_alloc((size_t)SECTOR, n_slots * s->slot_bytes);
        if (!s->memory) ok = fail(err, err_len, "out of memory for the expert cache");
    }
    for (size_t i = 0; ok && i < n_slots; i++) {
        s->slots[i].place = -1;
        s->slots[i].state = EMPTY;
        s->slots[i].buf = s->memory + i * s->slot_bytes;
    }
    for (size_t t = 0; ok && t < threads; t++) {
        worker_t* w = &s->workers[t];
        w->s = s;
        w->files = (dfile_t*)malloc(n_files * sizeof(dfile_t));
        if (!w->files) {
            ok = fail(err, err_len, "out of memory");
            break;
        }
        for (size_t f = 0; f < n_files; f++) dfile_init(&w->files[f]);
        s->n_workers = t + 1;
        for (size_t f = 0; ok && f < n_files; f++) {
            char why[160];
            if (!dfile_open(&w->files[f], paths[f], why, sizeof(why))) {
                if (err && err_len > 0) snprintf(err, err_len, "cannot open %s for direct I/O: %s", paths[f], why);
                ok = false;
            }
        }
    }
    for (size_t t = 0; ok && t < s->n_workers; t++) {
        worker_t* w = &s->workers[t];
        w->started = vitna_thread_start(&w->thread, worker_main, w, false);
        if (!w->started) ok = fail(err, err_len, "cannot start a thread to read experts");
    }
    if (!ok) {
        vitna_expert_stream_close(s);
        return NULL;
    }
    return s;
}

size_t vitna_expert_stream_slot_bytes(const vitna_expert_stream_t* s) {
    return s->slot_bytes;
}

size_t vitna_expert_stream_slot_bytes_for(const vitna_expert_place_t* places, size_t n_places, size_t n_files) {
    uint64_t most = 0;
    for (size_t p = 0; p < n_places; p++) {
        plan_t plan;
        if (!make_plan(&places[p], n_files, &plan, NULL, 0)) return 0;
        if (plan.bytes > most) most = plan.bytes;
    }
    return (size_t)most;
}

/* A slot nobody holds and nothing reads into: an empty one, or else the
 * least used, the least recently used between equals. -1 if there is none.
 * Called with the lock held. */
static int64_t free_slot(const vitna_expert_stream_t* s) {
    int64_t best = -1;
    for (size_t i = 0; i < s->n_slots; i++) {
        const slot_t* x = &s->slots[i];
        if (x->users || x->state == READING) continue;
        if (x->place < 0) return (int64_t)i;
        if (best < 0) {
            best = (int64_t)i;
            continue;
        }
        const slot_t* b = &s->slots[best];
        const uint32_t cx = s->uses[x->place], cb = s->uses[b->place];
        if (cx < cb || (cx == cb && x->last_use < b->last_use)) best = (int64_t)i;
    }
    return best;
}

/* Give slot i to place, and queue its read. Called with the lock held. */
static void start_read(vitna_expert_stream_t* s, size_t i, uint32_t place, bool prefetch) {
    slot_t* x = &s->slots[i];
    if (x->place >= 0) s->where[x->place] = -1;
    x->place = (int64_t)place;
    x->state = READING;
    x->prefetched = prefetch;
    x->last_use = ++s->clock;
    s->where[place] = (int64_t)i;
    s->queue[(s->q_head + s->q_len) % s->n_slots] = i;
    s->q_len++;
    vitna_cond_signal(&s->work);
}

/* Hold the k places, queueing reads for those the cache lacks. A place
 * whose prefetch failed is read again. Called with the lock held. */
static void hold(vitna_expert_stream_t* s, const uint32_t* ids, size_t k) {
    for (size_t i = 0; i < k; i++) {
        const uint32_t p = ids[i];
        int64_t x = s->where[p];
        s->stats.acquired++;
        s->uses[p]++;
        if (s->stats.acquired % 65536 == 0) {
            for (size_t q = 0; q < s->n_places; q++) s->uses[q] /= 2;
        }
        if (x < 0) {
            while ((x = free_slot(s)) < 0) vitna_cond_wait(&s->done, &s->lock);
            start_read(s, (size_t)x, p, false);
            s->stats.misses++;
        } else {
            slot_t* sl = &s->slots[x];
            if (sl->state == FAILED && sl->users == 0) {
                sl->state = READING;
                s->queue[(s->q_head + s->q_len) % s->n_slots] = (size_t)x;
                s->q_len++;
                vitna_cond_signal(&s->work);
                s->stats.misses++;
            } else if (sl->state == READING) {
                s->stats.in_flight++;
            } else {
                s->stats.hits++;
            }
            if (sl->prefetched) {
                s->stats.prefetch_used++;
                sl->prefetched = false;
            }
            sl->last_use = ++s->clock;
        }
        s->slots[x].users++;
    }
}

bool vitna_expert_stream_acquire(vitna_expert_stream_t* s, const uint32_t* ids, size_t k, vitna_expert_data_t* out) {
    vitna_mutex_lock(&s->lock);
    hold(s, ids, k);
    vitna_mutex_unlock(&s->lock);
    return vitna_expert_stream_wait(s, ids, k, out);
}

void vitna_expert_stream_hold(vitna_expert_stream_t* s, const uint32_t* ids, size_t k) {
    vitna_mutex_lock(&s->lock);
    hold(s, ids, k);
    vitna_mutex_unlock(&s->lock);
}

bool vitna_expert_stream_wait(vitna_expert_stream_t* s, const uint32_t* ids, size_t k, vitna_expert_data_t* out) {
    vitna_mutex_lock(&s->lock);
    double t0 = 0;
    bool waited = false, ok = true;
    for (size_t i = 0; i < k; i++) {
        const slot_t* sl = &s->slots[s->where[ids[i]]];
        while (sl->state == READING) {
            if (!waited) {
                t0 = vitna_time_ms();
                waited = true;
            }
            vitna_cond_wait(&s->done, &s->lock);
        }
        if (sl->state != READY) ok = false;
    }
    if (waited) s->stats.wait_ms += vitna_time_ms() - t0;
    if (!ok) {
        /* Let go of all of them; a slot whose read failed is emptied. */
        for (size_t i = 0; i < k; i++) {
            const int64_t x = s->where[ids[i]];
            slot_t* sl = &s->slots[x];
            if (sl->users) sl->users--;
            if (sl->state == FAILED && sl->users == 0) {
                s->where[ids[i]] = -1;
                sl->place = -1;
                sl->state = EMPTY;
                sl->prefetched = false;
            }
        }
        vitna_mutex_unlock(&s->lock);
        fprintf(stderr, "Reading an expert from the drive failed.\n");
        return false;
    }
    for (size_t i = 0; i < k; i++) {
        const slot_t* sl = &s->slots[s->where[ids[i]]];
        const plan_t* plan = &s->plans[ids[i]];
        for (size_t j = 0; j < plan->n_parts; j++) out[i].part[j] = sl->buf + plan->part_at[j];
    }
    vitna_mutex_unlock(&s->lock);
    return true;
}

void vitna_expert_stream_release(vitna_expert_stream_t* s, const uint32_t* ids, size_t k) {
    vitna_mutex_lock(&s->lock);
    for (size_t i = 0; i < k; i++) {
        const int64_t x = s->where[ids[i]];
        if (x >= 0 && s->slots[x].users) s->slots[x].users--;
    }
    vitna_mutex_unlock(&s->lock);
}

void vitna_expert_stream_prefetch(vitna_expert_stream_t* s, const uint32_t* ids, size_t k) {
    vitna_mutex_lock(&s->lock);
    for (size_t i = 0; i < k; i++) {
        if (s->where[ids[i]] >= 0) continue;
        const int64_t x = free_slot(s);
        if (x < 0) break;
        start_read(s, (size_t)x, ids[i], true);
        s->stats.prefetched++;
    }
    vitna_mutex_unlock(&s->lock);
}

vitna_expert_stream_stats_t vitna_expert_stream_stats(vitna_expert_stream_t* s) {
    vitna_mutex_lock(&s->lock);
    vitna_expert_stream_stats_t st = s->stats;
    vitna_mutex_unlock(&s->lock);
    return st;
}

void vitna_expert_stream_close(vitna_expert_stream_t* s) {
    if (!s) return;
    vitna_mutex_lock(&s->lock);
    s->closing = true;
    vitna_cond_broadcast(&s->work);
    vitna_mutex_unlock(&s->lock);
    for (size_t t = 0; t < s->n_workers; t++) {
        worker_t* w = &s->workers[t];
        if (w->started) vitna_thread_join(w->thread);
        if (w->files) {
            for (size_t f = 0; f < s->n_files; f++) dfile_close(&w->files[f]);
            free(w->files);
        }
    }
    vitna_aligned_free(s->memory);
    free(s->workers);
    free(s->queue);
    free(s->slots);
    free(s->uses);
    free(s->where);
    free(s->plans);
    vitna_cond_destroy(&s->work);
    vitna_cond_destroy(&s->done);
    vitna_mutex_destroy(&s->lock);
    free(s);
}
