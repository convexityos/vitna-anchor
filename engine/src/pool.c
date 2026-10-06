/**
 * pool.c - A few threads that share out a job's tasks (pool.h).
 *
 * One lock guards the job. The threads sleep on start until a job's number
 * changes, take tasks one at a time until none is left, and the last task to
 * finish wakes the caller, which runs tasks too while it waits.
 */

#include "pool.h"
#include "compat.h"
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    vitna_pool_t* pool;
    size_t worker;
} worker_arg_t;

struct vitna_pool {
    vitna_mutex_t lock;
    vitna_cond_t start, done;
    vitna_thread_t* threads; /* the threads started, the caller not among them */
    worker_arg_t* args;
    size_t n_started;
    size_t n_threads;        /* the caller counted */

    uint64_t job;            /* the job's number, one more each run */
    vitna_pool_task_t task;
    void* arg;
    size_t n, next, finished;
    bool stop;
};

/* Run the job's tasks that are left, as worker. The lock is held on entry
 * and on return, and let go around each task. */
static void work(vitna_pool_t* p, size_t worker) {
    while (p->next < p->n) {
        const size_t i = p->next++;
        vitna_mutex_unlock(&p->lock);
        p->task(p->arg, i, worker);
        vitna_mutex_lock(&p->lock);
        if (++p->finished == p->n) vitna_cond_broadcast(&p->done);
    }
}

static void worker_main(void* v) {
    const worker_arg_t* a = (const worker_arg_t*)v;
    vitna_pool_t* p = a->pool;
    uint64_t seen = 0;
    vitna_mutex_lock(&p->lock);
    for (;;) {
        while (!p->stop && p->job == seen) vitna_cond_wait(&p->start, &p->lock);
        if (p->stop) break;
        seen = p->job;
        work(p, a->worker);
    }
    vitna_mutex_unlock(&p->lock);
}

vitna_pool_t* vitna_pool_create(size_t threads) {
    if (threads == 0) return NULL;
    vitna_pool_t* p = (vitna_pool_t*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    vitna_mutex_init(&p->lock);
    vitna_cond_init(&p->start);
    vitna_cond_init(&p->done);
    p->n_threads = threads;
    if (threads > 1) {
        p->threads = (vitna_thread_t*)calloc(threads - 1, sizeof(vitna_thread_t));
        p->args = (worker_arg_t*)calloc(threads - 1, sizeof(worker_arg_t));
        if (!p->threads || !p->args) {
            vitna_pool_free(p);
            return NULL;
        }
        for (size_t i = 0; i + 1 < threads; i++) {
            p->args[i].pool = p;
            p->args[i].worker = i + 1;
            if (!vitna_thread_start(&p->threads[i], worker_main, &p->args[i], false)) {
                vitna_pool_free(p);
                return NULL;
            }
            p->n_started++;
        }
    }
    return p;
}

size_t vitna_pool_threads(const vitna_pool_t* p) {
    return p ? p->n_threads : 0;
}

void vitna_pool_run(vitna_pool_t* p, size_t n, vitna_pool_task_t task, void* arg) {
    if (n == 0) return;
    vitna_mutex_lock(&p->lock);
    p->task = task;
    p->arg = arg;
    p->n = n;
    p->next = 0;
    p->finished = 0;
    p->job++;
    if (p->n_started) vitna_cond_broadcast(&p->start);
    work(p, 0);
    while (p->finished < p->n) vitna_cond_wait(&p->done, &p->lock);
    vitna_mutex_unlock(&p->lock);
}

void vitna_pool_free(vitna_pool_t* p) {
    if (!p) return;
    vitna_mutex_lock(&p->lock);
    p->stop = true;
    vitna_cond_broadcast(&p->start);
    vitna_mutex_unlock(&p->lock);
    for (size_t i = 0; i < p->n_started; i++) vitna_thread_join(p->threads[i]);
    vitna_cond_destroy(&p->start);
    vitna_cond_destroy(&p->done);
    vitna_mutex_destroy(&p->lock);
    free(p->threads);
    free(p->args);
    free(p);
}
