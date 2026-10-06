/**
 * pool.h - A few threads that share out a job's tasks, the caller among them.
 *
 * vitna_pool_run hands tasks 0 to n - 1 of a job to the pool's threads and
 * to the thread that calls it, each task to one of them, and returns once
 * every task has run. Which thread runs which task, and when, is not fixed,
 * so a job's tasks must not depend on one another: the GPU's step gives it
 * the rows of the experts the device lacks (gate A8), each of which is
 * computed alone, so the values are the same however the tasks fall.
 */

#ifndef VITNA_POOL_H
#define VITNA_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vitna_pool vitna_pool_t;

/**
 * A task: task i of the job, run by thread worker of the pool's
 * vitna_pool_threads, the caller being worker 0, so that each thread can
 * have scratch of its own.
 */
typedef void (*vitna_pool_task_t)(void* arg, size_t i, size_t worker);

/**
 * A pool of threads threads in all, the caller of vitna_pool_run counted
 * among them: 1 starts none, and runs every task on the caller. NULL if
 * threads is 0, memory cannot be had or a thread cannot start.
 */
vitna_pool_t* vitna_pool_create(size_t threads);

/** The threads that run a job's tasks, the caller counted. */
size_t vitna_pool_threads(const vitna_pool_t* p);

/** Run tasks 0 to n - 1 of task, with arg, and return once all have run. One job at a time. */
void vitna_pool_run(vitna_pool_t* p, size_t n, vitna_pool_task_t task, void* arg);

/** Stop the threads and free the pool. NULL is ignored. */
void vitna_pool_free(vitna_pool_t* p);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_POOL_H */
