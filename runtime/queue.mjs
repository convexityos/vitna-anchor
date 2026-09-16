// Priority concurrency scheduler for the Vitna runtime.
//
// Regulates in-flight requests to eliminate upstream 429 rate limit storms
// and prevent local engine hardware thrashing.
//
// Rules:
// - Zero external runtime dependencies
// - No em-dashes anywhere in code or comments
// - Pure priority FIFO dispatching with queue wait measurement

/**
 * Priority levels for incoming traffic.
 */
export const PRIORITY = {
  INTERACTIVE_STREAM: 10,
  STANDARD_COMPLETION: 5,
  BACKGROUND_BATCH: 1,
};

/**
 * Create a priority concurrency queue.
 *
 * @param {{
 *   maxConcurrency?: number,
 *   now?: () => number,
 * }} [options]
 */
export function createPriorityQueue(options = {}) {
  const maxConcurrency = options.maxConcurrency && options.maxConcurrency > 0
    ? options.maxConcurrency
    : 0; // 0 means unbounded
  const now = options.now ?? (() => Date.now());

  let inFlight = 0;
  let totalProcessed = 0;
  let totalWaitMs = 0;

  /** @type {Array<{ fn: () => Promise<any>, priority: number, enqueuedAt: number, resolve: (v: any) => void, reject: (err: any) => void }>} */
  const queue = [];

  function pump() {
    if (maxConcurrency > 0 && inFlight >= maxConcurrency) {
      return;
    }
    if (queue.length === 0) {
      return;
    }

    // Dequeue highest priority item (stable FIFO among same priority)
    let bestIndex = 0;
    for (let i = 1; i < queue.length; i++) {
      if (queue[i].priority > queue[bestIndex].priority) {
        bestIndex = i;
      }
    }

    const [item] = queue.splice(bestIndex, 1);
    const waitMs = Math.max(0, now() - item.enqueuedAt);
    totalWaitMs += waitMs;

    inFlight++;
    Promise.resolve()
      .then(() => item.fn())
      .then((result) => {
        inFlight--;
        totalProcessed++;
        item.resolve({ result, waitMs });
        pump();
      })
      .catch((err) => {
        inFlight--;
        totalProcessed++;
        item.reject(err);
        pump();
      });
  }

  return {
    /**
     * Run a task through the concurrency queue.
     *
     * @template T
     * @param {() => Promise<T>} fn
     * @param {number} [priority]
     * @returns {Promise<{ result: T, waitMs: number }>}
     */
    run(fn, priority = PRIORITY.STANDARD_COMPLETION) {
      if (maxConcurrency <= 0) {
        // Unbounded fast-path
        return Promise.resolve().then(fn).then((result) => ({ result, waitMs: 0 }));
      }

      return new Promise((resolve, reject) => {
        queue.push({
          fn,
          priority,
          enqueuedAt: now(),
          resolve,
          reject,
        });
        pump();
      });
    },

    /**
     * Queue metrics.
     */
    stats() {
      return {
        inFlight,
        pending: queue.length,
        maxConcurrency,
        totalProcessed,
        avgWaitMs: totalProcessed > 0 ? Math.round(totalWaitMs / totalProcessed) : 0,
      };
    },
  };
}
