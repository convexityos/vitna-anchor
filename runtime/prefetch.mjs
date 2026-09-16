// prefetch.mjs - Async overlapped NVMe DMA expert pre-fetching pipeline.
//
// Simulates and coordinates non-blocking direct I/O DMA expert reads one layer
// ahead of attention forward passes, mirroring engine/include/async_io.h and
// the C11 vitna_expert_store_prefetch API.
//
// Rules:
// - Zero external runtime dependencies
// - Strictly zero em-dashes anywhere in comments, code, or strings
// - Realistic NVMe unbuffered DMA sector read latency and queue depth simulation

/**
 * Priority queue for asynchronous unbuffered DMA expert page-in requests.
 */
export class PrefetchQueue {
  /**
   * @param {{
   *   maxQueueDepth?: number,
   *   nvmeBandwidthMBps?: number,
   *   randomReadLatencyMs?: number,
   * }} options
   */
  constructor(options = {}) {
    this.maxQueueDepth = options.maxQueueDepth || 8;
    this.bandwidthMBps = options.nvmeBandwidthMBps || 3200;
    this.baseLatencyMs = options.randomReadLatencyMs || 0.08;
    this.queue = [];
    this.inFlight = new Map();
    this.completed = new Set();
    this.stats = {
      totalDispatched: 0,
      totalCompleted: 0,
      latencyHiddenMs: 0.0,
      stallsMs: 0.0,
      overlapEvents: 0,
    };
  }

  /**
   * Enqueue an expert prefetch request if not already resident or in-flight.
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @param {number} sizeBytes
   * @returns {boolean} True if scheduled, false if already queued or resident
   */
  enqueue(layerIdx, expertIdx, sizeBytes = 8 * 1024 * 1024) {
    const key = `${layerIdx}:${expertIdx}`;
    if (this.completed.has(key) || this.inFlight.has(key)) {
      return false;
    }

    if (this.queue.length >= this.maxQueueDepth) {
      // Queue saturated: oldest request dropped or caller waits
      return false;
    }

    const estIoTimeMs = this.baseLatencyMs + (sizeBytes / (1024 * 1024)) / (this.bandwidthMBps / 1000.0);
    const req = {
      key,
      layerIdx,
      expertIdx,
      sizeBytes,
      estIoTimeMs,
      enqueuedAt: performance.now(),
    };

    this.queue.push(req);
    this.inFlight.set(key, req);
    this.stats.totalDispatched++;
    return true;
  }

  /**
   * Advance simulation clock by elapsed execution time (e.g. attention or GEMM forward pass).
   * @param {number} elapsedMs
   */
  advance(elapsedMs) {
    for (const [key, req] of Array.from(this.inFlight.entries())) {
      req.estIoTimeMs -= elapsedMs;
      if (req.estIoTimeMs <= 0) {
        this.inFlight.delete(key);
        this.completed.add(key);
        this.stats.totalCompleted++;
        this.stats.overlapEvents++;
      }
    }
  }

  /**
   * Acquire expert: returns immediately if prefetch completed, or stalls for remainder.
   * @param {number} layerIdx
   * @param {number} expertIdx
   * @param {number} sizeBytes
   * @returns {{ hit: boolean, stallMs: number }}
   */
  acquire(layerIdx, expertIdx, sizeBytes = 8 * 1024 * 1024) {
    const key = `${layerIdx}:${expertIdx}`;

    if (this.completed.has(key)) {
      // 100% overlapped: zero stall
      return { hit: true, stallMs: 0.0 };
    }

    const inFlightReq = this.inFlight.get(key);
    if (inFlightReq) {
      // Partially overlapped: stall only for remaining I/O time
      const remainingStall = Math.max(0, inFlightReq.estIoTimeMs);
      this.inFlight.delete(key);
      this.completed.add(key);
      this.stats.totalCompleted++;
      this.stats.stallsMs += remainingStall;
      return { hit: false, stallMs: remainingStall };
    }

    // Completely un-prefetched: full synchronous NVMe read stall
    const fullStall = this.baseLatencyMs + (sizeBytes / (1024 * 1024)) / (this.bandwidthMBps / 1000.0);
    this.completed.add(key);
    this.stats.totalCompleted++;
    this.stats.stallsMs += fullStall;
    return { hit: false, stallMs: fullStall };
  }

  /**
   * Clear residency state for LRU eviction turn.
   */
  clear() {
    this.queue = [];
    this.inFlight.clear();
    this.completed.clear();
  }
}

/**
 * Simulate end-to-end MoE execution comparing synchronous vs asynchronous overlapped prefetching.
 *
 * @param {{
 *   layers?: number,
 *   expertsPerLayer?: number,
 *   topK?: number,
 *   expertSizeBytes?: number,
 *   attentionMs?: number,
 *   mlpComputeMs?: number,
 *   nvmeBandwidthMBps?: number,
 *   randomReadLatencyMs?: number,
 * }} options
 * @returns {{
 *   layers: number,
 *   expertsPerLayer: number,
 *   topK: number,
 *   syncTotalMs: number,
 *   overlappedTotalMs: number,
 *   latencyHiddenPct: number,
 *   effectiveSpeedup: number,
 *   syncToksPerSec: number,
 *   overlappedToksPerSec: number,
 *   perLayerStats: Array<{
 *     layer: number,
 *     topKExperts: number[],
 *     syncLayerMs: number,
 *     overlappedLayerMs: number,
 *     stallMs: number,
 *   }>,
 * }}
 */
export function simulateOverlappedExecution(options = {}) {
  const numLayers = options.layers || 16;
  const expertsPerLayer = options.expertsPerLayer || 8;
  const topK = options.topK || 2;
  const expertSize = options.expertSizeBytes || 8 * 1024 * 1024; // 8MB per expert slot
  const attnMs = options.attentionMs || 1.2; // Attention computation time per layer
  const mlpMs = options.mlpComputeMs || 2.4; // MLP forward computation time per layer
  const bandwidth = options.nvmeBandwidthMBps || 3400; // 3.4 GB/s NVMe read
  const randomReadLatency = options.randomReadLatencyMs || 0.08;

  const prefetchQueue = new PrefetchQueue({
    maxQueueDepth: 8,
    nvmeBandwidthMBps: bandwidth,
    randomReadLatencyMs: randomReadLatency,
  });

  // Single expert I/O transfer time
  const singleExpertIoMs = randomReadLatency + (expertSize / (1024 * 1024)) / (bandwidth / 1000.0);

  let syncTotalMs = 0.0;
  let overlappedTotalMs = 0.0;
  let totalHiddenIoMs = 0.0;
  let totalRawIoMs = 0.0;

  const perLayerStats = [];

  // Deterministic seed for routing prediction
  let rng = 0x4f8b1a3d;
  function pseudoRand(mod) {
    rng ^= (rng << 13) >>> 0;
    rng ^= (rng >>> 17) >>> 0;
    rng ^= (rng << 5) >>> 0;
    return (rng >>> 0) % mod;
  }

  // Pre-generate top-K experts for all layers
  const layerPredictions = [];
  for (let l = 0; l < numLayers; l++) {
    const selected = new Set();
    while (selected.size < topK) {
      selected.add(pseudoRand(expertsPerLayer));
    }
    layerPredictions.push(Array.from(selected));
  }

  // Warm initial prefetch for Layer 0
  for (const exp of layerPredictions[0]) {
    prefetchQueue.enqueue(0, exp, expertSize);
  }

  for (let l = 0; l < numLayers; l++) {
    const currentTopK = layerPredictions[l];

    // 1. Synchronous Baseline Execution:
    // Attention + Full Page-in I/O for all Top-K + MLP Compute
    const syncIoMs = currentTopK.length * singleExpertIoMs;
    const syncLayerMs = attnMs + syncIoMs + mlpMs;
    syncTotalMs += syncLayerMs;
    totalRawIoMs += syncIoMs;

    // 2. Overlapped Execution:
    // Step A: Attention forward pass executes on host processor
    // Step B: Speculatively dispatch prefetch for Layer L+1 during attention
    if (l + 1 < numLayers) {
      for (const nextExp of layerPredictions[l + 1]) {
        prefetchQueue.enqueue(l + 1, nextExp, expertSize);
      }
    }

    // Time elapsed during attention allows queued I/O to progress
    prefetchQueue.advance(attnMs);

    // Step C: Acquire current layer's experts (measure any residual stall)
    let layerStallMs = 0.0;
    for (const exp of currentTopK) {
      const { stallMs } = prefetchQueue.acquire(l, exp, expertSize);
      layerStallMs += stallMs;
    }

    // Step D: MLP forward pass executes (also overlaps with in-flight next-layer reads)
    prefetchQueue.advance(mlpMs);

    const overlappedLayerMs = attnMs + layerStallMs + mlpMs;
    overlappedTotalMs += overlappedLayerMs;

    const hiddenInLayer = Math.max(0.0, syncIoMs - layerStallMs);
    totalHiddenIoMs += hiddenInLayer;

    perLayerStats.push({
      layer: l,
      topKExperts: currentTopK,
      syncLayerMs: Number(syncLayerMs.toFixed(3)),
      overlappedLayerMs: Number(overlappedLayerMs.toFixed(3)),
      stallMs: Number(layerStallMs.toFixed(3)),
    });
  }

  const latencyHiddenPct = totalRawIoMs > 0 ? Number(((totalHiddenIoMs / totalRawIoMs) * 100).toFixed(1)) : 0.0;
  const effectiveSpeedup = Number((syncTotalMs / Math.max(0.1, overlappedTotalMs)).toFixed(2));
  const syncToksPerSec = Number((1000.0 / Math.max(0.1, syncTotalMs)).toFixed(1));
  const overlappedToksPerSec = Number((1000.0 / Math.max(0.1, overlappedTotalMs)).toFixed(1));

  return {
    layers: numLayers,
    expertsPerLayer,
    topK,
    syncTotalMs: Number(syncTotalMs.toFixed(2)),
    overlappedTotalMs: Number(overlappedTotalMs.toFixed(2)),
    latencyHiddenPct,
    effectiveSpeedup,
    syncToksPerSec,
    overlappedToksPerSec,
    perLayerStats,
  };
}

/**
 * Benchmark host NVMe prefetch efficiency and print summary metrics.
 * @param {object} options
 * @returns {object}
 */
export function benchmarkPrefetchEfficiency(options = {}) {
  return simulateOverlappedExecution(options);
}
