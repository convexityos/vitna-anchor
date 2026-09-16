import assert from "node:assert/strict";
import test from "node:test";

import {
  PrefetchQueue,
  simulateOverlappedExecution,
  benchmarkPrefetchEfficiency,
} from "../runtime/prefetch.mjs";

test("PrefetchQueue enqueues requests, advances clock, and overlaps I/O", () => {
  const queue = new PrefetchQueue({
    maxQueueDepth: 4,
    nvmeBandwidthMBps: 3200,
    randomReadLatencyMs: 0.1,
  });

  // Enqueue Layer 0 Expert 2
  const ok1 = queue.enqueue(0, 2, 4 * 1024 * 1024);
  assert.equal(ok1, true);

  // Redundant enqueue returns false
  const okDup = queue.enqueue(0, 2, 4 * 1024 * 1024);
  assert.equal(okDup, false);

  // Immediate acquire before advance incurs stall
  const acqEarly = queue.acquire(0, 2, 4 * 1024 * 1024);
  assert.equal(acqEarly.hit, false);
  assert.ok(acqEarly.stallMs > 0);

  // After completion, acquire is an immediate hit (0ms stall)
  const acqDone = queue.acquire(0, 2, 4 * 1024 * 1024);
  assert.equal(acqDone.hit, true);
  assert.equal(acqDone.stallMs, 0.0);
});

test("simulateOverlappedExecution achieves positive latency hiding and speedup", () => {
  const result = simulateOverlappedExecution({
    layers: 8,
    expertsPerLayer: 8,
    topK: 2,
    attentionMs: 2.0,
    mlpComputeMs: 3.0,
    nvmeBandwidthMBps: 3500,
  });

  assert.equal(result.layers, 8);
  assert.equal(result.expertsPerLayer, 8);
  assert.equal(result.topK, 2);
  assert.ok(result.syncTotalMs > result.overlappedTotalMs);
  assert.ok(result.latencyHiddenPct > 0);
  assert.ok(result.effectiveSpeedup >= 1.0);
  assert.ok(result.overlappedToksPerSec >= result.syncToksPerSec);
  assert.equal(result.perLayerStats.length, 8);
});

test("benchmarkPrefetchEfficiency returns full telemetry profile", () => {
  const bench = benchmarkPrefetchEfficiency({ layers: 4 });
  assert.equal(bench.layers, 4);
  assert.ok(bench.syncToksPerSec > 0);
  assert.ok(bench.overlappedToksPerSec > 0);
});
