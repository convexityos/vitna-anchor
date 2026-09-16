import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  sliceStripedDmaSlabs,
  benchmarkStripedReadThroughput,
  formatStripeSummary,
} from "../runtime/stripe.mjs";

test("benchmarkStripedReadThroughput scales bandwidth linearly across drive counts", () => {
  const single = benchmarkStripedReadThroughput(1, 7.45);
  assert.equal(single.driveCount, 1);
  assert.equal(single.aggregateGBps, 7.45);
  assert.equal(single.scalingEfficiencyPct, 100);

  const dual = benchmarkStripedReadThroughput(2, 7.45);
  assert.equal(dual.driveCount, 2);
  assert.equal(dual.aggregateGBps, 14.3);
  assert.equal(dual.scalingEfficiencyPct, 96);
  assert.ok(dual.projectedToksSec.deepseek671b > single.projectedToksSec.deepseek671b);

  const quad = benchmarkStripedReadThroughput(4, 7.45);
  assert.equal(quad.driveCount, 4);
  assert.equal(quad.aggregateGBps, 27.42);
  assert.equal(quad.scalingEfficiencyPct, 92);
  assert.ok(quad.projectedToksSec.deepseek671b > dual.projectedToksSec.deepseek671b);
});

test("sliceStripedDmaSlabs round-robins chunks across drive partitions", () => {
  const testRoot = join(tmpdir(), "vitna-test-stripe-" + Date.now());
  const drive0 = join(testRoot, "nvme0");
  const drive1 = join(testRoot, "nvme1");

  try {
    const manifest = sliceStripedDmaSlabs("test-moe.dma.anchor", [drive0, drive1], {
      chunkSizeBytes: 4096,
      syntheticSizeBytes: 32768, // 8 x 4KB sectors total -> 4 sectors per drive
    });

    assert.equal(manifest.driveCount, 2);
    assert.equal(manifest.chunkSizeBytes, 4096);
    assert.equal(manifest.totalSectorPaddedBytes, 32768);
    assert.equal(manifest.partitions.length, 2);

    assert.equal(manifest.partitions[0].sectorCount, 4);
    assert.equal(manifest.partitions[1].sectorCount, 4);
    assert.ok(existsSync(manifest.partitions[0].path));
    assert.ok(existsSync(manifest.partitions[1].path));
    assert.match(manifest.airgapSha256, /^[0-9a-f]{64}$/);

    const summary = formatStripeSummary(manifest);
    assert.ok(summary.includes("MULTI-DRIVE NVME STRIPE COMPLETE"));
    assert.ok(summary.includes("2 Drives Striped"));
  } finally {
    try {
      rmSync(testRoot, { recursive: true, force: true });
    } catch {}
  }
});
