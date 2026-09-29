import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { sliceStripedDmaSlabs, formatStripeSummary } from "../runtime/stripe.mjs";

test("sliceStripedDmaSlabs round-robins chunks across directories", () => {
  const testRoot = join(tmpdir(), "vitna-test-stripe-" + Date.now());
  const source = join(testRoot, "test-moe.dma.anchor");
  const drive0 = join(testRoot, "nvme0");
  const drive1 = join(testRoot, "nvme1");

  try {
    mkdirSync(testRoot, { recursive: true });
    // 8 chunks of 4096 bytes, each filled with its own index
    const payload = Buffer.alloc(8 * 4096);
    for (let c = 0; c < 8; c++) payload.fill(c + 1, c * 4096, (c + 1) * 4096);
    writeFileSync(source, payload);

    const manifest = sliceStripedDmaSlabs(source, [drive0, drive1], { chunkSizeBytes: 4096 });

    assert.equal(manifest.driveCount, 2);
    assert.equal(manifest.chunkSizeBytes, 4096);
    assert.equal(manifest.totalSectorPaddedBytes, 32768);
    assert.equal(manifest.partitions.length, 2);
    assert.equal("bandwidthScalingFactor" in manifest, false, "no throughput figure is invented");

    assert.equal(manifest.partitions[0].sectorCount, 4);
    assert.equal(manifest.partitions[1].sectorCount, 4);
    assert.ok(existsSync(manifest.partitions[0].path));
    assert.ok(existsSync(manifest.partitions[1].path));
    // The source is already a whole number of chunks, so no padding was added
    // and the striped digest is the source file's own digest.
    assert.equal(manifest.stripedDataSha256, createHash("sha256").update(payload).digest("hex"));

    // Even chunks land in the first directory and odd chunks in the second
    const first = readFileSync(manifest.partitions[0].path);
    const second = readFileSync(manifest.partitions[1].path);
    assert.deepEqual([...first.subarray(0, 4096 * 4)].filter((_, i) => i % 4096 === 0), [1, 3, 5, 7]);
    assert.deepEqual([...second.subarray(0, 4096 * 4)].filter((_, i) => i % 4096 === 0), [2, 4, 6, 8]);

    const summary = formatStripeSummary(manifest);
    assert.ok(summary.includes("STRIPE WRITTEN"));
    assert.ok(summary.includes("chunks written round-robin"));
  } finally {
    rmSync(testRoot, { recursive: true, force: true });
  }
});

test("sliceStripedDmaSlabs refuses a source file that does not exist", () => {
  const testRoot = join(tmpdir(), "vitna-test-stripe-missing-" + Date.now());
  try {
    assert.throws(
      () => sliceStripedDmaSlabs(join(testRoot, "missing.gguf"), [join(testRoot, "a"), join(testRoot, "b")]),
      /Source file not found/
    );
  } finally {
    rmSync(testRoot, { recursive: true, force: true });
  }
});
