import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { runSiliconTune, generateHardwareProfile } from "../runtime/tune.mjs";

test("runSiliconTune discovers hardware topology and storage sweep", () => {
  const profile = runSiliconTune({ quick: true });

  assert.ok(profile.system.cpuModel);
  assert.ok(profile.system.logicalCores > 0);
  assert.ok(profile.system.totalRamGb > 0);
  assert.ok(profile.memoryBandwidthGBps > 0);
  assert.ok(profile.random4kIops > 0);

  assert.equal(profile.storageSweep.length, 5);
  for (const sw of profile.storageSweep) {
    assert.ok(sw.blockSizeBytes > 0);
    assert.ok(sw.throughputMBps > 0);
  }

  const rec = profile.tuningRecommendations;
  assert.equal(rec.optimalSectorAlignment, 4096);
  assert.ok(rec.optimalDmaChunkSize >= 4096);
  assert.ok(rec.recommendedPrefetchDepth >= 8);
  assert.ok(rec.recommendedRamBudgetMb >= 2048);
  assert.ok(["VITNA_QUANT_INT4", "VITNA_QUANT_INT8", "VITNA_QUANT_FP16"].includes(rec.recommendedQuantTier));
});

test("runSiliconTune writes tuning profile to disk when outFile is specified", () => {
  const tempFile = join(tmpdir(), `vitna_tune_test_${Date.now()}.json`);

  try {
    const profile = runSiliconTune({ quick: true, outFile: tempFile });
    assert.ok(existsSync(tempFile));
    assert.equal(profile.profilePath, tempFile);
  } finally {
    if (existsSync(tempFile)) {
      try { unlinkSync(tempFile); } catch {}
    }
  }
});
