import assert from "node:assert/strict";
import test from "node:test";
import { writeFileSync, unlinkSync, rmdirSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  alignUp,
  parseSafeTensorsHeader,
  parseMoEExpertRole,
  sliceDmaAlignedSlabs,
  verifyDmaAlignment,
  runModelPull,
} from "../runtime/ingest.mjs";

function createSyntheticSafeTensorsFile(targetPath) {
  // Tensor 1: Dense embedding (100 floats = 400 bytes)
  const data1 = Buffer.alloc(400, 0x11);
  // Tensor 2: MoE Layer 0 Expert 0 (256 floats = 1024 bytes)
  const data2 = Buffer.alloc(1024, 0x22);
  // Tensor 3: MoE Layer 0 Expert 1 (512 floats = 2048 bytes)
  const data3 = Buffer.alloc(2048, 0x33);

  const headerObj = {
    "__metadata__": { "format": "pt" },
    "model.embed_tokens.weight": {
      "dtype": "F32",
      "shape": [100],
      "data_offsets": [0, 400],
    },
    "model.layers.0.mlp.experts.0.up_proj.weight": {
      "dtype": "F32",
      "shape": [256],
      "data_offsets": [400, 1424],
    },
    "model.layers.0.mlp.experts.1.up_proj.weight": {
      "dtype": "F32",
      "shape": [512],
      "data_offsets": [1424, 3472],
    },
  };

  const headerJson = Buffer.from(JSON.stringify(headerObj), "utf8");
  const headerLenBuf = Buffer.alloc(8);
  headerLenBuf.writeBigUInt64LE(BigInt(headerJson.length), 0);

  // Contiguous file: [8 bytes len] [headerJson] [data1] [data2] [data3]
  const fullFile = Buffer.concat([headerLenBuf, headerJson, data1, data2, data3]);
  writeFileSync(targetPath, fullFile);

  return {
    headerLen: headerJson.length,
    data1,
    data2,
    data3,
    totalBytes: fullFile.length,
  };
}

test("alignUp rounds offsets up to sector multiples", () => {
  assert.equal(alignUp(0, 4096), 0);
  assert.equal(alignUp(1, 4096), 4096);
  assert.equal(alignUp(4095, 4096), 4096);
  assert.equal(alignUp(4096, 4096), 4096);
  assert.equal(alignUp(4097, 4096), 8192);
});

test("parseMoEExpertRole identifies routing patterns", () => {
  const t1 = parseMoEExpertRole("model.layers.3.mlp.experts.7.up_proj.weight");
  assert.equal(t1.isExpert, true);
  assert.equal(t1.layerIdx, 3);
  assert.equal(t1.expertIdx, 7);

  const t2 = parseMoEExpertRole("layers.15.block_sparse_moe.experts.2.w1.weight");
  assert.equal(t2.isExpert, true);
  assert.equal(t2.layerIdx, 15);
  assert.equal(t2.expertIdx, 2);

  const t3 = parseMoEExpertRole("model.embed_tokens.weight");
  assert.equal(t3.isExpert, false);
  assert.equal(t3.layerIdx, -1);
});

test("parseSafeTensorsHeader parses binary headers and validates integrity", () => {
  const headerObj = {
    "__metadata__": { "author": "vitna" },
    "weight_a": { "dtype": "F32", "shape": [10], "data_offsets": [0, 40] },
  };
  const jsonBuf = Buffer.from(JSON.stringify(headerObj), "utf8");
  const lenBuf = Buffer.alloc(8);
  lenBuf.writeBigUInt64LE(BigInt(jsonBuf.length), 0);
  const fullBuf = Buffer.concat([lenBuf, jsonBuf]);

  const parsed = parseSafeTensorsHeader(fullBuf);
  assert.equal(parsed.headerLen, jsonBuf.length);
  assert.equal(parsed.metadata.author, "vitna");
  assert.deepEqual(parsed.tensors.weight_a.shape, [10]);
  assert.equal(parsed.rawDataBase, 8 + jsonBuf.length);

  assert.throws(() => parseSafeTensorsHeader(Buffer.alloc(4)), /Buffer too short/);
});

test("sliceDmaAlignedSlabs repacks safetensors with 100% 4KB aligned offsets", () => {
  const testDir = join(tmpdir(), `vitna_test_ingest_${Date.now()}`);
  mkdirSync(testDir, { recursive: true });

  const srcFile = join(testDir, "synth_model.safetensors");
  const outDir = join(testDir, "dma_out");
  const synth = createSyntheticSafeTensorsFile(srcFile);

  try {
    // 1. Dry run test
    const dryResult = sliceDmaAlignedSlabs(srcFile, outDir, { dryRun: true });
    assert.equal(dryResult.tensorCount, 3);
    assert.equal(dryResult.records.length, 3);
    for (const rec of dryResult.records) {
      assert.equal(rec.targetOffset % 4096, 0, `Record ${rec.name} must be 4KB aligned`);
    }

    // 2. Physical slice test
    const realResult = sliceDmaAlignedSlabs(srcFile, outDir, { dryRun: false });
    assert.equal(realResult.tensorCount, 3);
    assert.ok(existsSync(realResult.alignedFilePath), "DMA aligned file must exist");
    assert.ok(existsSync(realResult.manifestPath), "Index manifest must exist");
    assert.ok(realResult.airgapHash.length === 64, "SHA-256 hash must be 64 hex chars");

    // 3. Verify alignment through validator
    const check = verifyDmaAlignment(realResult.manifestPath);
    assert.equal(check.valid, true);
    assert.equal(check.errors.length, 0);
    assert.equal(check.tensorCount, 3);

    // 4. Verify data payload bit-for-bit
    const alignedBytes = readFileSync(realResult.alignedFilePath);
    for (const rec of realResult.records) {
      assert.equal(rec.targetOffset % 4096, 0);
      const extracted = alignedBytes.subarray(rec.targetOffset, rec.targetOffset + rec.rawSize);
      if (rec.name === "model.embed_tokens.weight") {
        assert.deepEqual(extracted, synth.data1);
      } else if (rec.name === "model.layers.0.mlp.experts.0.up_proj.weight") {
        assert.deepEqual(extracted, synth.data2);
      } else if (rec.name === "model.layers.0.mlp.experts.1.up_proj.weight") {
        assert.deepEqual(extracted, synth.data3);
      }
    }
  } finally {
    try {
      if (existsSync(srcFile)) unlinkSync(srcFile);
      if (existsSync(join(outDir, "synth_model.dma.anchor"))) unlinkSync(join(outDir, "synth_model.dma.anchor"));
      if (existsSync(join(outDir, "synth_model.anchor.index.json"))) unlinkSync(join(outDir, "synth_model.anchor.index.json"));
      if (existsSync(outDir)) rmdirSync(outDir);
      if (existsSync(testDir)) rmdirSync(testDir);
    } catch {}
  }
});

test("runModelPull executes local checkpoint ingestion and remote dry-run", async () => {
  const testDir = join(tmpdir(), `vitna_test_pull_${Date.now()}`);
  mkdirSync(testDir, { recursive: true });
  const srcFile = join(testDir, "local_test.safetensors");
  createSyntheticSafeTensorsFile(srcFile);

  try {
    // Local pull test
    const localResult = await runModelPull(srcFile, { outDir: join(testDir, "out") });
    assert.equal(localResult.tensorCount, 3);
    assert.ok(localResult.airgapHash);

    // Remote dry-run test
    const remoteResult = await runModelPull("Qwen/Qwen2.5-Coder-7B-Instruct", { dryRun: true });
    assert.equal(remoteResult.dryRun, true);
    assert.equal(remoteResult.sectorSize, 4096);
  } finally {
    try {
      if (existsSync(srcFile)) unlinkSync(srcFile);
      if (existsSync(join(testDir, "out", "local_test.dma.anchor"))) unlinkSync(join(testDir, "out", "local_test.dma.anchor"));
      if (existsSync(join(testDir, "out", "local_test.anchor.index.json"))) unlinkSync(join(testDir, "out", "local_test.anchor.index.json"));
      if (existsSync(join(testDir, "out"))) rmdirSync(join(testDir, "out"));
      if (existsSync(testDir)) rmdirSync(testDir);
    } catch {}
  }
});
