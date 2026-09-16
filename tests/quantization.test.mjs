import assert from "node:assert/strict";
import test from "node:test";
import { writeFileSync, unlinkSync, rmSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  alignUp,
  quantizeTensorBuffer,
  quantizeSlabFile,
} from "../runtime/quantize.mjs";

function createSyntheticSafeTensorsFile(targetPath) {
  // Tensor 1: 128 floats = 512 bytes
  const data1 = new Float32Array(128);
  for (let i = 0; i < 128; i++) data1[i] = Math.sin(i * 0.1);

  // Tensor 2: 256 floats = 1024 bytes
  const data2 = new Float32Array(256);
  for (let i = 0; i < 256; i++) data2[i] = Math.cos(i * 0.05);

  const b1 = Buffer.from(data1.buffer, data1.byteOffset, data1.byteLength);
  const b2 = Buffer.from(data2.buffer, data2.byteOffset, data2.byteLength);

  const headerObj = {
    "__metadata__": { "format": "pt" },
    "weight_1": { "dtype": "F32", "shape": [128], "data_offsets": [0, b1.length] },
    "weight_2": { "dtype": "F32", "shape": [256], "data_offsets": [b1.length, b1.length + b2.length] },
  };

  const headerJson = Buffer.from(JSON.stringify(headerObj), "utf8");
  const headerLenBuf = Buffer.alloc(8);
  headerLenBuf.writeBigUInt64LE(BigInt(headerJson.length), 0);

  const fullFile = Buffer.concat([headerLenBuf, headerJson, b1, b2]);
  writeFileSync(targetPath, fullFile);

  return { targetPath, data1, data2 };
}

test("quantizeTensorBuffer converts Float32 to INT8 with accurate scale and SNR", () => {
  const f32 = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    f32[i] = (i - 128) / 128.0;
  }

  const res = quantizeTensorBuffer(f32, 8, 64);
  assert.equal(res.bits, 8);
  assert.equal(res.groupSize, 64);
  assert.equal(res.originalBytes, 1024);
  assert.equal(res.quantizedBuffer.length, 256);
  assert.equal(res.scales.length, 4);
  assert.ok(res.compressionRatio > 3.0);
  assert.ok(res.snrDb > 35.0, `SNR should be > 35 dB for INT8, got ${res.snrDb}`);
});

test("quantizeTensorBuffer converts Float32 to packed INT4 with 2 nibbles per byte", () => {
  const f32 = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    f32[i] = (i - 128) / 128.0;
  }

  const res = quantizeTensorBuffer(f32, 4, 64);
  assert.equal(res.bits, 4);
  assert.equal(res.groupSize, 64);
  assert.equal(res.originalBytes, 1024);
  assert.equal(res.quantizedBuffer.length, 128); // 256 nibbles = 128 bytes
  assert.equal(res.scales.length, 4);
  assert.ok(res.compressionRatio > 6.0);
  assert.ok(res.snrDb > 15.0, `SNR should be > 15 dB for INT4, got ${res.snrDb}`);
});

test("quantizeSlabFile packs tensors into 4KB sector aligned quantized slabs", () => {
  const testDir = join(tmpdir(), `vitna_test_quant_${Date.now()}`);
  mkdirSync(testDir, { recursive: true });
  const srcFile = join(testDir, "test_model.safetensors");
  const outDir = join(testDir, "quant_out");

  createSyntheticSafeTensorsFile(srcFile);

  try {
    // 1. Dry run
    const dryRes = quantizeSlabFile(srcFile, outDir, { bits: 4, dryRun: true });
    assert.equal(dryRes.totalTensors, 2);
    assert.equal(dryRes.bits, 4);
    assert.ok(dryRes.netCompressionRatio > 0);

    // 2. Physical write
    const realRes = quantizeSlabFile(srcFile, outDir, { bits: 4, dryRun: false });
    assert.equal(realRes.totalTensors, 2);
    assert.ok(existsSync(realRes.quantizedSlabPath));
    assert.ok(existsSync(realRes.manifestPath));

    const manifest = JSON.parse(readFileSync(realRes.manifestPath, "utf8"));
    assert.equal(manifest.sectorSize, 4096);
    assert.equal(manifest.bits, 4);
    assert.equal(manifest.quantType, "VITNA_QUANT_INT4");

    for (const t of manifest.tensors) {
      assert.equal(t.slabOffset % 4096, 0, `Slab offset must be 4KB aligned: ${t.slabOffset}`);
      assert.equal(t.slabSize % 4096, 0, `Slab size must be 4KB aligned: ${t.slabSize}`);
      assert.equal(t.aligned_4k, true);
    }
  } finally {
    try {
      if (existsSync(srcFile)) unlinkSync(srcFile);
      if (existsSync(outDir)) rmSync(outDir, { recursive: true, force: true });
      if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    } catch {}
  }
});
