// tests/gguf-ingestion.test.mjs - Unit tests for GGUF binary header parser and 4KB DMA slab slicer.
// Zero external dependencies. Strictly zero em-dashes.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  parseGgufHeader,
  inspectGgufFile,
  sliceGgufDmaSlabs,
  calculateGgufTensorSize,
  GGML_TYPE_META,
  verifyDmaAlignment,
} from "../runtime/ingest.mjs";

/**
 * Helper to build a valid synthetic GGUF binary buffer.
 */
function buildSyntheticGguf({ metadata = {}, tensors = [] }) {
  const buffers = [];

  // 1. Header: Magic (4B), Version (4B), TensorCount (8B), MetadataCount (8B)
  const headerBuf = Buffer.alloc(24);
  headerBuf.write("GGUF", 0, 4, "utf8");
  headerBuf.writeUInt32LE(3, 4); // version 3
  headerBuf.writeBigUInt64LE(BigInt(tensors.length), 8);
  headerBuf.writeBigUInt64LE(BigInt(Object.keys(metadata).length), 16);
  buffers.push(headerBuf);

  // 2. Metadata Key-Value pairs
  for (const [key, val] of Object.entries(metadata)) {
    const keyBytes = Buffer.from(key, "utf8");
    const keyLenBuf = Buffer.alloc(8);
    keyLenBuf.writeBigUInt64LE(BigInt(keyBytes.length), 0);
    buffers.push(keyLenBuf, keyBytes);

    if (typeof val === "string") {
      const typeBuf = Buffer.alloc(4);
      typeBuf.writeUInt32LE(8, 0); // STRING = 8
      const valBytes = Buffer.from(val, "utf8");
      const valLenBuf = Buffer.alloc(8);
      valLenBuf.writeBigUInt64LE(BigInt(valBytes.length), 0);
      buffers.push(typeBuf, valLenBuf, valBytes);
    } else if (typeof val === "number") {
      const typeBuf = Buffer.alloc(4);
      typeBuf.writeUInt32LE(4, 0); // UINT32 = 4
      const valBuf = Buffer.alloc(4);
      valBuf.writeUInt32LE(val, 0);
      buffers.push(typeBuf, valBuf);
    } else if (typeof val === "boolean") {
      const typeBuf = Buffer.alloc(4);
      typeBuf.writeUInt32LE(7, 0); // BOOL = 7
      const valBuf = Buffer.alloc(1);
      valBuf.writeUInt8(val ? 1 : 0, 0);
      buffers.push(typeBuf, valBuf);
    }
  }

  // 3. Tensor Info descriptors
  let runningDataOffset = 0;
  const tensorDataBuffers = [];

  for (const t of tensors) {
    const nameBytes = Buffer.from(t.name, "utf8");
    const nameLenBuf = Buffer.alloc(8);
    nameLenBuf.writeBigUInt64LE(BigInt(nameBytes.length), 0);
    buffers.push(nameLenBuf, nameBytes);

    const nDimsBuf = Buffer.alloc(4);
    nDimsBuf.writeUInt32LE(t.shape.length, 0);
    buffers.push(nDimsBuf);

    for (const dim of t.shape) {
      const dimBuf = Buffer.alloc(8);
      dimBuf.writeBigUInt64LE(BigInt(dim), 0);
      buffers.push(dimBuf);
    }

    const typeBuf = Buffer.alloc(4);
    typeBuf.writeUInt32LE(t.type, 0);
    buffers.push(typeBuf);

    const offsetBuf = Buffer.alloc(8);
    offsetBuf.writeBigUInt64LE(BigInt(runningDataOffset), 0);
    buffers.push(offsetBuf);

    const tensorBytes = t.data || Buffer.alloc(calculateGgufTensorSize(t.type, t.shape), 0xaa);
    tensorDataBuffers.push(tensorBytes);
    runningDataOffset += tensorBytes.length;
  }

  // Concatenate header + metadata + tensor info
  const metaCombined = Buffer.concat(buffers);

  // Alignment padding to 32 bytes (default GGUF alignment)
  const padLen = (32 - (metaCombined.length % 32)) % 32;
  const padBuf = Buffer.alloc(padLen, 0);

  // Full file = meta + pad + raw data
  return Buffer.concat([metaCombined, padBuf, ...tensorDataBuffers]);
}

test("calculateGgufTensorSize accurately calculates byte lengths across GGML types", () => {
  // F32: 1024 elements * 4 bytes = 4096 bytes
  assert.equal(calculateGgufTensorSize(0, [32, 32]), 4096);

  // F16: 1024 elements * 2 bytes = 2048 bytes
  assert.equal(calculateGgufTensorSize(1, [32, 32]), 2048);

  // Q4_0: 1024 elements / 32 blocks = 32 blocks * 18 bytes = 576 bytes
  assert.equal(calculateGgufTensorSize(2, [32, 32]), 576);

  // Q8_0: 1024 elements / 32 blocks = 32 blocks * 34 bytes = 1088 bytes
  assert.equal(calculateGgufTensorSize(7, [32, 32]), 1088);

  // Q4_K: 1024 elements / 256 blocks = 4 blocks * 144 bytes = 576 bytes
  assert.equal(calculateGgufTensorSize(12, [32, 32]), 576);
});

test("parseGgufHeader validates magic, version, and extracts metadata", () => {
  const badMagic = Buffer.alloc(32);
  badMagic.write("FAIL", 0, 4, "utf8");
  assert.throws(() => parseGgufHeader(badMagic), /Invalid GGUF magic header/);

  const validBuf = buildSyntheticGguf({
    metadata: {
      "general.architecture": "llama",
      "general.alignment": 32,
      "general.file_type": 2,
    },
    tensors: [
      { name: "token_embd.weight", shape: [64, 128], type: 0 },
      { name: "blk.0.ffn_gate_exps.0.weight", shape: [64, 64], type: 2 },
    ],
  });

  const parsed = parseGgufHeader(validBuf);
  assert.equal(parsed.version, 3);
  assert.equal(parsed.tensorCount, 2);
  assert.equal(parsed.metadata["general.architecture"], "llama");
  assert.equal(parsed.metadata["general.alignment"], 32);
  assert.ok(parsed.tensors["token_embd.weight"]);
  assert.deepEqual(parsed.tensors["token_embd.weight"].shape, [64, 128]);
  assert.equal(parsed.tensors["token_embd.weight"].dtype, "F32");
  assert.ok(parsed.tensors["blk.0.ffn_gate_exps.0.weight"]);
  assert.equal(parsed.tensors["blk.0.ffn_gate_exps.0.weight"].dtype, "Q4_0");
});

test("sliceGgufDmaSlabs slices GGUF into 100% 4KB DMA aligned slabs with index manifest", () => {
  const tmpDir = mkdtempSync(join(tmpdir(), "vitna_test_gguf_"));
  const ggufPath = join(tmpDir, "model.gguf");
  const outDir = join(tmpDir, "out_dma");

  try {
    const rawTensor1 = Buffer.alloc(500, 0x11);
    const rawTensor2 = Buffer.alloc(1200, 0x22);

    const ggufBuf = buildSyntheticGguf({
      metadata: {
        "general.architecture": "qwen2",
        "general.name": "test-qwen",
      },
      tensors: [
        { name: "model.embed_tokens.weight", shape: [10, 50], type: 0, data: rawTensor1 },
        { name: "model.layers.0.mlp.experts.1.up_proj.weight", shape: [20, 60], type: 1, data: rawTensor2 },
      ],
    });

    writeFileSync(ggufPath, ggufBuf);

    // 1. Dry run verification
    const dryResult = sliceGgufDmaSlabs(ggufPath, outDir, { dryRun: true });
    assert.equal(dryResult.tensorCount, 2);
    assert.equal(dryResult.airgapHash, "DRY_RUN_ATTESTATION_PENDING");

    // 2. Real slice execution
    const result = sliceGgufDmaSlabs(ggufPath, outDir);
    assert.equal(result.tensorCount, 2);
    assert.ok(existsSync(result.alignedFilePath));
    assert.ok(existsSync(result.manifestPath));
    assert.equal(typeof result.airgapHash, "string");
    assert.equal(result.airgapHash.length, 64);

    // 3. Verify manifest contents and 4KB alignment invariant
    const manifest = JSON.parse(readFileSync(result.manifestPath, "utf8"));
    assert.equal(manifest.format, "vitna-anchor-dma-v1");
    assert.equal(manifest.sourceFormat, "gguf");
    assert.equal(manifest.sectorSize, 4096);
    assert.equal(manifest.tensors.length, 2);

    for (const t of manifest.tensors) {
      assert.equal(t.offset % 4096, 0, `Tensor ${t.name} offset ${t.offset} must be multiple of 4096`);
      assert.equal(t.aligned_4k, true);
    }

    // Expert role detection
    const expertTensor = manifest.tensors.find((t) => t.name.includes("experts"));
    assert.ok(expertTensor);
    assert.equal(expertTensor.isExpert, true);
    assert.equal(expertTensor.layerIdx, 0);
    assert.equal(expertTensor.expertIdx, 1);

    const alignCheck = verifyDmaAlignment(result.manifestPath);
    assert.equal(alignCheck.valid, true);
    assert.equal(alignCheck.errors.length, 0);
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
});
