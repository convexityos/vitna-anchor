import assert from "node:assert/strict";
import test from "node:test";
import { writeFileSync, unlinkSync, rmSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  alignUp,
  parseSafeTensorsHeader,
  parseSafeTensorsIndex,
  parseMoEExpertRole,
  sliceDmaAlignedSlabs,
  sliceShardedDmaSlabs,
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

function createSyntheticShardedSafeTensors(dir) {
  // Shard 1: embed_tokens + layer 0 expert 0
  const s1_data1 = Buffer.alloc(300, 0xaa);
  const s1_data2 = Buffer.alloc(600, 0xbb);
  const s1_header = {
    "model.embed_tokens.weight": { "dtype": "F32", "shape": [75], "data_offsets": [0, 300] },
    "model.layers.0.mlp.experts.0.up_proj.weight": { "dtype": "F32", "shape": [150], "data_offsets": [300, 900] },
  };
  const s1_json = Buffer.from(JSON.stringify(s1_header), "utf8");
  const s1_len = Buffer.alloc(8);
  s1_len.writeBigUInt64LE(BigInt(s1_json.length), 0);
  const s1_path = join(dir, "model-00001-of-00002.safetensors");
  writeFileSync(s1_path, Buffer.concat([s1_len, s1_json, s1_data1, s1_data2]));

  // Shard 2: layer 0 expert 1 + lm_head
  const s2_data1 = Buffer.alloc(600, 0xcc);
  const s2_data2 = Buffer.alloc(400, 0xdd);
  const s2_header = {
    "model.layers.0.mlp.experts.1.up_proj.weight": { "dtype": "F32", "shape": [150], "data_offsets": [0, 600] },
    "lm_head.weight": { "dtype": "F32", "shape": [100], "data_offsets": [600, 1000] },
  };
  const s2_json = Buffer.from(JSON.stringify(s2_header), "utf8");
  const s2_len = Buffer.alloc(8);
  s2_len.writeBigUInt64LE(BigInt(s2_json.length), 0);
  const s2_path = join(dir, "model-00002-of-00002.safetensors");
  writeFileSync(s2_path, Buffer.concat([s2_len, s2_json, s2_data1, s2_data2]));

  // Index JSON
  const indexObj = {
    "metadata": { "total_size": 1900 },
    "weight_map": {
      "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
      "model.layers.0.mlp.experts.0.up_proj.weight": "model-00001-of-00002.safetensors",
      "model.layers.0.mlp.experts.1.up_proj.weight": "model-00002-of-00002.safetensors",
      "lm_head.weight": "model-00002-of-00002.safetensors",
    },
  };
  const indexPath = join(dir, "model.safetensors.index.json");
  writeFileSync(indexPath, JSON.stringify(indexObj, null, 2));

  return { s1_path, s2_path, indexPath, s1_data1, s1_data2, s2_data1, s2_data2 };
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

test("parseSafeTensorsIndex extracts shards and weight mappings", () => {
  const indexObj = {
    "metadata": { "total_size": "5000000000" },
    "weight_map": {
      "layer.0.weight": "model-00001-of-00002.safetensors",
      "layer.1.weight": "model-00002-of-00002.safetensors",
    },
  };
  const parsed = parseSafeTensorsIndex(JSON.stringify(indexObj));
  assert.equal(parsed.totalSize, 5000000000);
  assert.equal(parsed.shards.length, 2);
  assert.deepEqual(parsed.shards, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]);
});

test("sliceDmaAlignedSlabs repacks safetensors with 100% 4KB aligned offsets", () => {
  const testDir = join(tmpdir(), `vitna_test_ingest_${Date.now()}`);
  mkdirSync(testDir, { recursive: true });

  const srcFile = join(testDir, "synth_model.safetensors");
  const outDir = join(testDir, "dma_out");
  const synth = createSyntheticSafeTensorsFile(srcFile);

  try {
    const dryResult = sliceDmaAlignedSlabs(srcFile, outDir, { dryRun: true });
    assert.equal(dryResult.tensorCount, 3);
    assert.equal(dryResult.records.length, 3);
    for (const rec of dryResult.records) {
      assert.equal(rec.targetOffset % 4096, 0, `Record ${rec.name} must be 4KB aligned`);
    }

    const realResult = sliceDmaAlignedSlabs(srcFile, outDir, { dryRun: false });
    assert.equal(realResult.tensorCount, 3);
    assert.ok(existsSync(realResult.alignedFilePath), "DMA aligned file must exist");
    assert.ok(existsSync(realResult.manifestPath), "Index manifest must exist");
    assert.ok(realResult.airgapHash.length === 64, "SHA-256 hash must be 64 hex chars");

    const check = verifyDmaAlignment(realResult.manifestPath);
    assert.equal(check.valid, true);
    assert.equal(check.errors.length, 0);
    assert.equal(check.tensorCount, 3);

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
      if (existsSync(outDir)) rmSync(outDir, { recursive: true, force: true });
      if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    } catch {}
  }
});

test("sliceShardedDmaSlabs repacks multi-shard checkpoints with per-shard sector alignment", () => {
  const testDir = join(tmpdir(), `vitna_test_sharded_${Date.now()}`);
  mkdirSync(testDir, { recursive: true });

  const synth = createSyntheticShardedSafeTensors(testDir);
  const outDir = join(testDir, "sharded_out");

  try {
    // 1. Dry run
    const dryResult = sliceShardedDmaSlabs([synth.s1_path, synth.s2_path], outDir, {
      modelName: "test_moe_sharded",
      dryRun: true,
    });
    assert.equal(dryResult.totalShards, 2);
    assert.equal(dryResult.tensorCount, 4);
    for (const rec of dryResult.records) {
      assert.equal(rec.targetOffset % 4096, 0);
    }

    // 2. Physical slice
    const realResult = sliceShardedDmaSlabs([synth.s1_path, synth.s2_path], outDir, {
      modelName: "test_moe_sharded",
      dryRun: false,
    });
    assert.equal(realResult.totalShards, 2);
    assert.equal(realResult.tensorCount, 4);
    assert.ok(existsSync(realResult.manifestPath));

    // Verify manifest validation passes
    const check = verifyDmaAlignment(realResult.manifestPath);
    assert.equal(check.valid, true);
    assert.equal(check.errors.length, 0);
    assert.equal(check.tensorCount, 4);

    // Verify shard 0 and shard 1 attribution
    const s0_rec = realResult.records.filter((r) => r.shardIdx === 0);
    const s1_rec = realResult.records.filter((r) => r.shardIdx === 1);
    assert.equal(s0_rec.length, 2);
    assert.equal(s1_rec.length, 2);

    // Verify payload bit-for-bit from shard 1 aligned file
    const s1_alignedFile = join(outDir, "model-00001-of-00002.dma.anchor");
    const s1_bytes = readFileSync(s1_alignedFile);
    const embedRec = s0_rec.find((r) => r.name === "model.embed_tokens.weight");
    assert.ok(embedRec);
    const extractedEmbed = s1_bytes.subarray(embedRec.targetOffset, embedRec.targetOffset + embedRec.rawSize);
    assert.deepEqual(extractedEmbed, synth.s1_data1);
  } finally {
    try {
      if (existsSync(synth.s1_path)) unlinkSync(synth.s1_path);
      if (existsSync(synth.s2_path)) unlinkSync(synth.s2_path);
      if (existsSync(synth.indexPath)) unlinkSync(synth.indexPath);
      if (existsSync(join(outDir, "model-00001-of-00002.dma.anchor"))) unlinkSync(join(outDir, "model-00001-of-00002.dma.anchor"));
      if (existsSync(join(outDir, "model-00002-of-00002.dma.anchor"))) unlinkSync(join(outDir, "model-00002-of-00002.dma.anchor"));
      if (existsSync(join(outDir, "test_moe_sharded.anchor.index.json"))) unlinkSync(join(outDir, "test_moe_sharded.anchor.index.json"));
      if (existsSync(outDir)) rmSync(outDir, { recursive: true, force: true });
      if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    } catch {}
  }
});

test("runModelPull executes local checkpoint ingestion, sharded index, and remote dry-run", async () => {
  const testDir = join(tmpdir(), `vitna_test_pull_${Date.now()}`);
  mkdirSync(testDir, { recursive: true });
  const srcFile = join(testDir, "local_test.safetensors");
  createSyntheticSafeTensorsFile(srcFile);

  const synthSharded = createSyntheticShardedSafeTensors(testDir);

  try {
    // 1. Local single-file pull test
    const localResult = await runModelPull(srcFile, { outDir: join(testDir, "out_single") });
    assert.equal(localResult.tensorCount, 3);
    assert.ok(localResult.airgapHash);

    // 2. Local sharded index pull test
    const shardedResult = await runModelPull(synthSharded.indexPath, { outDir: join(testDir, "out_sharded") });
    assert.equal(shardedResult.totalShards, 2);
    assert.equal(shardedResult.tensorCount, 4);

    // 3. Remote dry-run test
    const remoteResult = await runModelPull("Qwen/Qwen2.5-Coder-7B-Instruct", { dryRun: true });
    assert.equal(remoteResult.dryRun, true);
    assert.equal(remoteResult.sectorSize, 4096);
  } finally {
    try {
      if (existsSync(srcFile)) unlinkSync(srcFile);
      if (existsSync(synthSharded.s1_path)) unlinkSync(synthSharded.s1_path);
      if (existsSync(synthSharded.s2_path)) unlinkSync(synthSharded.s2_path);
      if (existsSync(synthSharded.indexPath)) unlinkSync(synthSharded.indexPath);
      if (existsSync(testDir)) rmSync(testDir, { recursive: true, force: true });
    } catch {}
  }
});
