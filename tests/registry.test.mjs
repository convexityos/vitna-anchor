import test from "node:test";
import assert from "node:assert/strict";

import {
  listRegistryModels,
  getRegistryModel,
  generateModelManifest,
  formatRegistryTable,
  VERIFIED_DMA_REGISTRY,
} from "../runtime/registry.mjs";

test("listRegistryModels returns verified model catalog", () => {
  const models = listRegistryModels();
  assert.ok(Array.isArray(models));
  assert.ok(models.length >= 4);

  const qwen = models.find((m) => m.id === "qwen2.5-coder-7b-dma");
  assert.ok(qwen);
  assert.equal(qwen.sectorSizeBytes, 4096);
  assert.equal(qwen.precision, "INT4");

  const deepseek = models.find((m) => m.id === "deepseek-v3-671b-dma");
  assert.ok(deepseek);
  assert.equal(deepseek.isMoe, true);
  assert.equal(deepseek.expertCount, 256);
});

test("getRegistryModel resolves exact ids and known aliases", () => {
  assert.ok(getRegistryModel("qwen2.5-coder-7b-dma"));
  assert.ok(getRegistryModel("qwen7b"));
  assert.ok(getRegistryModel("deepseek671b"));
  assert.ok(getRegistryModel("llama70b"));
  assert.equal(getRegistryModel("unknown-model-xyz"), null);
});

test("generateModelManifest formats valid 1.0.0 JSON manifest with airgap proof", () => {
  const manifest = generateModelManifest("qwen2.5-coder-7b-dma");
  assert.equal(manifest.manifestVersion, "1.0.0");
  assert.equal(manifest.engine, "vitna-anchor");
  assert.equal(manifest.model.id, "qwen2.5-coder-7b-dma");
  assert.equal(manifest.airgapProof.socketEgressBytes, 0);

  const table = formatRegistryTable();
  assert.ok(table.includes("SOVEREIGN MODEL HUB"));
  assert.ok(table.includes("qwen2.5-coder-7b-dma"));
  assert.ok(table.includes("deepseek-v3-671b-dma"));
});
