// Gate A10: a long context on Qwen3-30B-A3B, on the GPU, from Qwen's Q4_K_M
// file.
//
// A layer whose key-value cache the GPU has no room for keeps it in
// page-locked host memory, and its attention runs on a copy staged on the
// device (--gpu-kv-layers sets how many layers stay on the GPU), so the
// logits must be the same, bit for bit, wherever the cache is. Rows
// attending to more than 1,024 positions take a long context's attention,
// which tests/reference-qwen3-gguf.test.mjs checks against gate A9's
// reference. Here a short prompt, and one that crosses 1,024 positions, run
// with every layer's cache on the GPU, with none there and with half, and
// a token at a time; and a prompt of 32,768 tokens runs where the GPU has
// no room for its whole cache, as it does with none of it there.
//
// The long prompts are the first tokens of README.md, then
// reference/README.md, then engine/src/model_cuda.cu, read at the commit
// the test runs at: the comparisons are of the engine with itself.
//
// They need the engine built with CUDA, gate A9's model directory and its
// Q4_K_M file, and run only with VITNA_REQUIRE_QWEN3_GGUF=1 and
// VITNA_DEVICE=cuda; the 32,768-token prompt only with VITNA_REQUIRE_LONG=1
// as well, since it runs twice, minutes each.

import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const modelDir = process.env.ANCHOR_QWEN3_MODEL_DIR || here("../models/qwen3-30b-a3b/");
const ggufPath = join(process.env.ANCHOR_QWEN3_GGUF_DIR || here("../models/qwen3-30b-a3b-gguf/"), "Qwen3-30B-A3B-Q4_K_M.gguf");
const LAYERS = 48;

const needs = process.env.VITNA_REQUIRE_QWEN3_GGUF !== "1"
  ? "these run only with VITNA_REQUIRE_QWEN3_GGUF=1: the file takes much of a machine's memory"
  : process.env.VITNA_DEVICE !== "cuda" ? "VITNA_DEVICE is not cuda" : false;
const ON_GPU = { skip: needs };
const LONG = { skip: needs || (process.env.VITNA_REQUIRE_LONG !== "1" && "the 32,768-token prompt runs only with VITNA_REQUIRE_LONG=1") };
const WEIGHTS = ["--model", modelDir, "--weights", ggufPath, "--device", "cuda"];

const runEngine = (args, env = {}, input = "") => new Promise((resolve, reject) => {
  const child = spawn(engine, args, { env: { ...process.env, ...env } });
  let out = "";
  let err = "";
  child.stdout.setEncoding("utf8").on("data", (d) => (out += d));
  child.stderr.setEncoding("utf8").on("data", (d) => (err += d));
  child.on("error", reject);
  child.on("close", (code) => (code === 0 ? resolve({ out, err }) : reject(new Error(`${engine} ${args.join(" ")} exited ${code}: ${err}`))));
  child.stdin.end(input);
});

// The ids of the text the long prompts are cut from, by the model's own tokenizer.
let text = null;
const textIds = async () => {
  if (!text) {
    const body = ["../README.md", "../reference/README.md", "../engine/src/model_cuda.cu"].map((p) => readFileSync(here(p), "utf8")).join("\n\n");
    text = JSON.parse((await runEngine(["tokenize", "--model", modelDir], {}, JSON.stringify(body) + "\n")).out.trim());
  }
  return text;
};

const withDir = async (prefix, fn) => {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  try {
    return await fn((name) => join(dir, name));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
};

test("a short prompt's logits are the same, byte for byte, with every layer's cache on the GPU, none and half, in either attention", ON_GPU, async (t) => {
  await withDir("vitna-long-short-", async (at) => {
    const ids = (await textIds()).slice(0, 300);
    writeFileSync(at("ids"), ids.join(","));
    const placements = [["gpu", []], ["host", ["--gpu-kv-layers", "0"]], ["half", ["--gpu-kv-layers", String(LAYERS / 2)]]];
    for (const [arith, env] of [["the split", {}], ["a long context's", { VITNA_TEST_LONG_FROM: "0" }]]) {
      const files = [];
      for (const [name, extra] of placements) {
        await runEngine(["logits", ...WEIGHTS, "--ids-file", at("ids"), "--ctx", "512", "--out", at(name), ...extra], env);
        files.push([name, readFileSync(at(name))]);
      }
      for (const [name, bytes] of files.slice(1)) {
        assert.ok(bytes.length > 0 && bytes.equals(files[0][1]), `${arith} attention: ${name}'s logits are not the GPU's own`);
      }
    }
    t.diagnostic(`300 positions: with the split attention and a long context's, the logits with no layer's cache on the GPU, and with ${LAYERS / 2}, equal those with every one there, byte for byte`);
  });
});

test("with --precision fast, a prompt's logits are the same, byte for byte, wherever the cache is", ON_GPU, async (t) => {
  await withDir("vitna-long-fast-", async (at) => {
    const ids = (await textIds()).slice(0, 2000);
    writeFileSync(at("ids"), ids.join(","));
    const files = [];
    for (const [name, extra] of [["gpu", []], ["host", ["--gpu-kv-layers", "0"]]]) {
      const { out } = await runEngine(["generate", ...WEIGHTS, "--ids-file", at("ids"), "--ctx", "2048", "--max-new", "8", "--greedy",
        "--precision", "fast", "--logits-out", at(name), ...extra]);
      files.push([name, JSON.parse(out).ids, readFileSync(at(name))]);
    }
    assert.deepEqual(files[1][1], files[0][1], "with no layer's cache on the GPU, it chose other tokens");
    assert.ok(files[0][2].length > 0 && files[1][2].equals(files[0][2]), "with no layer's cache on the GPU, the logits differ");
    t.diagnostic("2,000 positions in --precision fast: 8 greedy tokens and their logits equal, byte for byte, with no layer's cache on the GPU");
  });
});

test("a prompt past 1,024 positions: 8 greedy tokens and their logits are the same, byte for byte, wherever the cache is, and as rows or a token at a time", ON_GPU, async (t) => {
  await withDir("vitna-long-4k-", async (at) => {
    const ids = (await textIds()).slice(0, 2000);
    writeFileSync(at("ids"), ids.join(","));
    const runs = [
      ["gpu", [], {}],
      ["host", ["--gpu-kv-layers", "0"], {}],
      ["half", ["--gpu-kv-layers", String(LAYERS / 2)], {}],
      ["steps", [], { VITNA_TEST_NO_ROWS: "1" }],
    ];
    const results = [];
    for (const [name, extra, env] of runs) {
      const { out } = await runEngine(["generate", ...WEIGHTS, "--ids-file", at("ids"), "--ctx", "2048", "--max-new", "8", "--greedy",
        "--logits-out", at(name), ...extra], env);
      results.push([name, JSON.parse(out).ids, readFileSync(at(name))]);
    }
    const [, gpuIds, gpuLogits] = results[0];
    assert.equal(gpuIds.length, 8);
    for (const [name, got, logits] of results.slice(1)) {
      assert.deepEqual(got, gpuIds, `${name} chose other tokens`);
      assert.ok(logits.length > 0 && logits.equals(gpuLogits), `${name}'s logits are not the GPU's own`);
    }
    t.diagnostic(`2,000 positions, the last 976 in a long context's attention: 8 greedy tokens and their logits equal, byte for byte, with no layer's cache on the GPU, with ${LAYERS / 2}, and a token at a time`);
  });
});

test("a 32,768-token prompt runs where the GPU has no room for its whole cache, and gives the logits that no layer's cache on the GPU gives, byte for byte", LONG, async (t) => {
  await withDir("vitna-long-32k-", async (at) => {
    const ids = (await textIds()).slice(0, 32768);
    assert.equal(ids.length, 32768, "the text is shorter than 32,768 tokens");
    writeFileSync(at("ids"), ids.join(","));
    const run = (name, extra) => runEngine(["generate", ...WEIGHTS, "--ids-file", at("ids"), "--ctx", "32784", "--max-new", "16", "--greedy",
      "--timing", "--logits-out", at(name), ...extra]);
    const auto = await run("auto", []);
    const room = auto.err.match(/The GPU has room for the key-value cache of (\d+) of the model's (\d+) layers/);
    assert.ok(room, "the GPU held the whole cache, so nothing was in host memory");
    const host = await run("host", ["--gpu-kv-layers", "0"]);
    const a = JSON.parse(auto.out).ids;
    assert.equal(a.length, 16);
    assert.deepEqual(JSON.parse(host.out).ids, a, "with no layer's cache on the GPU, it chose other tokens");
    const la = readFileSync(at("auto"));
    assert.ok(la.length > 0 && la.equals(readFileSync(at("host"))), "with no layer's cache on the GPU, the logits differ");
    const f = new Float32Array(la.buffer, la.byteOffset, la.byteLength / 4);
    assert.ok(f.every(Number.isFinite), "a logit is not finite");
    const timing = (r) => (r.err.match(/timing: .*/) ?? ["no timing line"])[0];
    t.diagnostic(`the GPU held the cache of ${room[1]} of ${room[2]} layers; 16 greedy tokens and their logits equal those with none there, byte for byte`);
    t.diagnostic(`as it held it: ${timing(auto)}`);
    t.diagnostic(`with none there: ${timing(host)}`);
  });
});
