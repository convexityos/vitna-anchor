#!/usr/bin/env node
// Head-to-head comparison between vitna-anchor and llama.cpp (llama-bench) on
// identical GGUF checkpoints, CPU and CUDA offload, with zero external deps.
//
// Usage:
//   node scripts/bench-llama-head-to-head.mjs [options]
//
// Options:
//   --engine <path>        vitna-anchor binary
//   --llama-bench <path>   llama-bench binary
//   --models-dir <dir>     root holding olmoe-1b-7b{,-gguf} and qwen3-30b-a3b{,-gguf}
//   --runs <n>             timed runs per configuration (default 3)
//   --threads <n>          CPU thread count passed to llama-bench (default 8)
//   --only <csv>           subset of cases: olmoe-cuda,olmoe-cpu,qwen3-cuda,qwen3-cpu
//   --json                 emit machine-readable JSON on stdout instead of text

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { cpus, release, type } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { summarize } from "../bench/stats.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const EXE = process.platform === "win32" ? ".exe" : "";

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const asJson = process.argv.includes("--json");
const runs = Number(option("--runs", "3"));
const threads = Number(option("--threads", "8"));
const onlyRaw = option("--only", "");
const only = onlyRaw ? new Set(onlyRaw.split(",").map((s) => s.trim()).filter(Boolean)) : null;

const engine = resolve(
  option("--engine", "") ||
    [
      process.env.VITNA_ENGINE,
      join(ROOT, `engine/vitna-anchor${EXE}`),
      join(ROOT, `engine/build/Release/vitna-anchor${EXE}`),
      join(ROOT, `engine/build/vitna-anchor${EXE}`),
    ]
      .filter(Boolean)
      .find((p) => existsSync(p)) ||
    "",
);

const llamaBench = resolve(
  option("--llama-bench", "") ||
    [
      process.env.LLAMA_BENCH,
      "D:/vitna-tools/llama-b11433/bin/llama-bench.exe",
      "llama-bench",
    ]
      .filter(Boolean)
      .find((p) => existsSync(p)) ||
    "",
);

const modelsRoot = resolve(option("--models-dir", "D:/vitna-models"));

if (!existsSync(engine)) throw new Error(`vitna-anchor binary not found at ${engine}`);
if (!existsSync(llamaBench)) throw new Error(`llama-bench binary not found; pass --llama-bench <path>`);
if (!Number.isInteger(runs) || runs < 1) throw new Error("--runs must be a positive integer");
if (!Number.isInteger(threads) || threads < 1) throw new Error("--threads must be a positive integer");

function runAnchorOnce({ modelDir, weights, device, tokens, ctx, precision }) {
  const args = [
    "generate",
    "--model",
    modelDir,
    "--weights",
    weights,
    "--prompt",
    "The capital of France is",
    "--max-new",
    String(tokens),
    "--greedy",
    "--timing",
    "--ctx",
    String(ctx),
    ...(device === "cuda" ? ["--device", "cuda"] : []),
    ...(precision ? ["--precision", precision] : []),
  ];
  const r = spawnSync(engine, args, { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) {
    throw new Error(`vitna-anchor failed (${r.status}): ${r.stderr}`);
  }
  const m = r.stderr.match(
    /timing: (\d+) prompt tokens in ([\d.]+) ms; (\d+) tokens after the first new one in ([\d.]+) ms, ([\d.]+) ms each/,
  );
  if (!m) throw new Error(`missing timing line in stderr: ${r.stderr}`);
  const msEach = Number(m[5]);
  return {
    msEach,
    toksPerSec: 1000 / msEach,
  };
}

function runLlamaBench({ weights, ngl, ncm, genTokens, flashAttn }) {
  const args = [
    "-m",
    weights,
    "-ngl",
    String(ngl),
    ...(ncm !== null ? ["-ncmoe", String(ncm)] : []),
    ...(flashAttn !== null ? ["-fa", String(flashAttn)] : []),
    "-t",
    String(threads),
    "-p",
    "0",
    "-n",
    String(genTokens),
    "-r",
    String(runs),
    "-o",
    "json",
  ];
  const r = spawnSync(llamaBench, args, { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) {
    throw new Error(`llama-bench failed (${r.status}): ${r.stderr}`);
  }
  const parsed = JSON.parse(r.stdout);
  const row = parsed[0];
  return {
    buildNumber: row.build_number,
    buildCommit: row.build_commit,
    samplesTs: row.samples_ts,
    avgTs: row.avg_ts,
    args,
  };
}

const CASES = [
  {
    id: "olmoe-cuda",
    model: "OLMoE-1B-7B",
    weightsLabel: "Q4_K_M",
    device: "cuda",
    modelDir: join(modelsRoot, "olmoe-1b-7b"),
    weights: join(modelsRoot, "olmoe-1b-7b-gguf", "olmoe-1b-7b-0924-q4_k_m.gguf"),
    tokens: 128,
    ctx: 512,
    llama: { ngl: 99, ncm: null, flashAttn: 1 },
  },
  {
    id: "olmoe-cpu",
    model: "OLMoE-1B-7B",
    weightsLabel: "Q4_K_M",
    device: "cpu",
    modelDir: join(modelsRoot, "olmoe-1b-7b"),
    weights: join(modelsRoot, "olmoe-1b-7b-gguf", "olmoe-1b-7b-0924-q4_k_m.gguf"),
    tokens: 32,
    ctx: 512,
    llama: { ngl: 0, ncm: null, flashAttn: null },
  },
  {
    id: "qwen3-cuda",
    model: "Qwen3-30B-A3B",
    weightsLabel: "Q4_K_M",
    device: "cuda",
    modelDir: join(modelsRoot, "qwen3-30b-a3b"),
    weights: join(modelsRoot, "qwen3-30b-a3b-gguf", "Qwen3-30B-A3B-Q4_K_M.gguf"),
    tokens: 128,
    ctx: 512,
    llama: { ngl: 99, ncm: 30, flashAttn: 1 },
  },
  {
    id: "qwen3-cpu",
    model: "Qwen3-30B-A3B",
    weightsLabel: "Q4_K_M",
    device: "cpu",
    modelDir: join(modelsRoot, "qwen3-30b-a3b"),
    weights: join(modelsRoot, "qwen3-30b-a3b-gguf", "Qwen3-30B-A3B-Q4_K_M.gguf"),
    tokens: 16,
    ctx: 512,
    llama: { ngl: 0, ncm: null, flashAttn: null },
  },
];

const selected = only ? CASES.filter((c) => only.has(c.id)) : CASES;
for (const c of selected) {
  if (!existsSync(c.modelDir)) throw new Error(`missing model directory: ${c.modelDir}`);
  if (!existsSync(c.weights)) throw new Error(`missing GGUF weights: ${c.weights}`);
}

const gpuQuery = spawnSync(
  "nvidia-smi",
  ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
  { encoding: "utf8" },
);

const report = {
  measuredAt: new Date().toISOString(),
  machine: `${cpus()[0].model.trim()}, ${type()} ${release()}`,
  gpu: gpuQuery.status === 0 ? gpuQuery.stdout.trim().split("\n")[0] : "unavailable",
  engine,
  llamaBench,
  runs,
  threads,
  results: [],
};

for (const item of selected) {
  if (!asJson) console.error(`Running ${item.id} (${item.model} ${item.weightsLabel}, ${item.device}, n=${item.tokens}, r=${runs})...`);
  // Warm OS file cache once before timed vitna-anchor runs, matching bench-decode/bench-experts.
  runAnchorOnce(item);
  const anchorRuns = [];
  for (let i = 0; i < runs; i++) {
    anchorRuns.push(runAnchorOnce(item));
  }
  const anchorToks = anchorRuns.map((r) => Number(r.toksPerSec.toFixed(4)));
  const anchorMs = anchorRuns.map((r) => Number(r.msEach.toFixed(4)));
  const llama = runLlamaBench({
    weights: item.weights,
    ngl: item.llama.ngl,
    ncm: item.llama.ncm,
    genTokens: item.tokens,
    flashAttn: item.llama.flashAttn,
  });
  report.results.push({
    id: item.id,
    model: item.model,
    weights: item.weightsLabel,
    device: item.device,
    tokens: item.tokens,
    anchor: {
      runsToksPerSec: anchorToks,
      runsMsEach: anchorMs,
      medianToksPerSec: summarize(anchorToks, "median").value,
      medianMsEach: summarize(anchorMs, "median").value,
    },
    llamaCpp: {
      buildNumber: llama.buildNumber,
      buildCommit: llama.buildCommit,
      runsToksPerSec: llama.samplesTs,
      meanToksPerSec: summarize(llama.samplesTs, "mean").value,
      medianToksPerSec: summarize(llama.samplesTs, "median").value,
      args: llama.args,
    },
  });
}

if (asJson) {
  console.log(JSON.stringify(report, null, 2));
} else {
  console.log(`machine: ${report.machine}; GPU: ${report.gpu}`);
  for (const r of report.results) {
    console.log(
      `${r.id}: anchor median=${r.anchor.medianToksPerSec.toFixed(2)} tok/s (${r.anchor.medianMsEach.toFixed(2)} ms/tok, runs=${r.anchor.runsToksPerSec.join(",")}) | ` +
        `llama.cpp b${r.llamaCpp.buildNumber} mean=${r.llamaCpp.meanToksPerSec.toFixed(2)} tok/s (median=${r.llamaCpp.medianToksPerSec.toFixed(2)}, runs=${r.llamaCpp.runsToksPerSec.join(",")})`,
    );
  }
}
