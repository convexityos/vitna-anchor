#!/usr/bin/env node
// Time the engine's decoding, one token at a time, and print the machine and
// the commands with the result.
//
//   node scripts/bench-decode.mjs [--device cpu|cuda] [--engine <path>] [--model <dir>]
//                                 [--prompt <text>] [--tokens <n>] [--runs <n>]
//
// Each run times `generate --greedy` twice: once for 32 new tokens and once
// for 32 + --tokens (default 256). The difference, divided by --tokens, is
// the time per generated token, with loading, the upload to a GPU and start-up
// taken out. The median of --runs (default 3) runs is reported.
//
// What it prints was measured on the machine it ran on, with that engine
// build, and says nothing about another. Zero external dependencies.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { cpus, release, type } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const EXE = process.platform === "win32" ? ".exe" : "";

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const device = option("--device", "cpu");
const model = resolve(option("--model", join(ROOT, "models", "smollm2-135m")));
const prompt = option("--prompt", "The capital of France is");
const tokens = Number(option("--tokens", "256"));
const runs = Number(option("--runs", "3"));
const engine = resolve(
  option("--engine", "") ||
    [
      process.env.VITNA_ENGINE,
      join(ROOT, `engine/vitna-anchor${EXE}`),
      join(ROOT, `engine/build/Release/vitna-anchor${EXE}`),
      join(ROOT, `engine/build/vitna-anchor${EXE}`),
    ].filter(Boolean).find((p) => existsSync(p)) || "",
);
const BASE = 32;

if (!["cpu", "cuda"].includes(device)) throw new Error(`--device must be cpu or cuda, not ${device}`);
if (!existsSync(engine)) throw new Error("no built engine found; pass --engine <path>");
if (!Number.isInteger(tokens) || tokens < 1 || !Number.isInteger(runs) || runs < 1) throw new Error("--tokens and --runs must be positive integers");

function args(n) {
  return ["generate", "--model", model, "--prompt", prompt, "--max-new", String(n), "--greedy", "--device", device];
}

function timed(n) {
  const t0 = process.hrtime.bigint();
  const r = spawnSync(engine, args(n), { encoding: "utf8", maxBuffer: 1 << 24 });
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  if (r.status !== 0) throw new Error(`${engine} ${args(n).join(" ")} exited ${r.status}: ${r.stderr}`);
  const produced = JSON.parse(r.stdout).ids.length;
  if (produced !== n) throw new Error(`asked for ${n} tokens and got ${produced}`);
  return ms;
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

const perToken = [];
for (let i = 0; i < runs; i++) {
  const short = timed(BASE);
  const long = timed(BASE + tokens);
  perToken.push((long - short) / tokens);
}
const ms = median(perToken);

let gpu = "";
if (device === "cuda") {
  const q = spawnSync("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"], { encoding: "utf8" });
  gpu = q.status === 0 ? q.stdout.trim().split("\n")[0] : "nvidia-smi not available";
}

console.log(`engine:  ${engine}`);
console.log(`command: ${[engine, ...args(BASE + tokens)].join(" ")}, less the same with --max-new ${BASE}`);
console.log(`machine: ${cpus()[0].model.trim()}, ${type()} ${release()}${gpu ? `; GPU ${gpu}` : ""}`);
console.log(`runs:    ${perToken.map((x) => x.toFixed(2)).join(", ")} ms per token`);
console.log(`median:  ${ms.toFixed(2)} ms per token, ${(1000 / ms).toFixed(0)} tokens/s, measured over the ${tokens} tokens after the first ${BASE}`);
