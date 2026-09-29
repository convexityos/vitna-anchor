#!/usr/bin/env node
// Time the engine's decoding, one token at a time, and print the machine and
// the command with the result.
//
//   node scripts/bench-decode.mjs [--device cpu|cuda] [--engine <path>] [--model <dir>]
//                                 [--prompt <text> | --prompt-tokens <n>]
//                                 [--tokens <n>] [--runs <n>]
//
// Each run is one `generate --greedy --timing`, which times the tokens after
// the first new one inside the engine: each is one forward step and one
// choice, with loading, the upload to a GPU and the prompt left out. The
// median of --runs runs (default 5) is reported, after one run thrown away
// to warm up. --prompt-tokens takes the first n tokens of README.md as the
// prompt, to time decoding that far into the context.
//
// What it prints was measured on the machine it ran on, with that engine
// build, and says nothing about another. Zero external dependencies.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
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
const promptTokens = option("--prompt-tokens", null);
const prompt = option("--prompt", "The capital of France is");
const tokens = Number(option("--tokens", "256"));
const runs = Number(option("--runs", "5"));
const engine = resolve(
  option("--engine", "") ||
    [
      process.env.VITNA_ENGINE,
      join(ROOT, `engine/vitna-anchor${EXE}`),
      join(ROOT, `engine/build/Release/vitna-anchor${EXE}`),
      join(ROOT, `engine/build/vitna-anchor${EXE}`),
    ].filter(Boolean).find((p) => existsSync(p)) || "",
);

if (!["cpu", "cuda"].includes(device)) throw new Error(`--device must be cpu or cuda, not ${device}`);
if (!existsSync(engine)) throw new Error("no built engine found; pass --engine <path>");
if (!Number.isInteger(tokens) || tokens < 2 || !Number.isInteger(runs) || runs < 1) throw new Error("--tokens must be at least 2, and --runs at least 1");

let promptArgs = ["--prompt", prompt];
let promptLength = null;
if (promptTokens !== null) {
  const n = Number(promptTokens);
  const tok = spawnSync(engine, ["tokenize", "--model", model], {
    input: JSON.stringify(readFileSync(join(ROOT, "README.md"), "utf8")) + "\n",
    encoding: "utf8",
    maxBuffer: 1 << 26,
  });
  if (tok.status !== 0) throw new Error(tok.stderr);
  const ids = JSON.parse(tok.stdout.trim());
  if (!Number.isInteger(n) || n < 1 || n > ids.length) throw new Error(`--prompt-tokens must be 1 to ${ids.length}`);
  promptArgs = ["--ids", ids.slice(0, n).join(",")];
  promptLength = n;
}

const args = ["generate", "--model", model, ...promptArgs, "--max-new", String(tokens), "--greedy", "--timing", "--device", device];

function once() {
  const r = spawnSync(engine, args, { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`${engine} exited ${r.status}: ${r.stderr}`);
  const m = r.stderr.match(/timing: (\d+) prompt tokens in [\d.]+ ms; (\d+) tokens after the first new one in [\d.]+ ms, ([\d.]+) ms each/);
  if (!m) throw new Error(`no timing line in: ${r.stderr}`);
  promptLength = Number(m[1]);
  return Number(m[3]);
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

once(); // warm-up, thrown away
const perToken = [];
for (let i = 0; i < runs; i++) perToken.push(once());
const ms = median(perToken);

let gpu = "";
if (device === "cuda") {
  const q = spawnSync("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"], { encoding: "utf8" });
  gpu = q.status === 0 ? q.stdout.trim().split("\n")[0] : "nvidia-smi not available";
}
const shown = promptTokens !== null ? ["--ids", `<the first ${promptTokens} tokens of README.md>`] : ["--prompt", JSON.stringify(prompt)];
const command = ["generate", "--model", model, ...shown, "--max-new", String(tokens), "--greedy", "--timing", "--device", device];

console.log(`engine:  ${engine}`);
console.log(`command: ${[engine, ...command].join(" ")}`);
console.log(`machine: ${cpus()[0].model.trim()}, ${type()} ${release()}${gpu ? `; GPU ${gpu}` : ""}`);
console.log(`runs:    ${perToken.map((x) => x.toFixed(3)).join(", ")} ms per token`);
console.log(`median:  ${ms.toFixed(3)} ms per token, ${(1000 / ms).toFixed(0)} tokens/s, measured over the ${tokens - 1} steps at positions ${promptLength} to ${promptLength + tokens - 2}`);
