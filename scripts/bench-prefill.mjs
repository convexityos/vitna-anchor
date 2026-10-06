#!/usr/bin/env node
// Time the engine's prompt (prefill), for one or more engine builds taking
// turns, and print the machine and the command with the result.
//
//   node scripts/bench-prefill.mjs [--device cpu|cuda] [--engine <path> ...] [--model <dir>] [--weights <file.gguf>]
//                                  [--prompt-tokens <n,n,...>] [--runs <n>] [--text <file> ...] [--ctx <n>]
//                                  [--gpu-kv-layers <n>]
//
// --weights reads the weights from a GGUF file, as the engine's own --weights
// does; --model still gives config.json and the tokenizer. --ctx and
// --gpu-kv-layers go to the engine as they are.
//
// Each run is one `generate --greedy --timing --max-new 2` over the first n
// tokens of the text, README.md unless --text names files, which are read
// one after another with a blank line between; the ids go to the engine in
// a file (--ids-file), since 32K of them are more than a Windows command
// line holds. The commit the files were read at is printed, so the prompt
// can be had again. The engine's own timing line gives the prompt's
// time: from its first token to the next token's logits on the host, with
// loading and the upload to a GPU left out. Every round runs every build once
// at a length, so builds compared with each other see the same interference
// from whatever else the machine is doing. The median of --runs rounds
// (default 5) is reported, after one round thrown away to warm up, with each
// build's speed relative to the first, and whether every build chose the same
// two tokens.
//
// What it prints was measured on the machine it ran on, with those engine
// builds, and says nothing about another. Zero external dependencies.

import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { cpus, release, tmpdir, type } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const EXE = process.platform === "win32" ? ".exe" : "";

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

function options(name) {
  const out = [];
  process.argv.forEach((a, i) => {
    if (a === name && i + 1 < process.argv.length) out.push(process.argv[i + 1]);
  });
  return out;
}

const device = option("--device", "cpu");
const model = resolve(option("--model", join(ROOT, "models", "smollm2-135m")));
const weights = option("--weights", "") ? resolve(option("--weights", "")) : "";
const lengths = option("--prompt-tokens", "8,100,500,2000").split(",").map(Number);
const runs = Number(option("--runs", "5"));
const texts = (options("--text").length ? options("--text") : ["README.md"]).map((p) => resolve(ROOT, p));
const passed = ["--ctx", "--gpu-kv-layers"].flatMap((name) => (option(name, "") ? [name, option(name, "")] : []));
const fallback = [
  process.env.VITNA_ENGINE,
  join(ROOT, `engine/vitna-anchor${EXE}`),
  join(ROOT, `engine/build/Release/vitna-anchor${EXE}`),
  join(ROOT, `engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const engines = (options("--engine").length ? options("--engine") : [fallback ?? ""]).map((p) => resolve(p));

if (!["cpu", "cuda"].includes(device)) throw new Error(`--device must be cpu or cuda, not ${device}`);
for (const e of engines) if (!existsSync(e)) throw new Error(`no built engine at ${e}; pass --engine <path>`);
if (!Number.isInteger(runs) || runs < 1) throw new Error("--runs must be at least 1");

const text = texts.map((p) => readFileSync(p, "utf8")).join("\n\n");
const tok = spawnSync(engines[0], ["tokenize", "--model", model], {
  input: JSON.stringify(text) + "\n",
  encoding: "utf8",
  maxBuffer: 1 << 26,
});
if (tok.status !== 0) throw new Error(tok.stderr);
const ids = JSON.parse(tok.stdout.trim());
for (const n of lengths) {
  if (!Number.isInteger(n) || n < 1 || n > ids.length) throw new Error(`--prompt-tokens must each be 1 to ${ids.length}`);
}
const scratch = mkdtempSync(join(tmpdir(), "bench-prefill-"));
process.on("exit", () => rmSync(scratch, { recursive: true, force: true }));
const idsFile = join(scratch, "ids.txt");

function once(engine, n) {
  writeFileSync(idsFile, ids.slice(0, n).join(","));
  const args = ["generate", "--model", model, ...(weights ? ["--weights", weights] : []), "--ids-file", idsFile, "--max-new", "2",
    "--greedy", "--timing", "--device", device, ...passed];
  const r = spawnSync(engine, args, { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`${engine} exited ${r.status}: ${r.stderr}`);
  const m = r.stderr.match(/timing: (\d+) prompt tokens in ([\d.]+) ms/);
  if (!m) throw new Error(`no timing line in: ${r.stderr}`);
  return { ms: Number(m[2]), ids: JSON.parse(r.stdout).ids.join(",") };
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

let gpu = "";
if (device === "cuda") {
  const q = spawnSync("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"], { encoding: "utf8" });
  gpu = q.status === 0 ? q.stdout.trim().split("\n")[0] : "nvidia-smi not available";
}
const commit = spawnSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).stdout.trim() || "unknown";
// The files are read from the working tree, which is the commit's only if none of them has changed since.
const changed = spawnSync("git", ["status", "--porcelain", "--", ...texts], { cwd: ROOT, encoding: "utf8" }).stdout.trim() !== "";
const named = texts.map((p) => p.slice(ROOT.length + 1).replace(/\\/g, "/")).join(", then ");
console.log(`command: <engine> generate --model ${model}${weights ? ` --weights ${weights}` : ""} --ids-file <the first n tokens of ${named}> --max-new 2 --greedy --timing --device ${device}${passed.length ? " " + passed.join(" ") : ""}`);
console.log(`text: ${named} at commit ${commit}${changed ? ", with changes not committed" : ""}, ${ids.length} tokens`);
console.log(`machine: ${cpus()[0].model.trim()}, ${type()} ${release()}${gpu ? `; GPU ${gpu}` : ""}`);
engines.forEach((e, i) => console.log(`engine ${i + 1}: ${e}`));

for (const n of lengths) {
  const times = engines.map(() => []);
  const chosen = new Set();
  for (let r = 0; r <= runs; r++) {
    engines.forEach((e, i) => {
      const res = once(e, n);
      chosen.add(res.ids);
      if (r > 0) times[i].push(res.ms);
    });
  }
  console.log(`\n${n}-token prompt${engines.length > 1 ? `; every engine chose the same two tokens: ${chosen.size === 1}` : ""}`);
  const base = median(times[0]);
  times.forEach((t, i) => {
    const med = median(t);
    const rel = i === 0 ? "" : `, ${(base / med).toFixed(2)}x engine 1`;
    console.log(`  engine ${i + 1}: median ${med.toFixed(2)} ms, ${((n / med) * 1000).toFixed(0)} tokens/s${rel}; range ${Math.min(...t).toFixed(2)} to ${Math.max(...t).toFixed(2)}`);
  });
}
