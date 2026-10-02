#!/usr/bin/env node
// Time decoding a mixture of experts with its experts read from the drive,
// at several sizes of the expert cache, and print the machine, the drive and
// the command with the results.
//
//   node scripts/bench-experts.mjs [--engine <path>] [--model <dir>] [--prompt <text>]
//                                  [--tokens <n>] [--runs <n>] [--caches <MiB,MiB,...>] [--no-mapped]
//                                  [--device cpu|cuda] [--gpu-caches <MiB|default,...>]
//
// Each run is one `generate --greedy --timing --expert-cache <MiB>`, which
// times the tokens after the first new one inside the engine. The expert
// cache starts empty in every run: it is the engine's own memory, and the
// reads bypass the operating system's file cache, so nothing carries from
// one run to the next. The mapped run, without --expert-cache, reads the
// experts through the file cache instead; one is run first to warm it, and
// thrown away. The median of --runs runs (default 3) is reported.
//
// With --device cuda the experts run on the GPU, copied into a cache there
// as layers want them, at each size --gpu-caches names (default: the
// engine's own choice, what the device has free less 512 MiB, then 4096,
// 2048 and 1024 MiB), from the mapped checkpoint and, for each size
// --caches names (none by default on the GPU), from a cache that size in
// memory. The GPU's cache starts empty in every run too. Loading, which
// registers the memory the copies come from, is not timed.
//
// What it prints was measured on the machine and the drive it ran on, with
// that engine build, and says nothing about another. Zero external dependencies.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { cpus, release, totalmem, type } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const EXE = process.platform === "win32" ? ".exe" : "";

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}
const list = (s) => (s ? s.split(",").filter(Boolean) : []);

const device = option("--device", "cpu");
const gpu = device === "cuda";
const model = resolve(option("--model", join(ROOT, "models", "olmoe-1b-7b")));
const prompt = option("--prompt", "The capital of France is");
const tokens = Number(option("--tokens", "64"));
const runs = Number(option("--runs", "3"));
const caches = list(option("--caches", gpu ? "" : "512,1024,2048,4096,8192,16384")).map(Number);
const gpuCaches = gpu ? list(option("--gpu-caches", "default,4096,2048,1024")).map((c) => (c === "default" ? 0 : Number(c))) : [null];
const mapped = !process.argv.includes("--no-mapped");
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
if (caches.some((c) => !Number.isInteger(c) || c < 1)) throw new Error("--caches is a list of sizes in MiB");
if (gpu && gpuCaches.some((c) => !Number.isInteger(c) || c < 0)) throw new Error("--gpu-caches is a list of sizes in MiB, or default");

const base = ["generate", "--model", model, "--prompt", prompt, "--max-new", String(tokens), "--greedy", "--timing", "--ctx", "512",
  ...(gpu ? ["--device", "cuda"] : [])];

const num = (s) => Number(s);
function once(cache, gpuCache) {
  const args = [...base];
  if (cache) args.push("--expert-cache", String(cache));
  if (gpuCache) args.push("--gpu-expert-cache", String(gpuCache));
  const r = spawnSync(engine, args, { encoding: "utf8", maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`${engine} exited ${r.status}: ${r.stderr}`);
  const t = r.stderr.match(/timing: (\d+) prompt tokens in [\d.]+ ms; (\d+) tokens after the first new one in [\d.]+ ms, ([\d.]+) ms each/);
  if (!t) throw new Error(`no timing line in: ${r.stderr}`);
  const run = { prompt: num(t[1]), steps: num(t[2]), ms: num(t[3]) };
  const e = r.stderr.match(
    /experts: (\d+) acquired: (\d+) already read, (\d+) still being read for a prefetch, (\d+) read when asked for\. (\d+) prefetched, (\d+) of those used\. ([\d.]+) MiB read in (\d+) reads, (\d+) ms of reading summed over (\d+) threads, (\d+) ms waited for\. The lookahead named ([\d.]+)%/,
  );
  if (cache && !e) throw new Error(`no expert report in: ${r.stderr}`);
  if (e) {
    run.acquired = num(e[1]);
    run.ready = num(e[2]);
    run.inFlight = num(e[3]);
    run.missed = num(e[4]);
    run.mib = num(e[7]);
    run.readMs = num(e[9]);
    run.lookahead = num(e[12]);
  }
  const g = r.stderr.match(
    /experts on the GPU: (\d+) slots, (\d+) MiB, copied from [^.]+\. (\d+) acquired: (\d+) already on the GPU, (\d+) still being copied for a guess, (\d+) copied when asked for\. (\d+) copied for a guess, (\d+) of those used\. ([\d.]+) MiB copied in (\d+) copies\. The lookahead named ([\d.]+)%/,
  );
  if (gpu && !g) throw new Error(`no GPU expert report in: ${r.stderr}`);
  if (g) {
    run.gpu = {
      slots: num(g[1]),
      mib: num(g[2]),
      acquired: num(g[3]),
      onGpu: num(g[4]),
      inFlight: num(g[5]),
      guessed: num(g[7]),
      guessUsed: num(g[8]),
      copiedMib: num(g[9]),
      lookahead: num(g[11]),
    };
  }
  return run;
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const medianRun = (r) => r.find((x) => x.ms === median(r.map((y) => y.ms)));

/* The drive the model is on, as the operating system names it. */
function drive() {
  if (process.platform === "win32") {
    const letter = model[0];
    const ps = `$d = Get-Partition -DriveLetter ${letter} | Get-Disk; "$($d.FriendlyName), $($d.BusType)"`;
    const r = spawnSync("powershell.exe", ["-NoProfile", "-Command", ps], { encoding: "utf8" });
    return r.status === 0 && r.stdout.trim() ? `${r.stdout.trim()} (${letter}:)` : "unknown";
  }
  const src = spawnSync("findmnt", ["-no", "SOURCE", "--target", model], { encoding: "utf8" });
  if (src.status === 0 && src.stdout.trim()) {
    const m = spawnSync("lsblk", ["-no", "MODEL,TRAN", src.stdout.trim()], { encoding: "utf8" });
    return `${src.stdout.trim()}${m.status === 0 && m.stdout.trim() ? `, ${m.stdout.trim().split("\n")[0]}` : ""}`;
  }
  const df = spawnSync("df", [model], { encoding: "utf8" });
  return df.status === 0 ? df.stdout.trim().split("\n").pop().split(/\s+/)[0] : "unknown";
}

/* The GPU, as nvidia-smi names it, with its memory and the widest PCIe link it and this machine allow. */
function gpuName() {
  const q = spawnSync("nvidia-smi", ["--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader"],
    { encoding: "utf8" });
  if (q.status !== 0) return "nvidia-smi not available";
  const [name, memory, driver, gen, width] = q.stdout.trim().split("\n")[0].split(",").map((s) => s.trim());
  return `${name}, ${memory}, driver ${driver}, PCIe ${gen}.0 x${width} at most`;
}

console.log(`engine:  ${engine}`);
console.log(`command: ${[engine, ...base].join(" ")} [--expert-cache <MiB>]${gpu ? " [--gpu-expert-cache <MiB>]" : ""}`);
console.log(`machine: ${cpus()[0].model.trim()}, ${Math.round(totalmem() / 2 ** 30)} GiB, ${type()} ${release()}`);
if (gpu) console.log(`GPU:     ${gpuName()}`);
console.log(`drive:   ${drive()}`);
console.log(`runs:    ${runs} at each size, the median shown; ${tokens - 1} steps timed after the --prompt text`);
console.log("");

if (!gpu) {
  const rows = [];
  if (mapped) {
    once(null, null); // warm the file cache, thrown away
    const r = Array.from({ length: runs }, () => once(null, null));
    rows.push({ cache: "mapped", ms: median(r.map((x) => x.ms)) });
  }
  for (const cache of caches) {
    const mid = medianRun(Array.from({ length: runs }, () => once(cache, null)));
    const positions = mid.prompt + mid.steps;
    rows.push({
      cache,
      ms: mid.ms,
      mibPerPosition: mid.mib / positions,
      ready: (100 * mid.ready) / mid.acquired,
      inFlight: (100 * mid.inFlight) / mid.acquired,
      readMBs: mid.readMs > 0 ? (mid.mib * 1.048576) / (mid.readMs / 1000) : 0,
      lookahead: mid.lookahead,
    });
  }
  console.log("cache      ms/token  tokens/s  MiB read/position  ready  in flight  read MB/s/thread  lookahead");
  for (const r of rows) {
    const name = r.cache === "mapped" ? "mapped" : `${r.cache} MiB`;
    if (r.cache === "mapped") {
      console.log(`${name.padEnd(9)} ${r.ms.toFixed(1).padStart(9)} ${(1000 / r.ms).toFixed(2).padStart(9)}`);
      continue;
    }
    console.log(
      `${name.padEnd(9)} ${r.ms.toFixed(1).padStart(9)} ${(1000 / r.ms).toFixed(2).padStart(9)} ${r.mibPerPosition.toFixed(1).padStart(18)} ` +
        `${r.ready.toFixed(0).padStart(5)}% ${r.inFlight.toFixed(0).padStart(9)}% ${r.readMBs.toFixed(0).padStart(17)} ${r.lookahead.toFixed(1).padStart(9)}%`,
    );
  }
  console.log("");
  console.log("ready: experts already read when a layer asked for them; in flight: being read for a prefetch then.");
  console.log("MiB read/position covers the prompt's positions too. read MB/s/thread: bytes over time reading, per thread.");
} else {
  const sources = [...(mapped ? [null] : []), ...caches];
  if (mapped) once(null, gpuCaches[0]); // warm the file cache, thrown away
  const rows = [];
  for (const cache of sources) {
    for (const gpuCache of gpuCaches) {
      const mid = medianRun(Array.from({ length: runs }, () => once(cache, gpuCache)));
      const positions = mid.prompt + mid.steps;
      const g = mid.gpu;
      rows.push({
        source: cache ? `${cache} MiB` : "mapped",
        gpuCache: `${g.mib} MiB, ${g.slots}`,
        ms: mid.ms,
        onGpu: (100 * g.onGpu) / g.acquired,
        inFlight: (100 * g.inFlight) / g.acquired,
        copiedPerPosition: g.copiedMib / positions,
        guessUsed: g.guessed ? (100 * g.guessUsed) / g.guessed : 0,
        lookahead: g.lookahead,
      });
    }
  }
  console.log("from     GPU cache, slots   ms/token  tokens/s  on the GPU  in flight  MiB copied/position  guesses used  lookahead");
  for (const r of rows) {
    console.log(
      `${r.source.padEnd(8)} ${r.gpuCache.padEnd(17)} ${r.ms.toFixed(1).padStart(9)} ${(1000 / r.ms).toFixed(2).padStart(9)} ` +
        `${r.onGpu.toFixed(0).padStart(10)}% ${r.inFlight.toFixed(0).padStart(9)}% ${r.copiedPerPosition.toFixed(1).padStart(20)} ` +
        `${r.guessUsed.toFixed(0).padStart(12)}% ${r.lookahead.toFixed(1).padStart(9)}%`,
    );
  }
  console.log("");
  console.log("from: where the GPU copies its experts from, the mapped checkpoint or an --expert-cache that size in memory.");
  console.log("on the GPU: experts a layer wanted that its cache held already; in flight: still being copied for a guess then.");
  console.log("MiB copied/position covers the prompt's positions too. guesses used: copies made for the lookahead's guess");
  console.log("that a layer then wanted.");
}
