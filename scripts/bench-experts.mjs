#!/usr/bin/env node
// Time decoding a mixture of experts with its experts read from the drive,
// at several sizes of the expert cache, and print the machine, the drive and
// the command with the results.
//
//   node scripts/bench-experts.mjs [--engine <path>] [--model <dir>] [--prompt <text>]
//                                  [--tokens <n>] [--runs <n>] [--caches <MiB,MiB,...>] [--no-mapped]
//
// Each run is one `generate --greedy --timing --expert-cache <MiB>`, which
// times the tokens after the first new one inside the engine. The expert
// cache starts empty in every run: it is the engine's own memory, and the
// reads bypass the operating system's file cache, so nothing carries from
// one run to the next. The mapped run, without --expert-cache, reads the
// experts through the file cache instead; one is run first to warm it, and
// thrown away. The median of --runs runs (default 3) is reported.
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

const model = resolve(option("--model", join(ROOT, "models", "olmoe-1b-7b")));
const prompt = option("--prompt", "The capital of France is");
const tokens = Number(option("--tokens", "64"));
const runs = Number(option("--runs", "3"));
const caches = option("--caches", "512,1024,2048,4096,8192,16384").split(",").map(Number);
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

if (!existsSync(engine)) throw new Error("no built engine found; pass --engine <path>");
if (!Number.isInteger(tokens) || tokens < 2 || !Number.isInteger(runs) || runs < 1) throw new Error("--tokens must be at least 2, and --runs at least 1");
if (caches.some((c) => !Number.isInteger(c) || c < 1)) throw new Error("--caches is a list of sizes in MiB");

const base = ["generate", "--model", model, "--prompt", prompt, "--max-new", String(tokens), "--greedy", "--timing", "--ctx", "512"];

const num = (s) => Number(s);
function once(cache) {
  const args = cache ? [...base, "--expert-cache", String(cache)] : base;
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
  return run;
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

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

const rows = [];
if (mapped) {
  once(null); // warm the file cache, thrown away
  const r = Array.from({ length: runs }, () => once(null));
  rows.push({ cache: "mapped", ms: median(r.map((x) => x.ms)) });
}
for (const cache of caches) {
  const r = Array.from({ length: runs }, () => once(cache));
  const mid = r.find((x) => x.ms === median(r.map((y) => y.ms)));
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

console.log(`engine:  ${engine}`);
console.log(`command: ${[engine, ...base].join(" ")} [--expert-cache <MiB>]`);
console.log(`machine: ${cpus()[0].model.trim()}, ${Math.round(totalmem() / 2 ** 30)} GiB, ${type()} ${release()}`);
console.log(`drive:   ${drive()}`);
console.log(`runs:    ${runs} at each size, the median shown; ${tokens - 1} steps timed after the --prompt text`);
console.log("");
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
