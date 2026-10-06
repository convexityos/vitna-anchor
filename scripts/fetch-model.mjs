#!/usr/bin/env node
// Fetch a pinned model's files from the Hugging Face Hub and check each one.
//
//   node scripts/fetch-model.mjs [name] [--dir <dir>] [--only <file>]... [--check]
//
// name defaults to smollm2-135m, whose pin is reference/smollm2-135m/model.json.
// Files land in models/<name>/ unless --dir or ANCHOR_MODEL_DIR says otherwise.
// Every file is downloaded at the pinned revision, never "main", and is kept
// only if its size and SHA-256 match the pin. A file already present with the
// right digest is not downloaded again. --check downloads nothing and exits 1
// if any file is missing or differs. --only narrows either to the pinned files
// named, as CI fetches OLMoE's tokenizer.json without its 13.8 GB of weights.
//
// Weights are downloaded, never committed: models/ is in .gitignore.
//
// Zero external dependencies.

import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function readPin(name = "smollm2-135m") {
  const pinPath = join(ROOT, "reference", name, "model.json");
  return JSON.parse(readFileSync(pinPath, "utf8"));
}

export function modelDir(name = "smollm2-135m") {
  return process.env.ANCHOR_MODEL_DIR || join(ROOT, "models", name);
}

export async function sha256File(path) {
  const hash = createHash("sha256");
  await pipeline(createReadStream(path), hash);
  return hash.digest("hex");
}

/**
 * Check every pinned file in dir. Returns a list of problems, empty when all match.
 */
export async function checkModelDir(pin, dir) {
  const problems = [];
  for (const file of pin.files) {
    const path = join(dir, file.path);
    if (!existsSync(path)) {
      problems.push(`${file.path}: missing`);
      continue;
    }
    const size = statSync(path).size;
    if (size !== file.size) {
      problems.push(`${file.path}: ${size} bytes, pinned ${file.size}`);
      continue;
    }
    const digest = await sha256File(path);
    if (digest !== file.sha256) {
      problems.push(`${file.path}: sha256 ${digest}, pinned ${file.sha256}`);
    }
  }
  return problems;
}

async function download(url, dest, expected) {
  const partial = `${dest}.partial`;
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`GET ${url}: HTTP ${res.status}`);
  }
  const hash = createHash("sha256");
  let size = 0;
  const body = Readable.fromWeb(res.body);
  body.on("data", (chunk) => {
    hash.update(chunk);
    size += chunk.length;
  });
  await pipeline(body, createWriteStream(partial));
  const digest = hash.digest("hex");
  if (size !== expected.size || digest !== expected.sha256) {
    rmSync(partial, { force: true });
    throw new Error(`${expected.path}: got ${size} bytes with sha256 ${digest}; pinned ${expected.size} bytes with ${expected.sha256}`);
  }
  renameSync(partial, dest);
}

/** The pin with only the files named in only, or the whole pin when only is empty. Throws on a name the pin lacks. */
export function narrowPin(pin, only = []) {
  const unknown = only.filter((f) => !pin.files.some((x) => x.path === f));
  if (unknown.length > 0) throw new Error(`not in the pin: ${unknown.join(", ")}`);
  return only.length > 0 ? { ...pin, files: pin.files.filter((f) => only.includes(f.path)) } : pin;
}

export async function fetchModel(name = "smollm2-135m", { dir = modelDir(name), log = console.log, only = [] } = {}) {
  const pin = narrowPin(readPin(name), only);
  mkdirSync(dir, { recursive: true });
  for (const file of pin.files) {
    const dest = join(dir, file.path);
    // A sentence-transformers model keeps its pooling config in a folder of its own.
    mkdirSync(dirname(dest), { recursive: true });
    if (existsSync(dest) && statSync(dest).size === file.size && (await sha256File(dest)) === file.sha256) {
      log(`  ok       ${file.path}`);
      continue;
    }
    const url = `https://huggingface.co/${pin.repo}/resolve/${pin.revision}/${file.path}`;
    log(`  fetching ${file.path} (${file.size} bytes)`);
    await download(url, dest, file);
    log(`  ok       ${file.path}`);
  }
  return { pin, dir };
}

function isEntryPoint() {
  try {
    return resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url);
  } catch {
    return false;
  }
}

if (isEntryPoint()) {
  const args = process.argv.slice(2);
  // An option's value is not the name: --dir and --only take one each.
  const taking = new Set(["--dir", "--only"]);
  const name = args.find((a, i) => !a.startsWith("--") && !taking.has(args[i - 1])) ?? "smollm2-135m";
  const dirIdx = args.indexOf("--dir");
  const dir = dirIdx >= 0 ? resolve(args[dirIdx + 1]) : modelDir(name);
  const only = args.flatMap((a, i) => (a === "--only" && args[i + 1] ? [args[i + 1]] : []));
  try {
    const pin = narrowPin(readPin(name), only);
    console.log(`${pin.repo} at ${pin.revision} -> ${dir}`);
    if (args.includes("--check")) {
      const problems = await checkModelDir(pin, dir);
      for (const p of problems) console.error(`  ${p}`);
      if (problems.length > 0) process.exit(1);
      console.log(pin.files.length === 1 ? "  the file matches the pin" : `  all ${pin.files.length} files match the pin`);
    } else {
      await fetchModel(name, { dir, only });
    }
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
