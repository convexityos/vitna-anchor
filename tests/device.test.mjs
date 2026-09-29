// The engine runs the model on the device it is asked for, or says why it
// cannot and stops. It never runs the model on the CPU in place of the GPU.
//
// Needs a built engine: engine/vitna-anchor(.exe), engine/build/Release/, or a
// path in VITNA_ENGINE. Without one these tests are skipped, unless
// VITNA_REQUIRE_ENGINE=1, where a missing binary fails them. No model is
// needed: a device the engine cannot use is refused before anything is
// loaded, so --model names a directory that does not exist.
//
// VITNA_EXPECT_NO_GPU=1 says this machine has no usable CUDA device, as on
// CI's runners, so an engine built with the CUDA path must refuse
// --device cuda too.

import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const EXE = process.platform === "win32" ? ".exe" : "";
const CANDIDATES = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean);
const engine = CANDIDATES.find((p) => existsSync(p));
const required = process.env.VITNA_REQUIRE_ENGINE === "1";
const skip = !engine && !required && "no built engine found";

function run(args) {
  assert.ok(engine, `no built engine found; looked in ${CANDIDATES.join(", ")}`);
  return spawnSync(engine, args, { encoding: "utf8" });
}

// Whether this engine has the CUDA path, as its usage says.
const cudaBuilt = engine ? !/This build has no CUDA path/.test(run(["--help"]).stdout) : false;

// Ask for logits on a device, with no model there to load. Returns the
// process's result, and whether it wrote the logits file.
function logitsOn(device) {
  const dir = mkdtempSync(join(tmpdir(), "vitna-device-"));
  const out = join(dir, "logits.f32");
  try {
    const r = run(["logits", "--model", join(dir, "no-model"), "--ids", "1,2,3", "--out", out, "--device", device]);
    return { ...r, wrote: existsSync(out) };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("a device the engine does not know is refused by name", { skip }, () => {
  const r = logitsOn("tpu");
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /--device must be cpu or cuda, not tpu/);
  assert.equal(r.wrote, false);
});

test("--device cpu is accepted, and the engine goes on to load the model", { skip }, () => {
  const r = logitsOn("cpu");
  assert.notEqual(r.status, 0, "there is no model to load");
  assert.match(r.stderr, /cannot read .*config\.json/);
});

test("an engine built without CUDA refuses --device cuda, says so, and writes nothing", { skip: skip || (cudaBuilt && "this engine has the CUDA path") }, () => {
  const r = logitsOn("cuda");
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /--device cuda: this engine was built without CUDA/);
  assert.doesNotMatch(r.stderr, /cannot read/, "refused before loading anything");
  assert.equal(r.stdout, "");
  assert.equal(r.wrote, false);
});

const noGpu = process.env.VITNA_EXPECT_NO_GPU === "1";
test("an engine built with CUDA, on a machine with no usable GPU, refuses --device cuda before loading anything", { skip: skip || (!cudaBuilt && "this engine has no CUDA path") || (!noGpu && "VITNA_EXPECT_NO_GPU is not set") }, () => {
  const r = logitsOn("cuda");
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /--device cuda: (no usable CUDA device|the CUDA runtime found no device)/);
  assert.doesNotMatch(r.stderr, /cannot read/, "refused before loading anything");
  assert.equal(r.stdout, "");
  assert.equal(r.wrote, false);
});
