// generate --speculate: drafted tokens, checked together, change nothing.
//
// With --speculate k the engine drafts up to k tokens after each one it takes,
// by finding the text's last tokens earlier in it, and checks the drafts in
// one call of vitna_llama_steps_exact, which gives every position the logits
// a step there would, bit for bit. Each token is then chosen from those rows
// in turn, so the reply must be the one decoding a token at a time gives: the
// same ids, and the same bytes from --logits-out, greedy and sampled.
//
// On the CPU the drafts are checked a step at a time, so this checks the
// drafting, the taking and the forgetting of drafts not taken. With
// VITNA_DEVICE=cuda they run in one pass on the GPU, and this checks that
// pass against the steps.
//
// Needs a built engine and the model files; skipped with the reason
// otherwise, unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where it fails.

import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFixture } from "../reference/compare.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const fixture = loadFixture(here("../reference/smollm2-135m/fixture.json"));
const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const modelDir = process.env.ANCHOR_MODEL_DIR || here("../models/smollm2-135m/");
const missing = !engine
  ? "no built engine found"
  : !existsSync(join(modelDir, "model.safetensors"))
    ? `no model in ${modelDir}; run node scripts/fetch-model.mjs`
    : null;
const SPEC = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];

// A run takes a few seconds. One that has not finished in this many minutes
// never will: it is killed, and fails the test instead of hanging the suite.
const RUN_MINUTES = 5;

function generate(ids, args) {
  assert.ok(engine, "no built engine found");
  const dir = mkdtempSync(join(tmpdir(), "vitna-speculate-"));
  const out = join(dir, "logits.f32");
  try {
    const r = spawnSync(engine, ["generate", "--model", modelDir, "--ids", ids.join(","), "--max-new", "24", "--logits-out", out, "--timing", ...onDevice, ...args],
      { encoding: "utf8", maxBuffer: 1 << 26, timeout: RUN_MINUTES * 60_000 });
    const ended = r.error?.code === "ETIMEDOUT" ? `timed out after ${RUN_MINUTES} minutes and was killed` : `exited ${r.status}`;
    assert.equal(r.status, 0, `${engine} ${ended}: ${r.stderr}`);
    const taken = r.stderr.match(/speculation: (\d+) passes, (\d+) tokens drafted, (\d+) of them taken/);
    return { ids: JSON.parse(r.stdout).ids, logits: readFileSync(out), taken: taken ? Number(taken[3]) : null };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const prompts = ["capital", "science", "code"].map((id) => fixture.prompts.find((p) => p.id === id));
const settings = [
  { name: "greedy", args: ["--greedy"] },
  { name: "sampled", args: ["--temperature", "0.8", "--top-p", "0.95", "--seed", "7"] },
];

test("drafted tokens change nothing: the same ids and logits as a token at a time, greedy and sampled", SPEC, (t) => {
  let taken = 0;
  for (const p of prompts) {
    for (const s of settings) {
      const plain = generate(p.ids, s.args);
      const drafted = generate(p.ids, [...s.args, "--speculate", "7"]);
      assert.deepEqual(drafted.ids, plain.ids, `${p.id}, ${s.name}: the ids`);
      assert.equal(drafted.logits.length, plain.logits.length, `${p.id}, ${s.name}: the logits' length`);
      assert.ok(drafted.logits.equals(plain.logits), `${p.id}, ${s.name}: the logits differ`);
      taken += drafted.taken;
    }
  }
  // "The capital of France is" repeats itself under greedy decoding, so its
  // drafts are taken: this was a test of drafts, not of their absence.
  assert.ok(taken > 0, "no drafted token was taken");
  t.diagnostic(`on ${DEVICE || "cpu"}, ${prompts.length * settings.length} replies the same with --speculate 7; ${taken} drafted tokens taken`);
});
