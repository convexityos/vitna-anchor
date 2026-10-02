// What the engine's run, generate and logits say when a step does not run:
// why, as far as the engine knows, and never --ctx for a step that failed
// for another reason.
//
// A step does not run when its token is outside the model's vocabulary
// (--ids takes any id), when its position is past the key-value cache
// (--ctx), or when the forward pass fails there: on a GPU when the device
// reports an error, and with experts read from a drive when a read fails,
// either of which the engine prints first. After some GPU errors the device
// can run nothing more in the process (CUDA calls them sticky), and the line
// says that too. The first two need only a CPU. The others are made, on
// either device, with VITNA_TEST_FAIL_STEP=<position>, the hook
// tests/step-failure.test.mjs starts its servers with, which these commands
// honour too: the step at that position fails once, before it computes
// anything, and with VITNA_TEST_LOSE_DEVICE=1 as well the device is lost
// with it.
//
// Needs a built engine and the model files (node scripts/fetch-model.mjs).
// Without them these tests are skipped, with the reason, unless
// VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail. VITNA_DEVICE=cuda
// runs the model on the GPU, for an engine built with the CUDA path.

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
const CLI = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];

// The 8-token science prompt, whose reference continuation is " convert
// light energy into chemical energy ...", and its text, for run.
const science = fixture.prompts.find((p) => p.id === "science");
const VOCAB = fixture.model.vocab_size;

// A run takes a second or two. One that has not ended in this many minutes
// never will: it is killed, and fails its test instead of hanging the suite.
const RUN_MINUTES = 2;

/** The engine, with args and the device asked for, and env added to this process's without the hooks. */
function engineRun(args, env = {}) {
  assert.ok(!missing, missing ?? "");
  const base = { ...process.env };
  delete base.VITNA_TEST_FAIL_STEP;
  delete base.VITNA_TEST_LOSE_DEVICE;
  const r = spawnSync(engine, [...args, ...onDevice], { env: { ...base, ...env }, encoding: "utf8", maxBuffer: 1 << 26, timeout: RUN_MINUTES * 60_000 });
  assert.notEqual(r.error?.code, "ETIMEDOUT", `${engine} ${args.join(" ")} timed out after ${RUN_MINUTES} minutes and was killed: ${r.stderr}`);
  return r;
}

const generate = (ids, args = [], env) => engineRun(["generate", "--model", modelDir, "--ids", ids.join(","), "--greedy", ...args], env);
const run = (text, args = [], env) => engineRun(["run", "--model", modelDir, "--prompt", text, ...args], env);

function logits(ids, args = [], env) {
  const dir = mkdtempSync(join(tmpdir(), "vitna-step-failure-cli-"));
  try {
    return engineRun(["logits", "--model", modelDir, "--ids", ids.join(","), "--out", join(dir, "logits.f32"), ...args], env);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** The hook's environment: the step at position at fails once, and with lose the device is lost with it. */
const failAt = (at, lose = false) => ({ VITNA_TEST_FAIL_STEP: String(at), ...(lose ? { VITNA_TEST_LOSE_DEVICE: "1" } : {}) });

// The engine's lines, as these tests expect them.
const outsideVocabulary = (token, at) => `token ${token} at position ${at} is outside the model's vocabulary, ids 0 to ${VOCAB - 1}`;
function cacheFull(ctx) {
  const max = JSON.parse(readFileSync(join(modelDir, "config.json"), "utf8")).max_position_embeddings;
  return `the key-value cache is full at ${ctx} positions; --ctx sets how many it holds, up to the model's maximum of ${max}`;
}
const armed = (at) => `VITNA_TEST_FAIL_STEP is set, for a test: the step at position ${at} will fail once.`;
const asked = (at) => `The step at position ${at} failed, as a test asked.`;
const passFailed = (at) => `the forward pass failed at position ${at}`;

/**
 * A run that stopped where a step did not run: status 1, and stderr ending
 * with lines, in order. No line blames --ctx unless the cache is what filled.
 */
function stoppedWith(r, lines) {
  assert.equal(r.status, 1, `exited ${r.status}: ${r.stderr}`);
  const got = r.stderr.split(/\r?\n/).filter((line) => line !== "");
  assert.deepEqual(got.slice(-lines.length), lines, r.stderr);
  if (!lines.some((line) => line.includes("--ctx"))) assert.doesNotMatch(r.stderr, /--ctx/, "nothing here filled the key-value cache");
}

test("a token outside the vocabulary is named, with the ids the model has, by generate and logits", CLI, () => {
  const ids = science.ids.with(5, VOCAB);
  const generated = generate(ids, ["--max-new", "4"]);
  stoppedWith(generated, [outsideVocabulary(VOCAB, 5)]);
  assert.equal(generated.stdout, "", "no reply is printed");
  stoppedWith(logits(ids), [outsideVocabulary(VOCAB, 5)]);
});

test("a prompt longer than the key-value cache says the cache is full, and how far --ctx goes, in generate, logits and run", CLI, () => {
  stoppedWith(generate(science.ids, ["--max-new", "4", "--ctx", "4"]), [cacheFull(4)]);
  stoppedWith(logits(science.ids, ["--ctx", "4"]), [cacheFull(4)]);
  stoppedWith(run(science.text, ["--max-new", "4", "--ctx", "4"]), [cacheFull(4)]);
});

test("a reply that runs past the key-value cache says the cache is full", CLI, () => {
  // The prompt and the first 4 new tokens fill the 12 positions; the step for the fifth would be the 13th.
  const r = generate(science.ids, ["--max-new", "10", "--ctx", "12"]);
  stoppedWith(r, [cacheFull(12)]);
  assert.equal(r.stdout, "", "no reply is printed");
});

test("a step that fails is named as the forward pass failing at its position, after the line that says why, in the prompt and in the reply", CLI, () => {
  // Position 5 is in the 8-token prompt; position 10 is the step for the third new token.
  for (const at of [5, 10]) {
    const r = generate(science.ids, ["--max-new", "6"], failAt(at));
    stoppedWith(r, [armed(at), asked(at), passFailed(at)]);
    assert.equal(r.stdout, "", "no reply is printed");
  }
  stoppedWith(logits(science.ids, [], failAt(5)), [armed(5), asked(5), passFailed(5)]);
  for (const at of [5, 10]) {
    const r = run(science.text, ["--max-new", "6"], failAt(at));
    stoppedWith(r, [armed(at), asked(at), passFailed(at)]);
    assert.ok(r.stdout.startsWith(science.text), r.stdout);
  }
});

test("with --speculate, a step that fails in a pass over drafted tokens is named the same way", CLI, () => {
  // The capital prompt and its reference continuation up to the second
  // "The". The next token, " capital", makes "The capital" again, as the
  // prompt began, so a draft follows it straight away; with 3 new tokens the
  // second and third then come from one pass over that token and its draft.
  const capital = fixture.prompts.find((p) => p.id === "capital");
  const ids = [...capital.ids, ...capital.greedy_ids.slice(0, 9)];
  const args = ["--max-new", "3", "--speculate", "7"];
  const plain = generate(ids, [...args, "--timing"]);
  assert.equal(plain.status, 0, plain.stderr);
  const counts = plain.stderr.match(/speculation: (\d+) passes, (\d+) tokens drafted, (\d+) of them taken/);
  assert.ok(counts, plain.stderr);
  // Each pass gives a token from its first row, and one more for each draft
  // taken. If passes give both tokens after the first, no step ran on its
  // own: both positions ran first in a pass, and the hook fails a row of one.
  const [passes, taken] = [Number(counts[1]), Number(counts[3])];
  assert.equal(passes + taken, 2, `a token after the first came from a step on its own: ${counts[0]}`);
  for (const at of [ids.length, ids.length + 1]) {
    const r = generate(ids, args, failAt(at));
    stoppedWith(r, [armed(at), asked(at), passFailed(at)]);
  }
});

test("a step that loses the device says the GPU can run nothing more in this process, and after which error", CLI, () => {
  const lost = (at) => `The step at position ${at} failed, and the device with it, as a test asked.`;
  const cause = (at) => `${passFailed(at)}, and the GPU can run nothing more in this process after this error: the step at position ${at} failed, and the device with it, as a test asked`;
  for (const at of [5, 10]) stoppedWith(generate(science.ids, ["--max-new", "6"], failAt(at, true)), [lost(at), cause(at)]);
  stoppedWith(logits(science.ids, [], failAt(5, true)), [lost(5), cause(5)]);
});

test("VITNA_TEST_FAIL_STEP fails only the step it names, and a value that is not a position is refused", CLI, () => {
  // A position the run does not reach changes nothing: the reference's reply.
  const reached = generate(science.ids, ["--max-new", "4"], failAt(100));
  assert.equal(reached.status, 0, reached.stderr);
  assert.ok(reached.stderr.includes(armed(100)), reached.stderr);
  assert.deepEqual(JSON.parse(reached.stdout).ids, science.greedy_ids.slice(0, 4));

  const words = generate(science.ids, ["--max-new", "4"], { VITNA_TEST_FAIL_STEP: "five" });
  stoppedWith(words, ["VITNA_TEST_FAIL_STEP must be a position, a whole number, not five"]);
  const alone = generate(science.ids, ["--max-new", "4"], { VITNA_TEST_LOSE_DEVICE: "1" });
  stoppedWith(alone, ["VITNA_TEST_LOSE_DEVICE needs VITNA_TEST_FAIL_STEP, the position of the step that loses the device"]);
  for (const r of [words, alone]) assert.equal(r.stdout, "", "nothing runs");
});
