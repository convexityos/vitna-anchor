// Gate A1: the reference fixture, and the comparison gate A2 must pass.
//
// The first tests need nothing but the repository: the fixture is the one its
// pin and inputs describe, and the comparison catches a wrong answer. The last
// three are gate A2: they run the engine against the fixture, and need the
// model files (node scripts/fetch-model.mjs) and a built engine.
//
// VITNA_DEVICE=cuda runs the model on the GPU (--device cuda), for an engine
// built with the CUDA path: that is gate A4, the same comparison with the same
// fixture and the same tolerance. Unset, or cpu, the engine uses the CPU.

import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { LOGIT_ATOL, compareGreedy, comparePrefill, compareTokenization, loadFixture } from "../reference/compare.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const REF = here("../reference/smollm2-135m/");
const fixture = loadFixture(join(REF, "fixture.json"));
const pin = JSON.parse(readFileSync(join(REF, "model.json"), "utf8"));
const VOCAB = fixture.model.vocab_size;

const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");

test("the fixture is the one its pin, its inputs and its recorder describe", () => {
  assert.equal(fixture.format, 1);
  assert.equal(fixture.model.repo, pin.repo);
  assert.equal(fixture.model.revision, pin.revision);
  assert.equal(fixture.inputs.sha256_lf, sha256Lf(join(REF, "prompts.json")), "prompts.json changed: record the fixture again");
  assert.equal(fixture.recorder.sha256_lf, sha256Lf(here("../reference/record.py")), "record.py changed: record the fixture again");
  assert.deepEqual(fixture.software, {
    torch: "2.14.0+cpu",
    transformers: "5.17.0",
    tokenizers: "0.23.2",
    safetensors: "0.8.0",
    numpy: "2.5.3",
    python: fixture.software.python,
  });
  assert.equal(fixture.settings.dtype, "float32");
  assert.ok(readFileSync(join(REF, "fixture.json")).length < 3 * 1024 * 1024, "the fixture stays under 3 MB");

  const inputs = JSON.parse(readFileSync(join(REF, "prompts.json"), "utf8"));
  assert.deepEqual(fixture.prompts.map((p) => p.text), inputs.prompts.map((p) => p.text));
  assert.deepEqual(fixture.corpus.map((c) => c.text), inputs.corpus);

  // transformers 5.17 and the model's own tokenizer.json disagree only where
  // the fixture says they do: one string of non-ASCII numerals, because
  // transformers drops the file's Digits step.
  const disagree = fixture.corpus.filter((c) => JSON.stringify(c.ids) !== JSON.stringify(c.tokenizer_json_ids)).map((c) => c.text);
  assert.deepEqual(disagree, fixture.tokenizer_disagreements);
  assert.deepEqual(disagree, ["٣٤٥ ⅫⅣ ½ ²"]);

  for (const p of fixture.prompts) {
    assert.equal(p.positions.length, p.ids.length, p.id);
    assert.equal(p.lastLogits.length, VOCAB, p.id);
    assert.equal(p.greedy.length, inputs.greedy_steps, p.id);
    assert.deepEqual(p.greedy.map((g) => g.id), p.greedy_ids, p.id);
    // The last row's own top entries are the ones recorded for that position,
    // and greedy step 0 chose the last position's argmax.
    const last = p.positions.at(-1);
    for (const [id, v] of last.top) assert.equal(p.lastLogits[id], v, `${p.id} id ${id}`);
    assert.equal(p.greedy[0].id, last.top[0][0], p.id);
  }
});

// The comparison, run on the one full row the fixture keeps for each prompt:
// its last position. The fixture's own values must pass, and a wrong answer
// must not.
function lastPositionOnly(p) {
  return { ...p, ids: [p.ids.at(-1)], positions: [p.positions.at(-1)] };
}

test("the comparison accepts the reference's own logits and rejects a wrong answer", () => {
  for (const p of fixture.prompts) {
    const one = lastPositionOnly(p);
    const exact = comparePrefill(fixture, one, p.lastLogits.slice());
    assert.deepEqual(exact.failures, [], p.id);
    assert.ok(exact.worst.logit === 0 && exact.worst.lse < 1e-9, p.id);

    // Within tolerance: every logit off by 0.4 of it.
    const close = p.lastLogits.map((v) => v + 0.4 * LOGIT_ATOL);
    assert.deepEqual(comparePrefill(fixture, one, close).failures, [], p.id);

    // One probe logit off by twice the tolerance.
    const probe = p.lastLogits.slice();
    probe[fixture.probe_ids[7]] += 2 * LOGIT_ATOL;
    assert.ok(comparePrefill(fixture, one, probe).failures.length > 0, `${p.id}: a wrong probe logit passed`);

    // A logit the fixture keeps only in the full row, off by twice the tolerance.
    const kept = new Set([...one.positions[0].top.map(([id]) => id), ...fixture.probe_ids]);
    const other = [...Array(VOCAB).keys()].find((i) => !kept.has(i) && i > 100);
    const tail = p.lastLogits.slice();
    tail[other] -= 2 * LOGIT_ATOL;
    assert.ok(comparePrefill(fixture, one, tail).failures.length > 0, `${p.id}: a wrong tail logit passed`);

    // The whole row shifted, which moves logsumexp too.
    const shifted = p.lastLogits.map((v) => v + 2 * LOGIT_ATOL);
    assert.ok(comparePrefill(fixture, one, shifted).failures.some((f) => f.includes("logsumexp")), p.id);
  }

  // Greedy: the reference's own choices pass, one changed token does not.
  for (const p of fixture.prompts) {
    const steps = new Float32Array(p.greedy.length * VOCAB).fill(-1e4);
    p.greedy.forEach((g, s) => g.top.forEach(([id, v]) => (steps[s * VOCAB + id] = v)));
    assert.deepEqual(compareGreedy(fixture, p, p.greedy_ids, steps).failures, [], p.id);
    const changed = p.greedy_ids.slice();
    changed[5] = changed[5] === 0 ? 1 : 0;
    assert.ok(compareGreedy(fixture, p, changed, steps).failures.length > 0, p.id);
  }

  // Tokenization: exact.
  const fileIds = (text) => {
    const c = [...fixture.corpus, ...fixture.prompts].find((x) => x.text === text);
    return c.tokenizer_json_ids ?? c.ids;
  };
  assert.deepEqual(compareTokenization(fixture, fileIds), []);
  // transformers' own ids fail where it departs from tokenizer.json.
  const hfIds = (text) => [...fixture.corpus, ...fixture.prompts].find((x) => x.text === text).ids;
  assert.deepEqual(compareTokenization(fixture, hfIds).map((m) => m.text), fixture.tokenizer_disagreements);
  assert.equal(compareTokenization(fixture, () => [0]).length > 0, true);
});

// ---------------------------------------------------------------------------
// The engine against the fixture: gate A2.

const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const modelDir = process.env.ANCHOR_MODEL_DIR || here("../models/smollm2-135m/");
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];

// Gate A2. Without a built engine or the model files these are skipped, with
// the reason, unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail.
const missing = !engine
  ? "no built engine found"
  : !existsSync(join(modelDir, "model.safetensors"))
    ? `no model in ${modelDir}; run node scripts/fetch-model.mjs`
    : null;
const A2 = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };

function runEngine(args, input) {
  assert.ok(engine, "no built engine found");
  assert.ok(existsSync(join(modelDir, "model.safetensors")), `no model in ${modelDir}; run node scripts/fetch-model.mjs`);
  const r = spawnSync(engine, args, { input, encoding: "utf8", maxBuffer: 1 << 26 });
  assert.equal(r.status, 0, `${engine} ${args.join(" ")} exited ${r.status}: ${r.stderr}`);
  return r.stdout;
}

function readF32(path) {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

test("the engine's tokenizer gives the ids of the model's own tokenizer.json", A2, (t) => {
  const cases = [...fixture.corpus, ...fixture.prompts];
  const lines = runEngine(["tokenize", "--model", modelDir], cases.map((c) => JSON.stringify(c.text)).join("\n") + "\n")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
  assert.equal(lines.length, cases.length);
  const byText = new Map(cases.map((c, i) => [c.text, lines[i]]));
  assert.deepEqual(compareTokenization(fixture, (text) => byText.get(text)), []);
  t.diagnostic(`${cases.length} of ${cases.length} strings tokenized exactly`);
});

test("the engine's logits match the reference within the stated tolerance", A2, (t) => {
  const dir = mkdtempSync(join(tmpdir(), "vitna-logits-"));
  let worst = 0;
  let worstLse = 0;
  try {
    for (const p of fixture.prompts) {
      const out = join(dir, `${p.id}.f32`);
      runEngine(["logits", "--model", modelDir, "--ids", p.ids.join(","), "--out", out, ...onDevice]);
      const result = comparePrefill(fixture, p, readF32(out));
      assert.deepEqual(result.failures.slice(0, 10), [], p.id);
      worst = Math.max(worst, result.worst.logit);
      worstLse = Math.max(worstLse, result.worst.lse);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
  t.diagnostic(`on ${DEVICE || "cpu"}, largest |engine - reference|: logits ${worst.toExponential(2)}, logsumexp ${worstLse.toExponential(2)}; tolerance ${LOGIT_ATOL}`);
});

test("the engine's greedy decoding matches the reference token for token", A2, (t) => {
  const dir = mkdtempSync(join(tmpdir(), "vitna-greedy-"));
  let worst = 0;
  let tokens = 0;
  try {
    for (const p of fixture.prompts) {
      const out = join(dir, `${p.id}.f32`);
      const stdout = runEngine([
        "generate", "--model", modelDir, "--ids", p.ids.join(","),
        "--max-new", String(p.greedy_ids.length), "--greedy", "--logits-out", out, ...onDevice,
      ]);
      const { ids } = JSON.parse(stdout);
      const result = compareGreedy(fixture, p, ids, readF32(out));
      assert.deepEqual(result.failures.slice(0, 10), [], p.id);
      worst = Math.max(worst, result.worst.logit);
      tokens += ids.length;
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
  t.diagnostic(`on ${DEVICE || "cpu"}, ${tokens} of ${tokens} greedy tokens equal; largest step logit difference ${worst.toExponential(2)}`);
});

// The logits command gives the model a prompt 256 tokens at a time, and every
// fixture prompt is shorter than that, so no fixture prompt runs on a GPU
// whose cache already holds earlier positions of its own. So the engine runs
// one longer prompt, the fixture's prompts and corpus end to end (565
// tokens, in three pieces), on both devices,
// and every position's logits must agree within the tolerance, choosing the
// same token wherever the CPU's top two are further apart than that.
test("on the GPU, a prompt run in pieces gives the CPU's logits at every position", {
  skip: A2.skip || (DEVICE !== "cuda" && "VITNA_DEVICE is not cuda"),
}, (t) => {
  const ids = [...fixture.prompts.flatMap((p) => p.ids), ...fixture.corpus.flatMap((c) => c.tokenizer_json_ids ?? c.ids)];
  assert.ok(ids.length > 512, `${ids.length} tokens do not cross two chunks`);
  const dir = mkdtempSync(join(tmpdir(), "vitna-chunks-"));
  let worst = 0;
  try {
    const rows = {};
    for (const device of ["cpu", "cuda"]) {
      const out = join(dir, `${device}.f32`);
      runEngine(["logits", "--model", modelDir, "--ids", ids.join(","), "--out", out, "--device", device]);
      rows[device] = readF32(out);
      assert.equal(rows[device].length, ids.length * VOCAB, device);
    }
    for (let p = 0; p < ids.length; p++) {
      const o = p * VOCAB;
      let d = 0, cpuTop = 0, cudaTop = 0;
      for (let i = 0; i < VOCAB; i++) {
        d = Math.max(d, Math.abs(rows.cpu[o + i] - rows.cuda[o + i]));
        if (rows.cpu[o + i] > rows.cpu[o + cpuTop]) cpuTop = i;
        if (rows.cuda[o + i] > rows.cuda[o + cudaTop]) cudaTop = i;
      }
      assert.ok(d <= LOGIT_ATOL, `position ${p}: logits differ by ${d}`);
      if (cpuTop !== cudaTop) {
        const gap = rows.cpu[o + cpuTop] - rows.cpu[o + cudaTop];
        assert.ok(gap <= LOGIT_ATOL, `position ${p}: the GPU chose ${cudaTop}, the CPU ${cpuTop}, ${gap} apart`);
      }
      worst = Math.max(worst, d);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
  t.diagnostic(`${ids.length} positions; largest |cuda - cpu| ${worst.toExponential(2)}; tolerance ${LOGIT_ATOL}`);
});
