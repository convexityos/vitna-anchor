// Gate A14: an embedding model's reference fixture, the comparison it is held
// to, and the engine against it.
//
// The first tests need nothing but the repository: the fixture is the one its
// pin, inputs and recorder describe, and the comparison catches a wrong
// answer. The rest run the engine against the fixture, and need the model
// files (node scripts/fetch-model.mjs bge-small-en-v1.5) and a built engine.
// Without them they are skipped, with the reason, unless
// VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail.

import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { EMBED_ATOL, compareEmbeddings } from "../reference/compare.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const REF = here("../reference/bge-small-en-v1.5/");
const fixture = JSON.parse(readFileSync(join(REF, "fixture.json"), "utf8"));
const pin = JSON.parse(readFileSync(join(REF, "model.json"), "utf8"));
const inputs = JSON.parse(readFileSync(join(REF, "inputs.json"), "utf8"));
const DIM = 384;

const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");
const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

test("the embedding fixture is the one its pin, its inputs and its recorder describe", () => {
  assert.equal(fixture.format, 1);
  assert.equal(fixture.model.repo, pin.repo);
  assert.equal(fixture.model.revision, pin.revision);
  assert.equal(fixture.model.hidden_size, DIM);
  assert.equal(fixture.model.max_tokens, 512);
  assert.equal(fixture.model.pooling, "cls");
  assert.equal(fixture.model.normalize, true);
  assert.equal(fixture.model.query_instruction, inputs.query_instruction);
  assert.equal(fixture.inputs.length, inputs.inputs.length);
  assert.equal(fixture.inputs_file.sha256_lf, sha256Lf(join(REF, "inputs.json")), "inputs.json changed: record the fixture again");
  assert.equal(fixture.recorder.sha256_lf, sha256Lf(here("../reference/record_embed.py")), "record_embed.py changed: record the fixture again");
  assert.equal(fixture.recorder.imports.sha256_lf, sha256Lf(here("../reference/record.py")), "record.py changed: record the embedding fixture again");
  assert.deepEqual(fixture.software, {
    torch: "2.14.0+cpu",
    transformers: "5.17.0",
    tokenizers: "0.23.2",
    safetensors: "0.8.0",
    numpy: "2.5.3",
    python: fixture.software.python,
  });
  assert.deepEqual(
    { dtype: fixture.settings.dtype, attn: fixture.settings.attn_implementation, threads: fixture.settings.threads },
    { dtype: "float32", attn: "eager", threads: 1 },
  );
  assert.ok(readFileSync(join(REF, "fixture.json")).length < 1024 * 1024, "the fixture stays under 1 MB");

  assert.deepEqual(fixture.inputs.map((i) => [i.id, i.text]), inputs.inputs.map((i) => [i.id, i.text]));
  assert.deepEqual(fixture.corpus.map((c) => c.text), inputs.corpus);
  // transformers and the model's own tokenizer.json agree on every string.
  assert.deepEqual(fixture.tokenizer_disagreements, []);
  for (const c of fixture.corpus) assert.deepEqual(c.ids, c.tokenizer_json_ids, JSON.stringify(c.text));

  for (const i of fixture.inputs) {
    assert.equal(i.ids[0], 101, `${i.id} starts with [CLS]`);
    assert.equal(i.ids.at(-1), 102, `${i.id} ends with [SEP]`);
    assert.ok(i.ids.length <= 512, i.id);
    assert.equal(i.embedding.length, DIM, i.id);
    assert.equal(i.mean_embedding.length, DIM, i.id);
    assert.ok(Math.abs(norm(i.embedding) - 1) < 1e-6, `${i.id}: the embedding is normalized`);
    assert.ok(Math.abs(norm(i.mean_embedding) - 1) < 1e-6, `${i.id}: the mean embedding is normalized`);
    assert.ok(i.cls_norm > 1, i.id);
  }
  // A query carries the instruction the model card gives for queries.
  for (const i of fixture.inputs.filter((x) => x.id.startsWith("query-"))) assert.ok(i.text.startsWith(inputs.query_instruction), i.id);
  // The longest input runs near the model's 512 positions.
  assert.ok(Math.max(...fixture.inputs.map((i) => i.ids.length)) > 500);
});

test("the embedding comparison accepts the reference's own embeddings and rejects a wrong answer", () => {
  const own = fixture.inputs.map((i) => ({ ids: i.ids, norm: i.cls_norm, embedding: i.embedding.slice() }));
  const ownMean = fixture.inputs.map((i) => i.mean_embedding.slice());
  const exact = compareEmbeddings(fixture, own, ownMean);
  assert.deepEqual(exact.failures, []);
  assert.deepEqual(exact.worst, { embedding: 0, mean: 0, norm: 0 });

  // Every value off by 0.4 of the tolerance passes.
  const close = own.map((o) => ({ ...o, embedding: o.embedding.map((v) => v + 0.4 * EMBED_ATOL) }));
  assert.deepEqual(compareEmbeddings(fixture, close, ownMean).failures, []);

  // One value of one embedding off by twice the tolerance does not.
  const one = own.map((o) => ({ ...o, embedding: o.embedding.slice() }));
  one[7].embedding[100] += 2 * EMBED_ATOL;
  assert.ok(compareEmbeddings(fixture, one, ownMean).failures.some((f) => f.includes("embedding value 100")));

  // Nor does a mean embedding off by twice the tolerance.
  const mean = ownMean.map((m) => m.slice());
  mean[3][0] -= 2 * EMBED_ATOL;
  assert.ok(compareEmbeddings(fixture, own, mean).failures.some((f) => f.includes("mean value 0")));

  // Nor a norm off by twice the tolerance, relatively, nor different ids, nor a missing input.
  const normed = own.map((o, k) => ({ ...o, norm: k === 2 ? o.norm * (1 + 2 * EMBED_ATOL) : o.norm }));
  assert.ok(compareEmbeddings(fixture, normed, ownMean).failures.some((f) => f.includes("[CLS] norm")));
  const ids = own.map((o, k) => ({ ...o, ids: k === 0 ? [101, 102] : o.ids }));
  assert.ok(compareEmbeddings(fixture, ids, ownMean).failures.some((f) => f.includes("ids")));
  assert.ok(compareEmbeddings(fixture, own.slice(1), ownMean).failures.length > 0);
});

// ---------------------------------------------------------------------------
// The engine against the fixture: gate A14.

const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const modelDir = process.env.ANCHOR_EMBED_MODEL_DIR || here("../models/bge-small-en-v1.5/");
const missing = !engine
  ? "no built engine found"
  : !existsSync(join(modelDir, "model.safetensors"))
    ? `no model in ${modelDir}; run node scripts/fetch-model.mjs bge-small-en-v1.5`
    : null;
const A14 = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };

function runEngine(args, input) {
  assert.ok(!missing, missing ?? "");
  // A run takes about a second; one that has not finished in 5 minutes never will, and is killed.
  const r = spawnSync(engine, args, { input, encoding: "utf8", maxBuffer: 1 << 26, timeout: 5 * 60_000 });
  const ended = r.error?.code === "ETIMEDOUT" ? "timed out after 5 minutes and was killed" : `exited ${r.status}`;
  assert.equal(r.status, 0, `${engine} ${args.join(" ")} ${ended}: ${r.stderr}`);
  return r.stdout;
}

const lines = (text) => text.trim().split("\n").map((l) => JSON.parse(l));
const asInput = (texts) => texts.map((t) => JSON.stringify(t)).join("\n") + "\n";

test("the engine's WordPiece tokenizer gives the ids of the model's own tokenizer.json", A14, (t) => {
  const cases = [...fixture.corpus.map((c) => [c.text, c.tokenizer_json_ids]), ...fixture.inputs.map((i) => [i.text, i.ids])];
  const got = lines(runEngine(["tokenize", "--model", modelDir], asInput(cases.map(([text]) => text))));
  assert.equal(got.length, cases.length);
  cases.forEach(([text, ids], k) => assert.deepEqual(got[k], ids, JSON.stringify(text)));
  t.diagnostic(`${cases.length} of ${cases.length} strings tokenized exactly`);
});

test("the engine's embeddings match the reference within the stated tolerance", A14, (t) => {
  const texts = fixture.inputs.map((i) => i.text);
  const cls = lines(runEngine(["embed", "--model", modelDir], asInput(texts)));
  const mean = lines(runEngine(["embed", "--model", modelDir, "--pooling", "mean"], asInput(texts))).map((o) => o.embedding);
  const result = compareEmbeddings(fixture, cls, mean);
  assert.deepEqual(result.failures.slice(0, 10), []);
  t.diagnostic(
    `largest |engine - reference|: ${result.worst.embedding.toExponential(2)} [CLS], ${result.worst.mean.toExponential(2)} mean, ` +
      `norm ${result.worst.norm.toExponential(2)} relative; tolerance ${EMBED_ATOL}`,
  );
});

test("the engine's embeddings are the same, bit for bit, on any number of threads and however texts are grouped", A14, () => {
  const texts = fixture.inputs.map((i) => i.text);
  const all = (threads) => runEngine(["embed", "--model", modelDir, "--threads", String(threads)], asInput(texts));
  const one = all(1);
  assert.equal(all(3), one, "three threads gave other bits than one");
  // Each text alone, and the texts in reverse order, run in passes of other shapes.
  const alone = texts.slice(0, 6).map((text) => runEngine(["embed", "--model", modelDir, "--threads", "2", "--text", text]));
  assert.deepEqual(alone.map((s) => s.trim()), one.trim().split("\n").slice(0, 6));
  const reversed = runEngine(["embed", "--model", modelDir, "--threads", "2"], asInput(texts.slice().reverse())).trim().split("\n").reverse();
  assert.deepEqual(reversed, one.trim().split("\n"));
});

test("the engine refuses a text longer than the model reads, rather than cutting it short", A14, () => {
  const long = "word ".repeat(600);
  const r = spawnSync(engine, ["embed", "--model", modelDir, "--text", long], { encoding: "utf8", timeout: 60_000 });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /has 602 tokens, and the model reads at most 512/);
  assert.equal(r.stdout, "");
});
