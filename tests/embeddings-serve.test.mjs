// Gate A14's server half: an embedding model served at /v1/embeddings with
// OpenAI's request and response, each embedding the one the embed command
// gives, bit for bit, and within the reference's tolerance.
//
// Needs a built engine and the model files (node scripts/fetch-model.mjs
// bge-small-en-v1.5). Without them these tests are skipped, with the reason,
// unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { EMBED_ATOL } from "../reference/compare.mjs";
import { keepStderr } from "./server-stderr.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const fixture = JSON.parse(readFileSync(here("../reference/bge-small-en-v1.5/fixture.json"), "utf8"));
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
const MODEL = "bge-small-en-v1.5";
const DIM = 384;

let child = null;
let base = null;

before(async () => {
  if (A14.skip) return;
  assert.ok(!missing, missing ?? "");
  child = spawn(engine, ["serve", "--model", modelDir, "--model-id", MODEL, "--port", "0", "--threads", "2"], { stdio: ["ignore", "pipe", "pipe"] });
  const stderr = keepStderr(child);
  base = await new Promise((resolve, reject) => {
    let out = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`the server did not start: ${out}${stderr.text}`));
    }, 30_000);
    child.stdout.on("data", (chunk) => {
      out += chunk;
      const m = out.match(/listening on (http:\/\/127\.0\.0\.1:\d+)/);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    });
    child.on("close", (code) => reject(new Error(`the server exited with ${code}: ${out}${stderr.text}`)));
  });
});

after(() => {
  if (child) child.kill();
});

async function post(path, body) {
  const res = await fetch(base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, headers: res.headers, text, json: res.headers.get("content-type")?.includes("json") ? JSON.parse(text) : null };
}

/** What the embed command gives for texts: one { ids, norm, embedding } each. */
function embedCommand(texts) {
  const r = spawnSync(engine, ["embed", "--model", modelDir, "--threads", "1"], {
    input: texts.map((t) => JSON.stringify(t)).join("\n") + "\n",
    encoding: "utf8",
    maxBuffer: 1 << 26,
    timeout: 5 * 60_000,
  });
  assert.equal(r.status, 0, r.stderr);
  return r.stdout.trim().split("\n").map((l) => JSON.parse(l));
}

test("health and the model list say an embedding model is served, and what it gives", A14, async () => {
  const models = await (await fetch(base + "/v1/models")).json();
  assert.deepEqual(models.data.map((m) => m.id), [MODEL]);
  const health = await (await fetch(base + "/v1/health")).json();
  assert.deepEqual(
    { ...health, threads: undefined },
    { ok: true, engine: "vitna-anchor", model: MODEL, generation: false, embeddings: true, dimensions: DIM, max_tokens: 512, pooling: "cls", normalized: true, threads: undefined },
  );
  assert.equal(health.threads, 2);
});

test("every reference input, sent at once, comes back in order within the tolerance, and as the embed command gives it, bit for bit", A14, async (t) => {
  const texts = fixture.inputs.map((i) => i.text);
  const r = await post("/v1/embeddings", { model: MODEL, input: texts });
  assert.equal(r.status, 200, r.text);
  assert.equal(r.json.object, "list");
  assert.equal(r.json.model, MODEL);
  assert.deepEqual(r.json.data.map((d) => [d.object, d.index]), texts.map((_, i) => ["embedding", i]));
  const tokens = fixture.inputs.reduce((n, i) => n + i.ids.length, 0);
  assert.deepEqual(r.json.usage, { prompt_tokens: tokens, total_tokens: tokens });
  let worst = 0;
  fixture.inputs.forEach((ref, i) => {
    const got = r.json.data[i].embedding;
    assert.equal(got.length, DIM);
    for (let k = 0; k < DIM; k++) worst = Math.max(worst, Math.abs(got[k] - ref.embedding[k]));
  });
  assert.ok(worst <= EMBED_ATOL, `largest difference ${worst}`);
  const cli = embedCommand(texts);
  r.json.data.forEach((d, i) => assert.deepEqual(d.embedding, cli[i].embedding, fixture.inputs[i].id));
  t.diagnostic(`23 inputs, ${tokens} tokens; largest |served - reference| ${worst.toExponential(2)}`);
});

test("one string is one embedding, and base64 is its float32 bytes, little-endian", A14, async () => {
  const text = fixture.inputs[0].text;
  const plain = await post("/v1/embeddings", { input: text });
  assert.equal(plain.status, 200, plain.text);
  assert.equal(plain.json.data.length, 1);
  const b64 = await post("/v1/embeddings", { input: text, encoding_format: "base64" });
  assert.equal(b64.status, 200, b64.text);
  const bytes = Buffer.from(b64.json.data[0].embedding, "base64");
  assert.equal(bytes.length, DIM * 4);
  const floats = Array.from({ length: DIM }, (_, k) => bytes.readFloatLE(k * 4));
  assert.deepEqual(floats, plain.json.data[0].embedding.map(Math.fround));
  // dimensions equal to the model's is allowed, and changes nothing.
  const same = await post("/v1/embeddings", { input: text, dimensions: DIM, encoding_format: "float", user: "anyone" });
  assert.equal(same.status, 200, same.text);
  assert.deepEqual(same.json.data[0].embedding, plain.json.data[0].embedding);
});

test("requests at once each get their own embeddings", A14, async () => {
  const texts = fixture.inputs.slice(0, 8).map((i) => i.text);
  const results = await Promise.all(texts.map((input) => post("/v1/embeddings", { input })));
  const cli = embedCommand(texts);
  results.forEach((r, i) => {
    assert.equal(r.status, 200, r.text);
    assert.deepEqual(r.json.data[0].embedding, cli[i].embedding, fixture.inputs[i].id);
  });
});

test("a field the server does not know is ignored and named in a header", A14, async () => {
  const r = await post("/v1/embeddings", { input: "hello", input_type: "query", truncate: true });
  assert.equal(r.status, 200, r.text);
  assert.equal(r.headers.get("x-vitna-ignored"), "input_type, truncate");
});

test("what this server cannot do is refused with a 400 that names it, never approximated", A14, async () => {
  const cases = [
    [{}, 400, "missing_required_parameter", "input"],
    [{ input: null }, 400, "missing_required_parameter", "input"],
    [{ input: 3 }, 400, "invalid_value", "input"],
    [{ input: [] }, 400, "invalid_value", "input"],
    [{ input: ["a", 2] }, 400, "unsupported_parameter", "input"],
    [{ input: [[101, 7592, 102]] }, 400, "unsupported_parameter", "input"],
    [{ input: ["a", { text: "b" }] }, 400, "invalid_value", "input"],
    [{ input: Array(2049).fill("a") }, 400, "invalid_value", "input"],
    [{ input: "a", dimensions: 256 }, 400, "unsupported_parameter", "dimensions"],
    [{ input: "a", dimensions: 0 }, 400, "invalid_value", "dimensions"],
    [{ input: "a", encoding_format: "int8" }, 400, "invalid_value", "encoding_format"],
    [{ input: "a", model: "text-embedding-3-small" }, 404, "model_not_found", "model"],
  ];
  for (const [body, status, code, param] of cases) {
    const r = await post("/v1/embeddings", body);
    assert.equal(r.status, status, `${JSON.stringify(body).slice(0, 80)}: ${r.text}`);
    assert.equal(r.json.error.code, code, JSON.stringify(body).slice(0, 80));
    assert.equal(r.json.error.param, param, JSON.stringify(body).slice(0, 80));
  }
  const bad = await post("/v1/embeddings", "{not json");
  assert.equal(bad.status, 400);
  assert.equal(bad.json.error.code, "invalid_json");
});

test("a text longer than the model reads is refused, and names which, rather than cut short", A14, async () => {
  const r = await post("/v1/embeddings", { input: ["short", "word ".repeat(600)] });
  assert.equal(r.status, 400, r.text);
  assert.equal(r.json.error.code, "context_length_exceeded");
  assert.match(r.json.error.message, /at most 512 tokens a text.*input\[1\] is 602 tokens/);
  // 300 texts of 500 tokens are more than one request may hold.
  const many = await post("/v1/embeddings", { input: Array(300).fill("word ".repeat(498)) });
  assert.equal(many.status, 400, many.text.slice(0, 200));
  assert.match(many.json.error.message, /150000 tokens together, and one request may hold 131072/);
});

test("generation is refused on an embedding model, and a GET on /v1/embeddings is a 405", A14, async () => {
  for (const [path, body] of [
    ["/v1/chat/completions", { messages: [{ role: "user", content: "hi" }] }],
    ["/v1/completions", { prompt: "hi" }],
  ]) {
    const r = await post(path, body);
    assert.equal(r.status, 404, r.text);
    assert.equal(r.json.error.code, "model_not_supported");
    assert.match(r.json.error.message, /produces embeddings and generates no text/);
  }
  const get = await fetch(base + "/v1/embeddings");
  assert.equal(get.status, 405);
  assert.equal(get.headers.get("allow"), "POST");
});
