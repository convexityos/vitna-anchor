// Gate A6: prefix reuse over the real key-value cache, and constrained
// decoding that masks real logits (JSON mode).
//
// Two servers run side by side on the pinned model, one reusing prefixes and
// one not (--no-prefix-cache). Reuse must change nothing a client sees but
// usage.prompt_tokens_details.cached_tokens: the same request gets the same
// reply from both, and the reference's own greedy output where the fixture
// has it. JSON mode must return text that parses as a JSON object when the
// object closes, and a valid start of one when max_tokens cuts it short.
//
// Needs a built engine and the model files; skipped with the reason
// otherwise, unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where it fails.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
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
const A6 = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };

const servers = {};

function start(args) {
  const child = spawn(engine, ["serve", "--model", modelDir, "--port", "0", "--ctx", "1024", ...args], { stdio: ["ignore", "pipe", "pipe"] });
  return new Promise((resolve, reject) => {
    let out = "";
    const timer = setTimeout(() => reject(new Error(`the server did not start: ${out}`)), 30_000);
    child.stdout.on("data", (chunk) => {
      out += chunk;
      const m = out.match(/listening on (http:\/\/127\.0\.0\.1:\d+)/);
      if (m) {
        clearTimeout(timer);
        resolve({ child, base: m[1] });
      }
    });
    child.on("exit", (code) => reject(new Error(`the server exited with ${code}: ${out}`)));
  });
}

before(async () => {
  if (A6.skip) return;
  assert.ok(!missing, missing ?? "");
  servers.reuse = await start([]);
  servers.fresh = await start(["--no-prefix-cache"]);
});

after(() => {
  for (const s of Object.values(servers)) s.child.kill();
});

async function post(server, path, body) {
  const res = await fetch(servers[server].base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, text, json: res.headers.get("content-type")?.includes("json") ? JSON.parse(text) : null };
}

function tokenCount(text) {
  const r = spawnSync(engine, ["tokenize", "--model", modelDir], { input: JSON.stringify(text) + "\n", encoding: "utf8" });
  assert.equal(r.status, 0, r.stderr);
  return JSON.parse(r.stdout.trim()).length;
}

function lcp(a, b) {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  return n;
}

// ---------------------------------------------------------------------------
// Prefix reuse

test("a repeated prompt reuses all but its last token, and the reply is the reference's", A6, async (t) => {
  const p = fixture.prompts.find((x) => x.id === "long");
  const body = { prompt: p.ids, max_tokens: p.greedy_ids.length, temperature: 0 };
  const first = await post("reuse", "/v1/completions", body);
  const again = await post("reuse", "/v1/completions", body);
  for (const r of [first, again]) {
    assert.equal(r.json.choices[0].text, p.greedy_text);
    assert.equal(r.json.usage.prompt_tokens, p.ids.length);
    assert.equal(r.json.usage.completion_tokens, p.greedy_ids.length);
  }
  assert.equal(again.json.usage.prompt_tokens_details.cached_tokens, p.ids.length - 1);
  t.diagnostic(`the second request recomputed 1 of ${p.ids.length} prompt tokens and reused ${p.ids.length - 1}`);
});

test("a shared prefix is reused and the rest recomputed, with the same reply as a server that reuses nothing", A6, async () => {
  const long = fixture.prompts.find((x) => x.id === "long").ids;
  const other = fixture.prompts.find((x) => x.id === "science").ids;
  const branch = [...long.slice(0, 100), ...other];
  await post("reuse", "/v1/completions", { prompt: long, max_tokens: 1, temperature: 0 });
  const body = { prompt: branch, max_tokens: 24, temperature: 0 };
  const reused = await post("reuse", "/v1/completions", body);
  const fresh = await post("fresh", "/v1/completions", body);
  assert.equal(reused.json.usage.prompt_tokens_details.cached_tokens, lcp(long, branch));
  assert.equal(fresh.json.usage.prompt_tokens_details.cached_tokens, 0);
  assert.equal(reused.json.choices[0].text, fresh.json.choices[0].text);
  assert.equal(reused.json.usage.completion_tokens, fresh.json.usage.completion_tokens);

  // Sampling too: the same seed gives the same draw with or without reuse.
  const sampled = { prompt: branch, max_tokens: 24, temperature: 0.8, seed: 5 };
  assert.equal((await post("reuse", "/v1/completions", sampled)).json.choices[0].text, (await post("fresh", "/v1/completions", sampled)).json.choices[0].text);
});

test("a conversation reuses its history from one turn to the next", A6, async () => {
  const system = { role: "system", content: "You answer in one short sentence." };
  const user1 = { role: "user", content: "What is the capital of France?" };
  const turn1 = { messages: [system, user1], max_tokens: 12, temperature: 0 };
  const r1 = await post("reuse", "/v1/chat/completions", turn1);
  const reply = r1.json.choices[0].message.content;
  const turn2 = {
    messages: [system, user1, { role: "assistant", content: reply }, { role: "user", content: "And of Italy?" }],
    max_tokens: 12,
    temperature: 0,
  };
  const reused = await post("reuse", "/v1/chat/completions", turn2);
  const fresh = await post("fresh", "/v1/chat/completions", turn2);
  const history = `<|im_start|>system\n${system.content}<|im_end|>\n<|im_start|>user\n${user1.content}<|im_end|>\n<|im_start|>assistant\n`;
  assert.ok(reused.json.usage.prompt_tokens_details.cached_tokens >= tokenCount(history), "at least the first turn's prompt is reused");
  assert.equal(reused.json.choices[0].message.content, fresh.json.choices[0].message.content);
  assert.equal(reused.json.usage.prompt_tokens, fresh.json.usage.prompt_tokens);
});

// ---------------------------------------------------------------------------
// JSON mode

/** A JSON object, or the valid start of one: V8 reports text cut short as the end of input. */
function assertJsonObjectOrPrefix(text, finish, label) {
  if (finish === "stop") {
    const value = JSON.parse(text);
    assert.ok(value !== null && typeof value === "object" && !Array.isArray(value), `${label}: a JSON object`);
  } else {
    assert.equal(finish, "length", label);
    assert.throws(() => JSON.parse(text), /Unexpected end of JSON input|Unterminated string|Expected/, `${label}: the start of a JSON object`);
    assert.ok(text.trimStart().startsWith("{"), `${label}: begins with {`);
  }
}

test("JSON mode returns a JSON object where the same request without it returns prose", A6, async (t) => {
  let closed = 0;
  for (const p of fixture.prompts) {
    const plain = await post("reuse", "/v1/completions", { prompt: p.text, max_tokens: 40, temperature: 0 });
    const json = await post("reuse", "/v1/completions", { prompt: p.text, max_tokens: 160, temperature: 0, response_format: { type: "json_object" } });
    assert.equal(json.status, 200, json.text);
    assert.throws(() => JSON.parse(plain.json.choices[0].text), undefined, `${p.id}: without JSON mode the reply is not JSON`);
    assertJsonObjectOrPrefix(json.json.choices[0].text, json.json.choices[0].finish_reason, p.id);
    if (json.json.choices[0].finish_reason === "stop") closed++;
  }
  // Chat, sampled, with several seeds.
  for (const seed of [1, 2, 3]) {
    const r = await post("reuse", "/v1/chat/completions", {
      messages: [{ role: "user", content: "Give me a JSON object describing a cat." }],
      response_format: { type: "json_object" },
      max_tokens: 160,
      temperature: 0.8,
      seed,
    });
    assertJsonObjectOrPrefix(r.json.choices[0].message.content, r.json.choices[0].finish_reason, `chat seed ${seed}`);
    if (r.json.choices[0].finish_reason === "stop") closed++;
  }
  assert.ok(closed >= 1, "at least one object closed within max_tokens");
  t.diagnostic(`${closed} of ${fixture.prompts.length + 3} JSON-mode replies closed their object within max_tokens; the rest were valid starts of one`);
});

test("JSON mode streams the same object, and a schema is refused rather than ignored", A6, async () => {
  const body = { prompt: "Data:", max_tokens: 60, temperature: 0, response_format: { type: "json_object" } };
  const whole = await post("reuse", "/v1/completions", body);
  const res = await fetch(servers.reuse.base + "/v1/completions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ...body, stream: true }) });
  const events = (await res.text()).split("\n\n").map((e) => e.trim()).filter((e) => e.startsWith("data: ") && !e.includes("[DONE]")).map((e) => JSON.parse(e.slice(6)));
  assert.equal(events.map((e) => e.choices[0].text).join(""), whole.json.choices[0].text);
  assertJsonObjectOrPrefix(whole.json.choices[0].text, whole.json.choices[0].finish_reason, "Data:");

  const schema = await post("reuse", "/v1/chat/completions", {
    messages: [{ role: "user", content: "hi" }],
    response_format: { type: "json_schema", json_schema: { name: "x", schema: { type: "object" } } },
  });
  assert.equal(schema.status, 400);
  assert.equal(schema.json.error.param, "response_format");
});
