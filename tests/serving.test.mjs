// Gate A3: the engine serves the pinned model over an OpenAI-compatible /v1,
// streamed or not, with usage counted from the tokens it read and produced.
//
// Needs a built engine and the model files (node scripts/fetch-model.mjs).
// Without them these tests are skipped, with the reason, unless
// VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail. VITNA_DEVICE=cuda
// serves from the GPU, for an engine built with the CUDA path.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import { request as httpRequest } from "node:http";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFixture } from "../reference/compare.mjs";
import { keepStderr } from "./server-stderr.mjs";

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
const A3 = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };
const MODEL = "smollm2-135m";
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];

let child = null;
let base = null;

before(async () => {
  if (A3.skip) return;
  assert.ok(!missing, missing ?? "");
  child = spawn(engine, ["serve", "--model", modelDir, "--model-id", MODEL, "--port", "0", "--ctx", "1024", ...onDevice], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  const stderr = keepStderr(child);
  base = await new Promise((resolve, reject) => {
    let out = "";
    // A server given up on is killed: left running, it would keep this file's process alive.
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
    // "close", not "exit": by then its stderr has all been read.
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

/** The three token counts of a usage object, without the cache detail, which depends on earlier requests. */
function counts(u) {
  return { prompt_tokens: u.prompt_tokens, completion_tokens: u.completion_tokens, total_tokens: u.total_tokens };
}

/** The data payloads of a server-sent event stream, [DONE] included. */
function events(text) {
  return text
    .split("\n\n")
    .map((e) => e.trim())
    .filter((e) => e.startsWith("data: "))
    .map((e) => e.slice(6))
    .map((d) => (d === "[DONE]" ? d : JSON.parse(d)));
}

function tokenCount(text) {
  // It takes well under a second; one that has not finished in 2 minutes never will, and is killed.
  const r = spawnSync(engine, ["tokenize", "--model", modelDir], { input: JSON.stringify(text) + "\n", encoding: "utf8", timeout: 2 * 60_000 });
  assert.equal(r.status, 0, r.error?.code === "ETIMEDOUT" ? `tokenize timed out after 2 minutes and was killed: ${r.stderr}` : r.stderr);
  return JSON.parse(r.stdout.trim()).length;
}

test("the model list and health name the served model", A3, async () => {
  const models = await (await fetch(base + "/v1/models")).json();
  assert.equal(models.object, "list");
  assert.deepEqual(models.data.map((m) => m.id), [MODEL]);
  const health = await (await fetch(base + "/v1/health")).json();
  assert.equal(health.model, MODEL);
  assert.equal(health.generation, true);
  assert.equal(health.context, 1024);
});

test("a greedy completion is the reference's greedy output, and usage counts its tokens", A3, async (t) => {
  for (const p of fixture.prompts) {
    const r = await post("/v1/completions", { model: MODEL, prompt: p.text, max_tokens: p.greedy_ids.length, temperature: 0 });
    assert.equal(r.status, 200, r.text);
    assert.equal(r.json.object, "text_completion");
    assert.equal(r.json.choices[0].text, p.greedy_text, p.id);
    assert.equal(r.json.choices[0].finish_reason, "length", p.id);
    assert.deepEqual(counts(r.json.usage), {
      prompt_tokens: p.ids.length,
      completion_tokens: p.greedy_ids.length,
      total_tokens: p.ids.length + p.greedy_ids.length,
    }, p.id);
    const cached = r.json.usage.prompt_tokens_details.cached_tokens;
    assert.ok(Number.isInteger(cached) && cached < p.ids.length, "cached tokens are a part of the prompt, never all of it");
  }
  // A prompt given as token ids reads exactly those tokens.
  const p = fixture.prompts[0];
  const ids = await post("/v1/completions", { prompt: p.ids, max_tokens: 8, temperature: 0 });
  assert.equal(ids.json.usage.prompt_tokens, p.ids.length);
  assert.equal(ids.json.choices[0].text, (await post("/v1/completions", { prompt: p.text, max_tokens: 8, temperature: 0 })).json.choices[0].text);
  t.diagnostic(`${fixture.prompts.length} of ${fixture.prompts.length} prompts: text, finish reason and usage as the reference`);
});

test("streaming sends the same text in deltas, and usage at the end", A3, async () => {
  const p = fixture.prompts.find((x) => x.id === "unicode");
  const body = { model: MODEL, prompt: p.text, max_tokens: p.greedy_ids.length, temperature: 0 };
  const whole = await post("/v1/completions", body);
  const streamed = await post("/v1/completions", { ...body, stream: true, stream_options: { include_usage: true } });
  assert.equal(streamed.status, 200);
  assert.match(streamed.headers.get("content-type"), /text\/event-stream/);
  const ev = events(streamed.text);
  assert.equal(ev.at(-1), "[DONE]");
  const chunks = ev.slice(0, -1);
  const final = chunks.find((c) => c.choices.length && c.choices[0].finish_reason);
  assert.equal(chunks.map((c) => c.choices[0]?.text ?? "").join(""), whole.json.choices[0].text);
  assert.equal(final.choices[0].finish_reason, "length");
  assert.deepEqual(counts(final.usage), counts(whole.json.usage));
  const usageOnly = chunks.at(-1);
  assert.deepEqual(usageOnly.choices, []);
  assert.deepEqual(usageOnly.usage, final.usage);
  // No chunk splits a UTF-8 character: every delta decodes cleanly.
  for (const c of chunks) assert.ok(!(c.choices[0]?.text ?? "").includes("�"));

  const chat = await post("/v1/chat/completions", {
    model: MODEL,
    messages: [{ role: "user", content: "Say hello." }],
    max_tokens: 6,
    temperature: 0,
    stream: true,
  });
  const cev = events(chat.text);
  assert.deepEqual(cev[0].choices[0].delta, { role: "assistant", content: "" });
  assert.equal(cev.at(-1), "[DONE]");
  const cfinal = cev.at(-2);
  assert.ok(["length", "stop"].includes(cfinal.choices[0].finish_reason));
  assert.ok(cfinal.usage.completion_tokens >= 1 && cfinal.usage.completion_tokens <= 6, "usage rides on the final chunk even unasked, for readers that expect it there");
});

test("a chat completion counts the ChatML prompt it read", A3, async () => {
  const messages = [
    { role: "system", content: "You answer briefly." },
    { role: "user", content: "What is the capital of France?" },
  ];
  const r = await post("/v1/chat/completions", { model: MODEL, messages, max_tokens: 12, temperature: 0 });
  assert.equal(r.status, 200, r.text);
  assert.equal(r.json.object, "chat.completion");
  assert.equal(r.json.choices[0].message.role, "assistant");
  const chatml = messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
  assert.equal(r.json.usage.prompt_tokens, tokenCount(chatml));
  assert.ok(r.json.usage.completion_tokens >= 1 && r.json.usage.completion_tokens <= 12);
  assert.equal(r.json.usage.total_tokens, r.json.usage.prompt_tokens + r.json.usage.completion_tokens);
  // A developer message is a system message, and content may come in text parts.
  const dev = await post("/v1/chat/completions", {
    messages: [{ role: "developer", content: [{ type: "text", text: "You answer briefly." }] }, messages[1]],
    max_tokens: 12,
    temperature: 0,
  });
  assert.equal(dev.json.choices[0].message.content, r.json.choices[0].message.content);
  assert.equal(dev.json.usage.prompt_tokens, r.json.usage.prompt_tokens);
});

test("a stop sequence ends the text before it, and the tokens spent reaching it are counted", A3, async () => {
  const p = fixture.prompts.find((x) => x.id === "capital");
  // The reference continues " the capital of the country." (tokens " the", " capital", " of", ...).
  const r = await post("/v1/completions", { prompt: p.text, max_tokens: 20, temperature: 0, stop: [" of", "zzz"] });
  assert.equal(r.json.choices[0].text, " the capital");
  assert.equal(r.json.choices[0].finish_reason, "stop");
  assert.equal(r.json.usage.completion_tokens, 3);
  const s = await post("/v1/completions", { prompt: p.text, max_tokens: 20, temperature: 0, stop: " of", stream: true });
  const ev = events(s.text);
  assert.equal(ev.slice(0, -1).map((c) => c.choices[0].text).join(""), " the capital");
  assert.equal(ev.at(-2).choices[0].finish_reason, "stop");
});

test("a seed reproduces a sampled completion, and the seed used is reported", A3, async () => {
  const body = { prompt: "Once upon a time", max_tokens: 16, temperature: 0.9, top_p: 0.95, seed: 42 };
  const a = await post("/v1/completions", body);
  const b = await post("/v1/completions", body);
  assert.equal(a.json.choices[0].text, b.json.choices[0].text);
  assert.equal(a.headers.get("x-vitna-seed"), "42");
  const unseeded = await post("/v1/completions", { ...body, seed: undefined });
  assert.match(unseeded.headers.get("x-vitna-seed"), /^\d+$/);
});

test("what it cannot do, it refuses by name", A3, async () => {
  const chat = (extra) => post("/v1/chat/completions", { messages: [{ role: "user", content: "hi" }], max_tokens: 2, ...extra });
  const cases = [
    [{ n: 2 }, 400, "n"],
    [{ tools: [{ type: "function", function: { name: "f" } }] }, 400, "tools"],
    [{ response_format: { type: "json_schema", json_schema: { name: "x", schema: { type: "object" } } } }, 400, "response_format"],
    [{ logprobs: true }, 400, "logprobs"],
    [{ presence_penalty: 0.5 }, 400, "presence_penalty"],
    [{ logit_bias: { 5: 10 } }, 400, "logit_bias"],
    [{ max_tokens: 5000 }, 400, "max_tokens"],
    [{ model: "gpt-4o" }, 404, "model"],
    [{ temperature: 3 }, 400, "temperature"],
    [{ stream_options: { include_usage: true } }, 400, "stream_options"],
  ];
  for (const [extra, status, param] of cases) {
    const r = await chat(extra);
    assert.equal(r.status, status, JSON.stringify(extra));
    assert.equal(r.json.error.param, param, JSON.stringify(extra));
  }
  const tool = await post("/v1/chat/completions", { messages: [{ role: "tool", content: "x" }] });
  assert.equal(tool.status, 400);
  const bad = await post("/v1/chat/completions", "{not json");
  assert.equal(bad.status, 400);
  assert.equal(bad.json.error.code, "invalid_json");
  const empty = await post("/v1/chat/completions", { messages: [] });
  assert.equal(empty.status, 400);
  const get = await fetch(base + "/v1/chat/completions");
  assert.equal(get.status, 405);
  const ignored = await chat({ temperature: 0, frobnicate: true });
  assert.equal(ignored.status, 200);
  assert.equal(ignored.headers.get("x-vitna-ignored"), "frobnicate");
});

test("a client that hangs up mid-stream does not take the server down", A3, async () => {
  await new Promise((resolve, reject) => {
    const body = JSON.stringify({ prompt: "The lighthouse keeper", max_tokens: 400, temperature: 0, stream: true });
    const url = new URL(base + "/v1/completions");
    const req = httpRequest({ hostname: url.hostname, port: url.port, path: url.pathname, method: "POST", headers: { "content-type": "application/json", "content-length": Buffer.byteLength(body) } }, (res) => {
      res.once("data", () => {
        req.destroy();
        resolve();
      });
    });
    req.on("error", (err) => (err.code === "ECONNRESET" ? resolve() : reject(err)));
    req.end(body);
  });
  const r = await post("/v1/completions", { prompt: "Hello", max_tokens: 2, temperature: 0 });
  assert.equal(r.status, 200);
  assert.equal(r.json.usage.completion_tokens, 2);
});
