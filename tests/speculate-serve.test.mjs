// serve --speculate: drafted tokens, checked together, change no response.
//
// Two servers run side by side on the pinned model, one drafting (--speculate
// 7) and one not, and get the same requests in the same order. Every response
// must be the same but for its id and time: the text, the finish reason, the
// usage (cached tokens included, so the cache each request leaves must be the
// one decoding a token at a time leaves), every event of a stream, and JSON
// mode's objects. A second pair, with a step made to fail
// (VITNA_TEST_FAIL_STEP), must fail the same request the same way.
//
// On the CPU the drafts are checked a step at a time, so this checks how the
// server drafts, takes and forgets; with VITNA_DEVICE=cuda they run in one
// pass. Needs a built engine and the model files; skipped with the reason
// otherwise, unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where it fails.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn } from "node:child_process";
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
const SPEC = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];

const servers = {};

function start(args, env = {}) {
  const child = spawn(engine, ["serve", "--model", modelDir, "--port", "0", "--ctx", "1024", ...onDevice, ...args], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, ...env },
  });
  const server = { child, stderr: "" };
  child.stderr.on("data", (chunk) => (server.stderr += chunk));
  return new Promise((resolve, reject) => {
    let out = "";
    const timer = setTimeout(() => reject(new Error(`the server did not start: ${out}`)), 30_000);
    child.stdout.on("data", (chunk) => {
      out += chunk;
      const m = out.match(/listening on (http:\/\/127\.0\.0\.1:\d+)/);
      if (m) {
        clearTimeout(timer);
        server.base = m[1];
        resolve(server);
      }
    });
    child.on("exit", (code) => reject(new Error(`the server exited with ${code}: ${out}`)));
  });
}

before(async () => {
  if (SPEC.skip) return;
  assert.ok(!missing, missing ?? "");
  servers.plain = await start([]);
  servers.drafted = await start(["--speculate", "7"]);
});

after(() => {
  for (const s of Object.values(servers)) s.child.kill();
});

async function post(server, path, body) {
  const res = await fetch(server.base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, text: await res.text() };
}

// A response without what differs from one server to another: its id and its time.
function normalize(text) {
  const clean = (s) => {
    const o = JSON.parse(s);
    delete o.id;
    delete o.created;
    return JSON.stringify(o);
  };
  if (text.startsWith("data: ")) {
    return text.split("\n\n").filter(Boolean).map((e) => {
      const data = e.replace(/^data: /, "");
      return data === "[DONE]" ? data : clean(data);
    });
  }
  return clean(text);
}

// Each request to both servers, in the same order, and the same response from each.
async function same(pair, path, body, label) {
  const a = await post(servers[pair[0]], path, body);
  const b = await post(servers[pair[1]], path, body);
  assert.equal(b.status, a.status, `${label}: the status`);
  assert.deepEqual(normalize(b.text), normalize(a.text), `${label}: the response`);
  return a;
}

function takenSoFar(server) {
  return [...server.stderr.matchAll(/speculation: \d+ passes, \d+ tokens drafted, (\d+) of them taken/g)].reduce((s, m) => s + Number(m[1]), 0);
}

test("drafted tokens change no response: completions, chat, streams, JSON mode, stops and reuse", SPEC, async (t) => {
  const pair = ["plain", "drafted"];
  const capital = fixture.prompts.find((p) => p.id === "capital").ids;
  const code = fixture.prompts.find((p) => p.id === "code").ids;
  const long = fixture.prompts.find((p) => p.id === "long").ids;
  let n = 0;
  const check = async (path, body, label) => {
    n++;
    return same(pair, path, body, label);
  };

  await check("/v1/completions", { prompt: capital, max_tokens: 64, temperature: 0 }, "greedy completion");
  const again = await check("/v1/completions", { prompt: capital, max_tokens: 64, temperature: 0 }, "the same again, reusing the cache");
  assert.equal(JSON.parse(again.text).usage.prompt_tokens_details.cached_tokens, capital.length - 1);
  await check("/v1/completions", { prompt: code, max_tokens: 64, temperature: 0, stop: ["\n\n"] }, "a stop sequence");
  await check("/v1/completions", { prompt: long, max_tokens: 48, temperature: 0.8, top_p: 0.95, seed: 11 }, "sampled, top-p");
  await check("/v1/completions", { prompt: "The red fox jumps. The red fox jumps. The red", max_tokens: 40, temperature: 1.2, top_k: 40, seed: 3 }, "sampled, top-k");
  await check("/v1/completions", { prompt: capital, max_tokens: 3, temperature: 0 }, "max_tokens cut inside a pass");

  const system = { role: "system", content: "You answer briefly." };
  const user1 = { role: "user", content: "Say the words 'the red fox' four times." };
  const r1 = await check("/v1/chat/completions", { messages: [system, user1], max_tokens: 40, temperature: 0 }, "chat");
  const reply = JSON.parse(r1.text).choices[0].message.content;
  await check("/v1/chat/completions", {
    messages: [system, user1, { role: "assistant", content: reply }, { role: "user", content: "Now say them twice." }],
    max_tokens: 40,
    temperature: 0,
  }, "the next turn, reusing the conversation");
  await check("/v1/chat/completions", { messages: [system, user1], max_tokens: 40, temperature: 0, stream: true, stream_options: { include_usage: true } }, "a chat stream with usage");
  await check("/v1/completions", { prompt: capital, max_tokens: 32, temperature: 0, stream: true }, "a completion stream");

  const ask = [{ role: "user", content: "Give a JSON object with a name and an age." }];
  await check("/v1/chat/completions", { messages: ask, max_tokens: 60, temperature: 0, response_format: { type: "json_object" } }, "JSON mode");
  await check("/v1/chat/completions", { messages: ask, max_tokens: 60, temperature: 0, response_format: { type: "json_object" }, stream: true }, "JSON mode streamed");
  await check("/v1/chat/completions", { messages: ask, max_tokens: 60, temperature: 0.9, seed: 21, response_format: { type: "json_object" } }, "JSON mode sampled");

  const taken = takenSoFar(servers.drafted);
  assert.ok(taken > 0, "the drafting server took no drafted token");
  assert.equal(takenSoFar(servers.plain), 0, "the other server drafted");
  t.diagnostic(`on ${DEVICE || "cpu"}, ${n} requests answered the same by both servers; ${taken} drafted tokens taken`);
});

test("drafted tokens change no failure: a step made to fail fails the same request the same way", SPEC, async () => {
  const capital = fixture.prompts.find((p) => p.id === "capital").ids;
  // The step at position 15 fails: the eleventh token after the prompt's five.
  const fail = { VITNA_TEST_FAIL_STEP: "15" };
  servers.plainFail = await start([], fail);
  servers.draftedFail = await start(["--speculate", "7"], fail);
  const pair = ["plainFail", "draftedFail"];
  const failed = await same(pair, "/v1/completions", { prompt: capital, max_tokens: 40, temperature: 0, stream: true }, "a stream whose step fails");
  assert.match(failed.text, /server_error/);
  // The step has failed once; the next request runs, and reuses what the cache kept, the same on both.
  const next = await same(pair, "/v1/completions", { prompt: capital, max_tokens: 40, temperature: 0 }, "the request after it");
  assert.equal(next.status, 200);
});
