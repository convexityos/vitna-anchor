// A step the model fails to run is an error the client sees, never a reply
// cut short: a 500 with a server_error before the response has started, or,
// in a stream whose 200 has gone out, an error event with no [DONE] after it.
//
// On the GPU a step fails when the device reports an error. On the CPU it
// fails only for a token outside the vocabulary or a full cache, and with the
// pinned model no request reaches either: prompt ids are checked against the
// vocabulary, the tokenizer's ids and sampled ids come from it, and the room
// left in the context is checked before generating. So each test starts a
// server with VITNA_TEST_FAIL_STEP=<position>, a hook for tests only, which
// makes the step at that position fail once, before it computes anything.
// Everything else is the server as it serves.
//
// Some errors on a GPU leave it able to run nothing more in the process (CUDA
// calls them sticky: an illegal address, a kernel that faulted), and then the
// server answers every request still open with an error and exits with
// status 75, for whatever started it to start it again. With
// VITNA_TEST_LOSE_DEVICE=1 as well, the failed step loses the device in that
// way, so the last tests here check that too, on either device.
//
// Needs a built engine and the model files (node scripts/fetch-model.mjs).
// Without them these tests are skipped, with the reason, unless
// VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail. VITNA_DEVICE=cuda
// serves from the GPU, for an engine built with the CUDA path; the hook fails
// the step there too, before the device is asked to run it.

import assert from "node:assert/strict";
import { test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { request as httpRequest } from "node:http";
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

// An 8-token prompt whose reference continuation, " convert light energy into
// chemical energy. The process is ...", is a token a word.
const science = fixture.prompts.find((x) => x.id === "science");
const STEP_FAILED = {
  message: "The model failed to run a step, so the reply could not be completed.",
  type: "server_error",
  param: null,
  code: null,
};

/**
 * A server whose step at position failAt fails once, and loses the device
 * with it when lose is set; parallel is how many requests it runs at once.
 */
async function start(failAt, { lose = false, parallel = 1 } = {}) {
  assert.ok(!missing, missing ?? "");
  const env = { VITNA_TEST_FAIL_STEP: String(failAt), ...(lose ? { VITNA_TEST_LOSE_DEVICE: "1" } : {}) };
  const args = ["serve", "--model", modelDir, "--model-id", MODEL, "--port", "0", "--ctx", "1024", ...onDevice];
  if (parallel > 1) args.push("--parallel", String(parallel));
  const child = spawn(engine, args, { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, ...env } });
  const server = { child, base: null, stderr: "" };
  child.stderr.on("data", (chunk) => (server.stderr += chunk));
  keepStderr(child, env);
  // The status it exits with, once its stderr has all been read.
  server.closed = new Promise((resolve) => child.on("close", (code) => resolve(code)));
  server.base = await new Promise((resolve, reject) => {
    let out = "";
    // A server given up on is killed: left running, it would keep this file's process alive.
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`the server did not start: ${out}${server.stderr}`));
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
    child.on("close", (code) => reject(new Error(`the server exited with ${code}: ${out}${server.stderr}`)));
  });
  return server;
}

async function withServer(failAt, fn, options) {
  const server = await start(failAt, options);
  try {
    await fn(server);
  } finally {
    server.child.kill();
  }
}

/**
 * What promise settles to, or an error naming what if it takes longer than
 * ms: a server that never answers fails the test rather than holding it up.
 */
async function within(promise, ms, what) {
  let timer;
  const late = new Promise((_, reject) => (timer = setTimeout(() => reject(new Error(`${what}: nothing within ${ms / 1000} s`)), ms)));
  try {
    return await Promise.race([promise, late]);
  } finally {
    clearTimeout(timer);
  }
}

/** The status the server's process exited with, once it has. */
const exitStatus = (server) => within(server.closed, 10_000, "the server's exit");

/**
 * A request whose response is read as it comes: started settles once its
 * status line arrives (for a stream, once the request has its sequence),
 * done once the whole response has.
 */
function send(server, path, body) {
  let begun, refused;
  const started = new Promise((resolve, reject) => ((begun = resolve), (refused = reject)));
  const done = new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const url = new URL(server.base + path);
    const headers = { "content-type": "application/json", "content-length": Buffer.byteLength(payload) };
    const req = httpRequest({ hostname: url.hostname, port: url.port, path: url.pathname, method: "POST", headers }, (res) => {
      begun();
      let text = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => (text += chunk));
      res.on("end", () => resolve({ status: res.statusCode, text }));
    });
    req.on("error", (e) => {
      refused(e);
      reject(e);
    });
    req.end(payload);
  });
  started.catch(() => {}); // a test that waits only for done hears of the error there
  return { started, done };
}

async function post(server, path, body) {
  const res = await fetch(server.base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, headers: res.headers, text, json: res.headers.get("content-type")?.includes("json") ? JSON.parse(text) : null };
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

function tokenize(text) {
  // It takes well under a second; one that has not finished in 2 minutes never will, and is killed.
  const r = spawnSync(engine, ["tokenize", "--model", modelDir], { input: JSON.stringify(text) + "\n", encoding: "utf8", timeout: 2 * 60_000 });
  assert.equal(r.status, 0, r.error?.code === "ETIMEDOUT" ? `tokenize timed out after 2 minutes and was killed: ${r.stderr}` : r.stderr);
  return JSON.parse(r.stdout.trim());
}

/** stderr arrives on its own pipe, so it can trail the response; wait for it. */
async function stderrMatches(server, re) {
  for (let i = 0; i < 500 && !re.test(server.stderr); i++) await new Promise((r) => setTimeout(r, 10));
  assert.match(server.stderr, re);
}

test("a step that fails while the prompt is read is a 500, and the next request runs that step again", A3, () =>
  withServer(5, async (server) => {
    // The prompt is 8 tokens, so the step at position 5 reads one of them.
    const body = { model: MODEL, prompt: science.ids, max_tokens: science.greedy_ids.length, temperature: 0 };
    const failed = await post(server, "/v1/completions", body);
    assert.equal(failed.status, 500, failed.text);
    assert.deepEqual(failed.json, { error: STEP_FAILED }, "no choices and no usage beside the error");
    await stderrMatches(server, /VITNA_TEST_FAIL_STEP is set, for a test: the step at position 5 will fail once\.\r?\nThe step at position 5 failed, as a test asked\./);

    const again = await post(server, "/v1/completions", body);
    assert.equal(again.status, 200, again.text);
    assert.equal(again.json.choices[0].text, science.greedy_text);
    assert.equal(again.json.choices[0].finish_reason, "length");
    assert.equal(again.json.usage.completion_tokens, science.greedy_ids.length);
    assert.equal(again.json.usage.prompt_tokens_details.cached_tokens, 5, "the 5 tokens whose steps ran are reused, and the one whose step failed is run again");
  }));

test("a step that fails mid-reply is a 500, not the reply so far, and its token is not reused", A3, () =>
  withServer(science.ids.length + 3, async (server) => {
    // Position 11 is the step for the fourth new token, " into".
    const failed = await post(server, "/v1/completions", { prompt: science.ids, max_tokens: 8, temperature: 0 });
    assert.equal(failed.status, 500, failed.text);
    assert.deepEqual(failed.json, { error: STEP_FAILED });

    // A prompt that runs past position 11: the same prompt, then the first
    // six tokens of the reference's continuation. The cache kept the prompt
    // and the three new tokens whose steps ran; had it kept " into" too, 12
    // positions would be reused, one of them never computed.
    const read = " convert light energy into chemical energy";
    assert.deepEqual(tokenize(read), science.greedy_ids.slice(0, 6));
    assert.ok(science.greedy_text.startsWith(read));
    const next = await post(server, "/v1/completions", {
      prompt: [...science.ids, ...science.greedy_ids.slice(0, 6)],
      max_tokens: science.greedy_ids.length - 6,
      temperature: 0,
    });
    assert.equal(next.status, 200, next.text);
    assert.equal(next.json.usage.prompt_tokens_details.cached_tokens, science.ids.length + 3);
    assert.equal(next.json.choices[0].text, science.greedy_text.slice(read.length), "the rest of the reference's continuation");
  }));

test("a stream whose step fails mid-reply ends with an error event, with no final chunk, no usage and no [DONE]", A3, () =>
  withServer(science.ids.length + 3, async (server) => {
    const body = { prompt: science.ids, max_tokens: 8, temperature: 0, stream: true, stream_options: { include_usage: true } };
    const failed = await post(server, "/v1/completions", body);
    assert.equal(failed.status, 200, "the 200 went out before the step failed");
    assert.match(failed.headers.get("content-type"), /text\/event-stream/);
    // The error is the last event: a data event with no event name, whose
    // JSON is the body a 500 would have had. openai-python and openai-node
    // raise an APIError on an event like it.
    assert.ok(failed.text.endsWith(`data: ${JSON.stringify({ error: STEP_FAILED })}\n\n`), failed.text);
    const ev = events(failed.text);
    assert.ok(!ev.includes("[DONE]"));
    assert.deepEqual(ev.at(-1), { error: STEP_FAILED });

    // Four tokens were chosen and streamed before the step for the fourth failed.
    const chunks = ev.slice(0, -1);
    const sent = " convert light energy into";
    assert.deepEqual(tokenize(sent), science.greedy_ids.slice(0, 4));
    assert.equal(chunks.map((c) => c.choices[0]?.text).join(""), sent);
    for (const c of chunks) {
      assert.equal(c.choices.length, 1);
      assert.equal(c.choices[0].finish_reason, null);
      assert.equal(c.usage, undefined);
    }

    // The step failed once: the same stream now runs to its end.
    const again = events((await post(server, "/v1/completions", body)).text);
    assert.equal(again.at(-1), "[DONE]");
    const final = again.find((c) => c !== "[DONE]" && c.choices[0]?.finish_reason);
    assert.equal(final.choices[0].finish_reason, "length");
    assert.equal(final.usage.completion_tokens, 8);
    assert.ok(science.greedy_text.startsWith(again.slice(0, -1).map((c) => c.choices[0]?.text ?? "").join("")));
  }));

test("a chat stream whose step fails before the first token sends its role, then the error", A3, () =>
  withServer(2, async (server) => {
    const r = await post(server, "/v1/chat/completions", {
      model: MODEL,
      messages: [{ role: "user", content: "Say hello." }],
      max_tokens: 6,
      temperature: 0,
      stream: true,
    });
    assert.equal(r.status, 200);
    const ev = events(r.text);
    assert.equal(ev.length, 2, r.text);
    assert.deepEqual(ev[0].choices[0].delta, { role: "assistant", content: "" });
    assert.deepEqual(ev[1], { error: STEP_FAILED });
  }));

/** What a request is refused with once the step at position lost the device. */
function deviceLost(position) {
  return {
    message:
      `The GPU can run nothing more in this process after this error: the step at position ${position} failed, and the device with it, as a test asked. ` +
      "The server is stopping; start it again to use the GPU.",
    type: "server_error",
    param: null,
    code: "device_lost",
  };
}

test("a step that loses the device is a 500, then the server exits with status 75 and takes no more connections", A3, () =>
  withServer(
    5,
    async (server) => {
      const failed = await within(post(server, "/v1/completions", { model: MODEL, prompt: science.ids, max_tokens: 4, temperature: 0 }), 60_000, "the request");
      assert.equal(failed.status, 500, failed.text);
      assert.deepEqual(failed.json, { error: STEP_FAILED }, "the request that was running gets the error a failed step gives");
      assert.equal(await exitStatus(server), 75);
      assert.match(
        server.stderr,
        new RegExp(
          [
            "VITNA_TEST_LOSE_DEVICE is set, for a test: the device is lost with that step\\.",
            "The step at position 5 failed, and the device with it, as a test asked\\.",
            "The GPU can run nothing more in this process after this error: the step at position 5 failed, and the device with it, as a test asked\\. " +
              "Each request still open is answered with an error, and the server stops: start it again to use the GPU\\.",
            "POST /v1/completions 500 ",
          ].join("\\r?\\n"),
        ),
      );
      await assert.rejects(fetch(server.base + "/v1/health"), (e) => e.cause?.code === "ECONNREFUSED", "nothing listens on its port any more");
    },
    { lose: true },
  ));

test("once the device is lost, every request running ends with the error, one waiting for a sequence is refused with a 503, and the server exits", A3, () => {
  // A 48-token prompt that reaches the step at position 78, which loses the
  // device, with its 31st new token. The request that started first decodes
  // the 8-token prompt and is far short of position 78 then: it ends because
  // the device is lost, not because the hook failed a step of its own. (On
  // the CPU its step in that round has run; on a GPU it was in the pass that
  // no longer runs.)
  const long = Array.from({ length: 6 }, () => science.ids).flat();
  const position = long.length + 30;
  return withServer(
    position,
    async (server) => {
      const first = send(server, "/v1/completions", { prompt: science.ids, max_tokens: 200, temperature: 0, stream: true });
      await within(first.started, 60_000, "the first request's start");
      const second = send(server, "/v1/completions", { prompt: long, max_tokens: 40, temperature: 0, stream: true });
      await within(second.started, 60_000, "the second request's start");
      // Both sequences are taken, so this one waits for one to come free.
      const waiting = await within(post(server, "/v1/completions", { prompt: science.ids, max_tokens: 4, temperature: 0 }), 60_000, "the waiting request");
      assert.equal(waiting.status, 503, waiting.text);
      assert.deepEqual(waiting.json, { error: deviceLost(position) });
      for (const [name, s] of [["first", first], ["second", second]]) {
        const r = await within(s.done, 10_000, `the ${name} request's end`);
        assert.equal(r.status, 200, `${name}: the 200 went out before the device was lost`);
        const ev = events(r.text);
        assert.ok(ev.length > 1, `${name}: some tokens streamed first: ${r.text}`);
        assert.deepEqual(ev.at(-1), { error: STEP_FAILED }, `${name} ends with the error event`);
        assert.ok(!ev.includes("[DONE]"), name);
      }
      assert.equal(await exitStatus(server), 75);
      assert.equal(server.stderr.match(/The GPU can run nothing more in this process/g)?.length, 1, server.stderr);
      assert.match(server.stderr, /POST \/v1\/completions 503 /);
    },
    { lose: true, parallel: 2 },
  );
});
