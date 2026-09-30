// serve --parallel: requests running at once change no response.
//
// A server started with --parallel n runs up to n requests together, each in
// a sequence of the key-value cache of its own, their next tokens computed
// in one call each round; more wait their turn. Every token's logits are the
// ones a step of that request alone would give, bit for bit, so every
// response must be the one a server running one request at a time gives: the
// text, the finish reason, the usage, every event of a stream, JSON mode's
// objects, a seeded sample and a failed step. The servers compared run
// without prefix reuse, which on a GPU can move logits by float32 rounding
// (see the README); reuse across sequences is checked by its counts.
//
// On the CPU the requests' rows run a step at a time, so this checks how the
// server takes turns; with VITNA_DEVICE=cuda they run in passes that mix the
// requests. Needs a built engine and the model files; skipped with the reason
// otherwise, unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where it fails.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import { request as httpRequest } from "node:http";
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
const PAR = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];
const ids = (id) => fixture.prompts.find((p) => p.id === id).ids;

const servers = {};

function start(args, env = {}) {
  const child = spawn(engine, ["serve", "--model", modelDir, "--port", "0", "--ctx", "1024", ...onDevice, ...args], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, ...env },
  });
  const server = { child, stdout: "", stderr: "" };
  child.stderr.on("data", (chunk) => (server.stderr += chunk));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`the server did not start: ${server.stdout}`)), 30_000);
    child.stdout.on("data", (chunk) => {
      server.stdout += chunk;
      const m = server.stdout.match(/listening on (http:\/\/127\.0\.0\.1:\d+)/);
      if (m && !server.base) {
        clearTimeout(timer);
        server.base = m[1];
        resolve(server);
      }
    });
    child.on("exit", (code) => reject(new Error(`the server exited with ${code}: ${server.stdout}${server.stderr}`)));
  });
}

before(async () => {
  if (PAR.skip) return;
  assert.ok(!missing, missing ?? "");
  servers.alone = await start(["--parallel", "1", "--no-prefix-cache"]);
  servers.together = await start(["--parallel", "3", "--no-prefix-cache"]);
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

// A response without what differs from one run to another: its id and its time.
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

// The requests one at a time to one server, then all at once to the other,
// and the same response to each.
async function sameAtOnce(alone, together, requests) {
  const want = [];
  for (const [path, body] of requests) want.push(await post(alone, path, body));
  const got = await Promise.all(requests.map(([path, body]) => post(together, path, body)));
  got.forEach((g, i) => {
    const label = `request ${i}, ${JSON.stringify(requests[i][1]).slice(0, 80)}`;
    assert.equal(g.status, want[i].status, `${label}: the status`);
    assert.deepEqual(normalize(g.text), normalize(want[i].text), `${label}: the response`);
  });
  return got;
}

test("requests sent at once are answered as one at a time: completions, chat, streams, JSON mode, stops, sampling, and more than run at once", PAR, async (t) => {
  const ask = [{ role: "user", content: "Give a JSON object with a name and an age." }];
  const requests = [
    // Two samples first, so they run together: each must draw from its own seed's stream.
    ["/v1/completions", { prompt: ids("science"), max_tokens: 16, temperature: 0.8, top_p: 0.95, seed: 11 }],
    ["/v1/chat/completions", { messages: ask, max_tokens: 16, temperature: 0.9, seed: 5, stream: true, stream_options: { include_usage: true } }],
    ["/v1/completions", { prompt: ids("capital"), max_tokens: 24, temperature: 0 }],
    ["/v1/completions", { prompt: ids("code"), max_tokens: 20, temperature: 0, stop: ["\n\n"] }],
    ["/v1/chat/completions", { messages: [{ role: "user", content: "Say the words 'the red fox' four times." }], max_tokens: 20, temperature: 0 }],
    ["/v1/chat/completions", { messages: ask, max_tokens: 24, temperature: 0, response_format: { type: "json_object" } }],
    ["/v1/completions", { prompt: ids("numbers"), max_tokens: 16, temperature: 1.1, top_k: 40, seed: 3, stream: true }],
  ];
  // Seven requests, three sequences: four wait for one to come free.
  await sameAtOnce(servers.alone, servers.together, requests);
  // And again, now that each sequence has served a request before.
  await sameAtOnce(servers.alone, servers.together, requests);
  t.diagnostic(`on ${DEVICE || "cpu"}, ${requests.length} requests at once to --parallel 3, twice, each answered as alone`);
});

test("a request is answered while another is still streaming", PAR, async () => {
  const s = servers.together;
  // A long stream starts; once its first token has come, a short request goes in.
  let streamEnded = null, shortDone = null;
  const events = [];
  const long = new Promise((resolve, reject) => {
    const body = JSON.stringify({ prompt: ids("capital"), max_tokens: 200, temperature: 0, stream: true });
    const url = new URL(s.base + "/v1/completions");
    const req = httpRequest({ hostname: url.hostname, port: url.port, path: url.pathname, method: "POST", headers: { "content-type": "application/json", "content-length": Buffer.byteLength(body) } }, (res) => {
      let first = true;
      res.on("data", (d) => {
        events.push(String(d));
        if (first) {
          first = false;
          post(s, "/v1/completions", { prompt: ids("science"), max_tokens: 4, temperature: 0 }).then((r) => {
            shortDone = { at: performance.now(), r };
          }, reject);
        }
      });
      res.on("end", () => {
        streamEnded = performance.now();
        resolve();
      });
    });
    req.on("error", reject);
    req.end(body);
  });
  await long;
  while (!shortDone) await new Promise((r) => setTimeout(r, 20));
  assert.equal(shortDone.r.status, 200, shortDone.r.text);
  assert.equal(JSON.parse(shortDone.r.text).usage.completion_tokens, 4);
  assert.ok(shortDone.at < streamEnded, "the short request waited for the stream to end, as with one request at a time");
  assert.ok(events.join("").includes("[DONE]"), "the stream ran to its end");
});

test("each request keeps a sequence of its own, so two clients taking turns reuse their own prefixes", PAR, async () => {
  const s = (servers.reuse = await start(["--parallel", "2"]));
  const one = await start(["--parallel", "1"]);
  servers.reuseOne = one;
  const a = { prompt: ids("code"), max_tokens: 4, temperature: 0 };
  const b = { prompt: ids("numbers"), max_tokens: 4, temperature: 0 };
  const cached = async (server, body) => {
    const r = await post(server, "/v1/completions", body);
    assert.equal(r.status, 200, r.text);
    return JSON.parse(r.text).usage.prompt_tokens_details.cached_tokens;
  };
  // A, B, then A again: with two sequences A's is still there; with one, B's replaced it.
  for (const server of [s, one]) {
    await cached(server, a);
    await cached(server, b);
  }
  assert.equal(await cached(s, a), ids("code").length - 1, "with --parallel 2, A finds its prefix");
  assert.ok((await cached(one, a)) < ids("code").length - 1, "with --parallel 1, B's request took A's cache");
});

test("a client that hangs up mid-stream ends its own request and no other", PAR, async () => {
  const s = servers.together;
  const other = { prompt: ids("code"), max_tokens: 12, temperature: 0 };
  const want = await post(servers.alone, "/v1/completions", other);
  const hungUp = new Promise((resolve, reject) => {
    const body = JSON.stringify({ prompt: ids("capital"), max_tokens: 300, temperature: 0, stream: true });
    const url = new URL(s.base + "/v1/completions");
    const req = httpRequest({ hostname: url.hostname, port: url.port, path: url.pathname, method: "POST", headers: { "content-type": "application/json", "content-length": Buffer.byteLength(body) } }, (res) => {
      res.once("data", () => {
        req.destroy();
        resolve();
      });
    });
    req.on("error", (err) => (err.code === "ECONNRESET" ? resolve() : reject(err)));
    req.end(body);
  });
  const [, got] = await Promise.all([hungUp, post(s, "/v1/completions", other)]);
  assert.equal(got.status, 200, got.text);
  assert.deepEqual(normalize(got.text), normalize(want.text));
  // The server runs on.
  const after = await post(s, "/v1/completions", { prompt: ids("science"), max_tokens: 2, temperature: 0 });
  assert.equal(after.status, 200, after.text);
});

test("a step that fails fails only its own request: the others at once are answered as alone", PAR, async () => {
  // The step at position 150 fails once. Only the long prompt (147 tokens)
  // reaches it, with its fourth new token, whichever order the requests run in.
  const fail = { VITNA_TEST_FAIL_STEP: "150" };
  servers.aloneFail = await start(["--parallel", "1", "--no-prefix-cache"], fail);
  servers.togetherFail = await start(["--parallel", "3", "--no-prefix-cache"], fail);
  const got = await sameAtOnce(servers.aloneFail, servers.togetherFail, [
    ["/v1/completions", { prompt: ids("capital"), max_tokens: 12, temperature: 0 }],
    ["/v1/completions", { prompt: ids("long"), max_tokens: 12, temperature: 0 }],
    ["/v1/completions", { prompt: ids("science"), max_tokens: 12, temperature: 0, stream: true }],
  ]);
  assert.deepEqual(got.map((g) => g.status), [200, 500, 200]);
  assert.match(got[1].text, /server_error/);
  assert.match(servers.togetherFail.stderr, /The step at position 150 failed, as a test asked\./);
});

test("--parallel runs from 1 to 64 requests at once, and the server says how many", PAR, async () => {
  for (const n of ["0", "65"]) {
    const r = spawnSync(engine, ["serve", "--model", modelDir, "--port", "0", "--parallel", n, ...onDevice], { encoding: "utf8", timeout: 30_000 });
    assert.notEqual(r.status, 0, `--parallel ${n} was accepted`);
    assert.match(r.stderr, /--parallel takes a number of requests from 1 to 64/);
  }
  assert.match(servers.together.stdout, /3 requests at a time/);
  assert.match(servers.alone.stdout, /one request at a time/);
  const health = await fetch(servers.together.base + "/v1/health").then((r) => r.json());
  assert.equal(health.parallel, 3);
});
