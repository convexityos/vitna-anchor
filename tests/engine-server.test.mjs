// The C engine's HTTP server, started without a model, says so.
//
// Needs a built engine: engine/vitna-anchor(.exe), engine/build/Release/
// vitna-anchor.exe, or a path in VITNA_ENGINE. Without one the test is skipped,
// unless VITNA_REQUIRE_ENGINE=1, as in CI, where a missing binary fails it.

import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const EXE = process.platform === "win32" ? ".exe" : "";
const CANDIDATES = [
  process.env.VITNA_ENGINE,
  fileURLToPath(new URL(`../engine/vitna-anchor${EXE}`, import.meta.url)),
  fileURLToPath(new URL(`../engine/build/Release/vitna-anchor${EXE}`, import.meta.url)),
  fileURLToPath(new URL(`../engine/build/vitna-anchor${EXE}`, import.meta.url)),
].filter(Boolean);

const engine = CANDIDATES.find((p) => existsSync(p));
const required = process.env.VITNA_REQUIRE_ENGINE === "1";

function startEngine() {
  return new Promise((resolve, reject) => {
    const child = spawn(engine, ["serve", "--port", "0"], { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`engine did not report a port within 10 s; output so far: ${out}`));
    }, 10_000);
    child.stdout.on("data", (chunk) => {
      out += chunk;
      const match = out.match(/listening on http:\/\/127\.0\.0\.1:(\d+)/);
      if (match) {
        clearTimeout(timer);
        resolve({ child, port: Number(match[1]), out });
      }
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

async function call(port, method, path, body) {
  const res = await fetch(`http://127.0.0.1:${port}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  return { status: res.status, headers: res.headers, body: await res.json() };
}

test("the engine's server, started without a model, says so and answers generation with 501", { skip: !engine && !required && "no built engine found" }, async () => {
  assert.ok(engine, `no built engine found; looked in ${CANDIDATES.join(", ")}`);
  const { child, port, out } = await startEngine();
  try {
    assert.match(out, /started without a model/);

    const health = await call(port, "GET", "/v1/health");
    assert.equal(health.status, 200);
    assert.equal(health.body.model, null);
    assert.equal(health.body.generation, false);
    assert.equal("airgap" in health.body, false);

    const models = await call(port, "GET", "/v1/models");
    assert.deepEqual(models.body, { object: "list", data: [] });

    const completion = await call(port, "POST", "/v1/chat/completions", {
      model: "anything",
      messages: [{ role: "user", content: "hello" }],
    });
    assert.equal(completion.status, 501);
    assert.equal(completion.body.error.code, "no_model");
    assert.equal(completion.body.choices, undefined);
    assert.equal(completion.body.usage, undefined);
    assert.equal(completion.body.sovereign_proof, undefined);
    assert.equal(completion.headers.get("access-control-allow-origin"), null);

    const telemetry = await call(port, "GET", "/v1/telemetry");
    assert.equal(telemetry.status, 404, "the canned telemetry is gone");
  } finally {
    child.kill();
  }
});
