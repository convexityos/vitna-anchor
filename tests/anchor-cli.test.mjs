import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { request as httpRequest } from "node:http";
import { fileURLToPath } from "node:url";

import { startAnchorServer, NO_MODEL_MESSAGE, REMOVED_COMMANDS } from "../runtime/anchor-run.mjs";

const BIN = fileURLToPath(new URL("../bin/vitna-anchor.mjs", import.meta.url));

function runBin(...args) {
  return spawnSync(process.execPath, [BIN, ...args], { encoding: "utf8" });
}

function httpCall(port, method, path, data) {
  return new Promise((resolve, reject) => {
    const body = data === undefined ? "" : JSON.stringify(data);
    const req = httpRequest(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method,
        headers: body ? { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) } : {},
      },
      (res) => {
        let raw = "";
        res.on("data", (chunk) => (raw += chunk));
        res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: JSON.parse(raw) }));
      }
    );
    req.on("error", reject);
    if (body) req.write(body);
    req.end();
  });
}

function listen(server) {
  return new Promise((resolve) => server.on("listening", () => resolve(server.address().port)));
}

test("serve says the engine serves the model, not it, and answers generation with 501", async () => {
  const server = startAnchorServer({ port: 0, quiet: true });
  const port = await listen(server);
  try {
    const health = await httpCall(port, "GET", "/v1/health");
    assert.equal(health.status, 200);
    assert.equal(health.body.model, null);
    assert.equal(health.body.generation, false);
    assert.equal(health.body.message, NO_MODEL_MESSAGE);
    assert.equal("airgap" in health.body, false);

    const models = await httpCall(port, "GET", "/v1/models");
    assert.equal(models.status, 200);
    assert.deepEqual(models.body, { object: "list", data: [] });

    for (const path of ["/v1/chat/completions", "/v1/completions"]) {
      const res = await httpCall(port, "POST", path, {
        model: "anything",
        messages: [{ role: "user", content: "hello" }],
        stream: true,
      });
      assert.equal(res.status, 501, path);
      assert.equal(res.body.error.code, "no_model");
      assert.equal(res.body.error.message, NO_MODEL_MESSAGE);
      assert.equal(res.body.choices, undefined, "no completion is invented");
      assert.equal(res.body.usage, undefined, "no token count is invented");
      assert.equal(res.headers["access-control-allow-origin"], undefined, "no page on another origin may call it");
    }

    const missing = await httpCall(port, "GET", "/v1/telemetry");
    assert.equal(missing.status, 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("chat says the engine serves the model, and exits non-zero", () => {
  const result = runBin("chat");
  assert.equal(result.status, 1);
  assert.match(result.stderr, /does not serve a model\. The C engine does/);
  assert.equal(result.stdout, "");
});

test("a removed command says why it went, and exits non-zero", () => {
  for (const command of ["probe", "route", "draft", "bench", "tune", "registry"]) {
    const result = runBin(command);
    assert.equal(result.status, 1, command);
    assert.match(result.stderr, new RegExp(`"${command}" was removed: `));
    assert.ok(result.stderr.includes(REMOVED_COMMANDS[command]), command);
  }
});

test("the bin entry runs the command line and prints no install command", () => {
  const result = runBin("--help");
  assert.equal(result.status, 0);
  assert.match(result.stdout, /does not serve a model\. The C engine does/);
  assert.doesNotMatch(result.stdout, /npx|npm install/);
});

test("schema refuses a path that is neither a file nor JSON, rather than compiling a stand-in", () => {
  const result = runBin("schema", "./no-such-schema.json");
  assert.equal(result.status, 1);
  assert.match(result.stderr, /neither an existing file nor valid JSON/);

  const inline = runBin("schema", '{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}', "--json");
  assert.equal(inline.status, 0);
  const pda = JSON.parse(inline.stdout);
  assert.equal(pda.requiredCount, 1);
});
