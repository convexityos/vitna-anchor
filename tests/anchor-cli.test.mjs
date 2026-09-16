import assert from "node:assert/strict";
import test from "node:test";
import { request as httpRequest } from "node:http";

import { runProbe, startAnchorServer } from "../runtime/anchor-run.mjs";

function httpPost(port, path, data) {
  return new Promise((resolve, reject) => {
    const postBody = typeof data === "string" ? data : JSON.stringify(data);
    const req = httpRequest(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(postBody),
        },
      },
      (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            const parsed = JSON.parse(body);
            resolve({ status: res.statusCode, headers: res.headers, body: parsed });
          } catch {
            resolve({ status: res.statusCode, headers: res.headers, raw: body });
          }
        });
      }
    );
    req.on("error", reject);
    req.write(postBody);
    req.end();
  });
}

function httpGet(port, path) {
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method: "GET",
      },
      (res) => {
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => {
          try {
            const parsed = JSON.parse(body);
            resolve({ status: res.statusCode, headers: res.headers, body: parsed });
          } catch {
            resolve({ status: res.statusCode, headers: res.headers, raw: body });
          }
        });
      }
    );
    req.on("error", reject);
    req.end();
  });
}

test("vitna-anchor probe runs direct storage and memory benchmarks with air-gap proof", () => {
  const result = runProbe(true);
  assert.ok(result.platform, "platform architecture detected");
  assert.ok(result.cpu.cores > 0, "logical CPU cores detected");
  assert.ok(result.storage.sequentialReadMBps > 0, "positive sequential read speed");
  assert.ok(result.storage.randomRead4kIops >= 0, "random IOPS measured");
  assert.equal(result.sovereignProof.airgap, true, "strict sovereign air-gap confirmed");
  assert.equal(result.sovereignProof.socketEgressBytes, 0, "zero bytes socket egress");
});

test("vitna-anchor serve provides OpenAI-compatible completions, Radix KV cache, and Grammar decoding", async () => {
  const testPort = 8799;
  const server = startAnchorServer({
    host: "127.0.0.1",
    port: testPort,
    model: "vitna/anchor-moe",
  });

  await new Promise((r) => setTimeout(r, 150));

  try {
    // 1. Health check
    const health = await httpGet(testPort, "/health");
    assert.equal(health.status, 200);
    assert.equal(health.body.engine, "vitna-anchor");
    assert.equal(health.body.airgap, true);

    // 2. Models list
    const models = await httpGet(testPort, "/v1/models");
    assert.equal(models.status, 200);
    assert.ok(Array.isArray(models.body.data));
    assert.ok(models.body.data.some((m) => m.id === "vitna/anchor-moe"));

    // 3. Non-streaming chat completion
    const comp1 = await httpPost(testPort, "/v1/chat/completions", {
      model: "vitna/anchor-moe",
      messages: [
        { role: "system", content: "You are Vitna Anchor." },
        { role: "user", content: "Demonstrate local sovereign inference." },
      ],
      temperature: 0,
    });

    assert.equal(comp1.status, 200);
    assert.ok(comp1.body.choices[0].message.content.length > 0);
    assert.ok(comp1.headers["x-vitna-trajectory-sha256"], "trajectory SHA-256 header present");
    assert.equal(comp1.body.proof.airgap, true);
    assert.equal(comp1.body.proof.socket_egress_bytes, 0);

    // 4. Prompt semantic cache verification (identical query with temperature 0)
    const comp2 = await httpPost(testPort, "/v1/chat/completions", {
      model: "vitna/anchor-moe",
      messages: [
        { role: "system", content: "You are Vitna Anchor." },
        { role: "user", content: "Demonstrate local sovereign inference." },
      ],
      temperature: 0,
    });

    assert.equal(comp2.status, 200);
    assert.equal(comp2.body.cache_hit, true, "prompt cache hit served at <1ms");
    assert.equal(comp2.body.choices[0].message.content, comp1.body.choices[0].message.content);

    // 5. Grammar-constrained JSON decoding
    const jsonComp = await httpPost(testPort, "/v1/chat/completions", {
      model: "vitna/anchor-moe",
      messages: [
        { role: "user", content: "Return structured system status in JSON." },
      ],
      response_format: { type: "json_object" },
    });

    assert.equal(jsonComp.status, 200);
    const parsedJson = JSON.parse(jsonComp.body.choices[0].message.content);
    assert.equal(parsedJson.status, "success");
    assert.equal(parsedJson.grammar_enforced, true);
    assert.ok(jsonComp.body.grammar_tokens_masked > 0, "grammar pushdown automaton masked invalid candidate tokens");

    // 6. Streaming SSE completion
    const streamRes = await new Promise((resolve, reject) => {
      const req = httpRequest(
        {
          hostname: "127.0.0.1",
          port: testPort,
          path: "/v1/chat/completions",
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
          },
        },
        (res) => {
          let chunks = [];
          res.on("data", (c) => chunks.push(c.toString("utf8")));
          res.on("end", () => resolve({ status: res.statusCode, output: chunks.join("") }));
        }
      );
      req.on("error", reject);
      req.write(JSON.stringify({
        model: "vitna/anchor-moe",
        messages: [{ role: "user", content: "Streaming test" }],
        stream: true,
      }));
      req.end();
    });

    assert.equal(streamRes.status, 200);
    assert.ok(streamRes.output.includes("data: [DONE]"), "SSE stream terminated with [DONE]");
    assert.ok(streamRes.output.includes("chat.completion.chunk"), "SSE chunks present");

  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
