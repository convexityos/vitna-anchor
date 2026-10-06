// Gate A11: a conversation written in the model's own chat template.
//
// reference/qwen3-30b-a3b/chat-template.json holds conversations in OpenAI's
// message shape, with tools, tool calls, tool results and reasoning, each with
// the prompt Qwen3-30B-A3B's chat template renders for it through
// transformers (reference/record_chat_template.py). The engine writes the
// template in C (engine/src/chat.c), and its chat-prompt command must write
// every one of those prompts, byte for byte. No model is needed, only a built
// engine, so CI runs it.

import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const fixture = JSON.parse(readFileSync(here("../reference/qwen3-30b-a3b/chat-template.json"), "utf8"));
const cases = JSON.parse(readFileSync(here("../reference/qwen3-30b-a3b/chat-cases.json"), "utf8")).cases;
const pin = JSON.parse(readFileSync(here("../reference/qwen3-30b-a3b/model.json"), "utf8"));
const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");

const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const ENGINE = { skip: !engine && process.env.VITNA_REQUIRE_ENGINE !== "1" && "no built engine found" };

test("the fixture is the one its pin, its inputs and its recorder describe", () => {
  assert.equal(fixture.format, 1);
  const entry = pin.files.find((f) => f.path === "tokenizer_config.json");
  assert.deepEqual([fixture.model.repo, fixture.model.revision, fixture.model.sha256], [pin.repo, pin.revision, entry.sha256]);
  assert.equal(fixture.inputs.sha256_lf, sha256Lf(here("../reference/qwen3-30b-a3b/chat-cases.json")), "chat-cases.json changed: record again");
  assert.equal(fixture.recorder.sha256_lf, sha256Lf(here("../reference/record_chat_template.py")), "record_chat_template.py changed: record again");
  assert.deepEqual(fixture.software, { transformers: "5.17.0", jinja2: "3.1.6", python: fixture.software.python });
  assert.deepEqual(fixture.cases.map((c) => c.id), cases.map((c) => c.id));
  for (const [i, c] of fixture.cases.entries()) {
    const { prompt, request, ...input } = c;
    assert.deepEqual(input, cases[i], `${c.id}: the recorded input is not the case`);
    assert.equal(typeof prompt, "string");
    const { id, ...req } = cases[i];
    assert.deepEqual(JSON.parse(request), req, `${c.id}: the request is not the case`);
  }
});

test("the engine writes every conversation as Qwen3's chat template does, byte for byte", ENGINE, (t) => {
  assert.ok(engine, "no built engine found");
  // Each request as Python wrote it: JSON.stringify would write 100.0 as 100.
  const input = fixture.cases.map((c) => c.request).join("\n") + "\n";
  const r = spawnSync(engine, ["chat-prompt"], { input, encoding: "utf8", maxBuffer: 1 << 26 });
  assert.equal(r.status, 0, r.stderr);
  const lines = r.stdout.trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, fixture.cases.length);
  for (const [i, c] of fixture.cases.entries()) {
    if (lines[i] === c.prompt) continue;
    let at = 0;
    while (at < c.prompt.length && lines[i][at] === c.prompt[at]) at++;
    assert.fail(`${c.id}: differs at character ${at}: engine ${JSON.stringify(lines[i].slice(at, at + 60))}, template ${JSON.stringify(c.prompt.slice(at, at + 60))}`);
  }
  t.diagnostic(`all ${fixture.cases.length} conversations: tools, tool calls with string and object arguments, tool results, reasoning and the thinking switch`);
});
