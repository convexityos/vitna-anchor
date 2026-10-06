// Gate A11: the three APIs coding agents speak, OpenAI's chat completions and
// Responses and Anthropic's Messages, as one conversation underneath. Each is
// converted into the conversation the chat template reads, written in the
// model's template, and its reply read back into reasoning, text and tool
// calls and returned in that API's own shape, whole or streamed.
//
// The pinned model is SmolLM2, too small to call a tool, so one server runs it
// with two test hooks: VITNA_TEST_TEMPLATE=qwen3 writes conversations in
// Qwen3's chat template, which tests/chat-template.test.mjs holds to
// transformers' rendering, and VITNA_TEST_REPLY=1 takes a request's
// vitna_test_reply as its reply, a token at a time in place of the model's
// choices, so everything after the choice runs as it would for a model's. In
// that mode each response names a hash of its prompt, which these tests
// compare with the conversation converted by hand and rendered by the
// engine's own chat-prompt. A second server runs SmolLM2 as it is, in ChatML,
// and answers one conversation through all three APIs with the model's reply.
//
// Needs a built engine and the model files (node scripts/fetch-model.mjs).
// Without them these tests are skipped, with the reason, unless
// VITNA_REQUIRE_REFERENCE=1 (as in CI), where they fail.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { keepStderr } from "./server-stderr.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
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
const A11 = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };
const MODEL = "smollm2-135m";
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];

const servers = {};

function start(args, env = {}) {
  const child = spawn(engine, ["serve", "--model", modelDir, "--model-id", MODEL, "--port", "0", ...onDevice, ...args], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, ...env },
  });
  const stderr = keepStderr(child, env);
  return new Promise((resolve, reject) => {
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
        resolve({ child, base: m[1] });
      }
    });
    // "close", not "exit": by then its stderr has all been read.
    child.on("close", (code) => reject(new Error(`the server exited with ${code}: ${out}${stderr.text}`)));
  });
}

before(async () => {
  if (A11.skip) return;
  assert.ok(!missing, missing ?? "");
  servers.qwen = await start(["--ctx", "2048"], { VITNA_TEST_TEMPLATE: "qwen3", VITNA_TEST_REPLY: "1" });
  servers.chatml = await start(["--ctx", "1024"]);
});

after(() => {
  for (const s of Object.values(servers)) s.child.kill();
});

async function post(server, path, body) {
  const res = await fetch(servers[server].base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
  const text = await res.text();
  return { status: res.status, headers: res.headers, text, json: res.headers.get("content-type")?.includes("json") ? JSON.parse(text) : null };
}

/** OpenAI's chat stream: each event's data, [DONE] included. */
function dataEvents(text) {
  return text
    .split("\n\n")
    .map((e) => e.trim())
    .filter((e) => e.startsWith("data: "))
    .map((e) => e.slice(6))
    .map((d) => (d === "[DONE]" ? d : JSON.parse(d)));
}

/** Anthropic's and Responses' streams: each event's name and data. */
function namedEvents(text) {
  return text
    .split("\n\n")
    .map((e) => e.trim())
    .filter(Boolean)
    .map((e) => {
      const lines = e.split("\n");
      return { event: lines.find((l) => l.startsWith("event: "))?.slice(7), data: JSON.parse(lines.find((l) => l.startsWith("data: ")).slice(6)) };
    });
}

/** FNV-1a over a prompt's UTF-8 bytes, as the server names its prompt in x-vitna-test-prompt. */
function fnv(text) {
  let h = 0xcbf29ce484222325n;
  for (const b of Buffer.from(text, "utf8")) h = BigInt.asUintN(64, (h ^ BigInt(b)) * 0x100000001b3n);
  return h.toString(16).padStart(16, "0");
}

/** The prompt Qwen3's template renders for a conversation, by the engine's chat-prompt. */
function rendered(req) {
  const r = spawnSync(engine, ["chat-prompt"], { input: JSON.stringify(req) + "\n", encoding: "utf8", timeout: 60_000 });
  assert.equal(r.status, 0, r.stderr);
  return JSON.parse(r.stdout.trim());
}

/** The usage counts that do not depend on what an earlier request left in the cache. */
function anthropicInput(u) {
  return u.input_tokens + u.cache_read_input_tokens;
}

// One conversation: a system prompt, a question, a tool call and its result,
// and a second question.
const SYSTEM = "You are terse.";
const WEATHER = {
  name: "get_weather",
  description: "The weather in a city.",
  parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
};

// As the template reads it, converted by hand: a call's arguments are an
// object, and a tool is a function with its name, description and parameters.
const CONVERSATION = [
  { role: "system", content: SYSTEM },
  { role: "user", content: "Weather in Paris?" },
  { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "get_weather", arguments: { city: "Paris" } } }] },
  { role: "tool", content: "Sunny, 21 C", tool_call_id: "c1" },
  { role: "user", content: "And Oslo?" },
];
const TOOLS = [{ type: "function", function: WEATHER }];

// As each API carries it. A call's arguments as a string of compact JSON, and
// fields the model was never trained to read, such as strict.
const CHAT_TOOLS = [{ type: "function", function: { ...WEATHER, strict: true } }];
const MESSAGES_TOOLS = [{ name: WEATHER.name, description: WEATHER.description, input_schema: WEATHER.parameters, cache_control: { type: "ephemeral" } }];
const RESPONSES_TOOLS = [{ type: "function", ...WEATHER, strict: false }];

const asChat = {
  messages: [
    { role: "system", content: SYSTEM },
    { role: "user", content: "Weather in Paris?" },
    { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }] },
    { role: "tool", tool_call_id: "c1", content: "Sunny, 21 C" },
    { role: "user", content: "And Oslo?" },
  ],
  tools: CHAT_TOOLS,
};
const asMessages = {
  system: [{ type: "text", text: SYSTEM, cache_control: { type: "ephemeral" } }],
  messages: [
    { role: "user", content: "Weather in Paris?" },
    { role: "assistant", content: [{ type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Paris" } }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: [{ type: "text", text: "Sunny, 21 C" }] }, { type: "text", text: "And Oslo?" }] },
  ],
  tools: MESSAGES_TOOLS,
  thinking: { type: "enabled", budget_tokens: 1024 },
  max_tokens: 64,
};
const asResponses = {
  instructions: SYSTEM,
  input: [
    { role: "user", content: "Weather in Paris?" },
    { type: "function_call", call_id: "c1", name: "get_weather", arguments: '{"city":"Paris"}' },
    { type: "function_call_output", call_id: "c1", output: "Sunny, 21 C" },
    { type: "message", role: "user", content: [{ type: "input_text", text: "And Oslo?" }] },
  ],
  tools: RESPONSES_TOOLS,
};

test("the three APIs are one conversation: each is written as the conversation converted by hand, in Qwen3's template", A11, async () => {
  const want = fnv(rendered({ messages: CONVERSATION, tools: TOOLS }));
  const reply = { vitna_test_reply: "Rain." };
  const chat = await post("qwen", "/v1/chat/completions", { ...asChat, ...reply, max_tokens: 16 });
  const messages = await post("qwen", "/v1/messages", { ...asMessages, ...reply });
  const responses = await post("qwen", "/v1/responses", { ...asResponses, ...reply });
  for (const [name, r] of [["chat", chat], ["messages", messages], ["responses", responses]]) {
    assert.equal(r.status, 200, `${name}: ${r.text}`);
    assert.equal(r.headers.get("x-vitna-test-prompt"), want, `${name}: the prompt`);
  }
  // The same tokens read, as each API counts them, and as Anthropic's token counter counts them.
  const n = chat.json.usage.prompt_tokens;
  assert.equal(anthropicInput(messages.json.usage), n);
  assert.equal(responses.json.usage.input_tokens, n);
  const counted = await post("qwen", "/v1/messages/count_tokens", asMessages);
  assert.equal(counted.status, 200, counted.text);
  assert.deepEqual(counted.json, { input_tokens: n });
  // Anthropic's API thinks only when asked: without thinking, the template's empty reasoning opens the reply.
  const off = await post("qwen", "/v1/messages", { ...asMessages, thinking: undefined, ...reply });
  assert.equal(off.headers.get("x-vitna-test-prompt"), fnv(rendered({ messages: CONVERSATION, tools: TOOLS, enable_thinking: false })));
  const health = await (await fetch(servers.qwen.base + "/v1/health")).json();
  assert.equal(health.chat_template, "qwen3");
});

// A reply in Qwen3's format: reasoning, text, then a call.
const REPLY = '<think>\nThe user wants Oslo.\n</think>\n\nChecking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>';
const ASK = { role: "user", content: "Weather in Oslo?" };
// A call's arguments as the template writes them, which is how the model was trained to read them back.
const ARGS = '{"city": "Oslo"}';

test("OpenAI's chat returns the reasoning, the text and the call, whole and streamed", A11, async () => {
  const body = { messages: [ASK], tools: CHAT_TOOLS, max_tokens: 200, vitna_test_reply: REPLY };
  const whole = await post("qwen", "/v1/chat/completions", body);
  assert.equal(whole.status, 200, whole.text);
  const choice = whole.json.choices[0];
  assert.equal(choice.finish_reason, "tool_calls");
  assert.equal(choice.message.role, "assistant");
  assert.equal(choice.message.reasoning_content, "The user wants Oslo.");
  assert.equal(choice.message.content, "Checking.");
  assert.equal(choice.message.tool_calls.length, 1);
  const call = choice.message.tool_calls[0];
  assert.match(call.id, /^call_[0-9a-f]{16}$/);
  assert.equal(call.type, "function");
  assert.equal(call.function.name, "get_weather");
  assert.equal(call.function.arguments, ARGS);

  const stream = await post("qwen", "/v1/chat/completions", { ...body, stream: true });
  assert.equal(stream.status, 200, stream.text);
  const ev = dataEvents(stream.text);
  assert.equal(ev.at(-1), "[DONE]");
  const deltas = ev.slice(0, -1).map((c) => c.choices[0].delta);
  assert.deepEqual(deltas[0], { role: "assistant", content: "" });
  assert.equal(deltas.map((d) => d.reasoning_content ?? "").join(""), "The user wants Oslo.");
  assert.equal(deltas.map((d) => d.content ?? "").join(""), "Checking.");
  const lastReasoning = deltas.findLastIndex((d) => d.reasoning_content !== undefined);
  const firstContent = deltas.findIndex((d, i) => i > 0 && d.content !== undefined);
  assert.ok(lastReasoning < firstContent, "the reasoning streams before the text");
  const calls = deltas.flatMap((d) => d.tool_calls ?? []);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].index, 0);
  assert.match(calls[0].id, /^call_[0-9a-f]{16}$/);
  assert.deepEqual(calls[0].function, { name: "get_weather", arguments: ARGS });
  const last = ev.at(-2);
  assert.equal(last.choices[0].finish_reason, "tool_calls");
  assert.equal(last.usage.prompt_tokens, whole.json.usage.prompt_tokens);
  assert.equal(last.usage.completion_tokens, whole.json.usage.completion_tokens);
});

test("Anthropic's Messages returns thinking, text and tool use, whole and as its stream's events", A11, async () => {
  const body = { messages: [ASK], tools: MESSAGES_TOOLS, thinking: { type: "enabled", budget_tokens: 1024 }, max_tokens: 200, vitna_test_reply: REPLY };
  const whole = await post("qwen", "/v1/messages", body);
  assert.equal(whole.status, 200, whole.text);
  const m = whole.json;
  assert.match(m.id, /^msg_[0-9a-f]{16}$/);
  assert.equal(m.type, "message");
  assert.equal(m.role, "assistant");
  assert.equal(m.model, MODEL);
  assert.equal(m.stop_reason, "tool_use");
  assert.equal(m.stop_sequence, null);
  assert.equal(m.content.length, 3);
  assert.deepEqual(m.content[0], { type: "thinking", thinking: "The user wants Oslo.", signature: "" });
  assert.deepEqual(m.content[1], { type: "text", text: "Checking." });
  const { id, ...use } = m.content[2];
  assert.match(id, /^toolu_[0-9a-f]{16}$/);
  assert.deepEqual(use, { type: "tool_use", name: "get_weather", input: { city: "Oslo" } });
  assert.deepEqual(Object.keys(m.usage), ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]);

  const stream = await post("qwen", "/v1/messages", { ...body, stream: true });
  assert.equal(stream.status, 200, stream.text);
  assert.match(stream.headers.get("content-type"), /^text\/event-stream/);
  const ev = namedEvents(stream.text);
  for (const e of ev) assert.equal(e.event, e.data.type, "each event is named for its type");
  assert.deepEqual(
    ev.map((e) => e.event).filter((t) => t !== "content_block_delta"),
    ["message_start", "content_block_start", "content_block_stop", "content_block_start", "content_block_stop", "content_block_start", "content_block_stop", "message_delta", "message_stop"],
  );
  // The blocks rebuilt from the events are the message's.
  const blocks = [];
  for (const { data } of ev) {
    if (data.type === "content_block_start") blocks[data.index] = { ...data.content_block, partial: "" };
    if (data.type === "content_block_delta") {
      const b = blocks[data.index];
      if (data.delta.type === "thinking_delta") b.thinking += data.delta.thinking;
      else if (data.delta.type === "text_delta") b.text += data.delta.text;
      else if (data.delta.type === "input_json_delta") b.partial += data.delta.partial_json;
    }
  }
  const rebuilt = blocks.map(({ partial, ...b }) => (b.type === "tool_use" ? { ...b, input: JSON.parse(partial) } : b));
  const withoutIds = (bs) => bs.map(({ id, ...b }) => b);
  assert.deepEqual(withoutIds(rebuilt), withoutIds(m.content));
  const opening = ev[0].data.message;
  assert.equal(opening.stop_reason, null);
  assert.deepEqual(opening.content, []);
  assert.equal(anthropicInput(opening.usage), anthropicInput(m.usage));
  const end = ev.at(-2).data;
  assert.deepEqual(end.delta, { stop_reason: "tool_use", stop_sequence: null });
  assert.equal(end.usage.output_tokens, m.usage.output_tokens);
});

test("OpenAI's Responses returns reasoning, a message and a function call as output items, whole and as its stream's events", A11, async () => {
  const body = { input: [ASK], tools: RESPONSES_TOOLS, max_output_tokens: 200, vitna_test_reply: REPLY };
  const whole = await post("qwen", "/v1/responses", body);
  assert.equal(whole.status, 200, whole.text);
  const r = whole.json;
  assert.match(r.id, /^resp_[0-9a-f]{16}$/);
  assert.equal(r.object, "response");
  assert.equal(r.status, "completed");
  assert.equal(r.model, MODEL);
  assert.equal(r.error, null);
  assert.deepEqual(r.output.map((o) => o.type), ["reasoning", "message", "function_call"]);
  assert.deepEqual(r.output[0].content, [{ type: "reasoning_text", text: "The user wants Oslo." }]);
  assert.equal(r.output[1].role, "assistant");
  assert.equal(r.output[1].status, "completed");
  assert.deepEqual(r.output[1].content, [{ type: "output_text", text: "Checking.", annotations: [], logprobs: [] }]);
  const fc = r.output[2];
  assert.match(fc.id, /^fc_[0-9a-f]{16}$/);
  assert.match(fc.call_id, /^call_[0-9a-f]{16}$/);
  assert.equal(fc.name, "get_weather");
  assert.equal(fc.arguments, ARGS);
  assert.equal(fc.status, "completed");
  const u = r.usage;
  assert.equal(u.total_tokens, u.input_tokens + u.output_tokens);
  const thought = u.output_tokens_details.reasoning_tokens;
  assert.ok(thought > 0 && thought < u.output_tokens, `the reasoning's tokens are some of the reply's: ${thought} of ${u.output_tokens}`);
  assert.deepEqual(r.tools, RESPONSES_TOOLS);

  const stream = await post("qwen", "/v1/responses", { ...body, stream: true });
  assert.equal(stream.status, 200, stream.text);
  const ev = namedEvents(stream.text);
  ev.forEach((e, i) => {
    assert.equal(e.event, e.data.type, "each event is named for its type");
    assert.equal(e.data.sequence_number, i, "events are numbered in order from 0");
  });
  assert.deepEqual(ev.slice(0, 2).map((e) => e.event), ["response.created", "response.in_progress"]);
  assert.equal(ev[0].data.response.status, "in_progress");
  assert.equal(ev.at(-1).event, "response.completed");
  const done = ev.at(-1).data.response;
  const withoutIds = (items) => items.map(({ id, call_id, ...o }) => o);
  assert.deepEqual(withoutIds(done.output), withoutIds(r.output));
  assert.deepEqual(done.usage.output_tokens_details, u.output_tokens_details);
  // Each item's done event holds it as the response does, and its deltas add up to it.
  assert.deepEqual(ev.filter((e) => e.event === "response.output_item.done").map((e) => e.data.item), done.output);
  const joined = (type) => ev.filter((e) => e.event === type).map((e) => e.data.delta).join("");
  assert.equal(joined("response.reasoning_text.delta"), "The user wants Oslo.");
  assert.equal(joined("response.output_text.delta"), "Checking.");
  assert.equal(joined("response.function_call_arguments.delta"), ARGS);
});

test("a forced tool call is written into the prompt for the model to finish, and tool_choice none leaves a call as text", A11, async () => {
  const conversation = [{ role: "user", content: "Weather in Rome?" }];
  // A forced call is written from the reply's first token, so the model does not think first.
  const plain = rendered({ messages: conversation, tools: TOOLS, enable_thinking: false });
  const required = await post("qwen", "/v1/chat/completions", {
    messages: conversation,
    tools: CHAT_TOOLS,
    tool_choice: "required",
    max_tokens: 64,
    vitna_test_reply: 'get_weather", "arguments": {"city": "Rome"}}\n</tool_call>',
  });
  assert.equal(required.status, 200, required.text);
  assert.equal(required.headers.get("x-vitna-test-prompt"), fnv(plain + '<tool_call>\n{"name": "'), "the opening, up to the name the model chooses");
  const choice = required.json.choices[0];
  assert.equal(choice.finish_reason, "tool_calls");
  assert.equal(choice.message.content, null);
  assert.deepEqual(choice.message.tool_calls.map((c) => [c.function.name, JSON.parse(c.function.arguments)]), [["get_weather", { city: "Rome" }]]);

  const named = await post("qwen", "/v1/messages", {
    messages: conversation,
    tools: MESSAGES_TOOLS,
    tool_choice: { type: "tool", name: "get_weather" },
    max_tokens: 64,
    vitna_test_reply: '{"city": "Rome"}}\n</tool_call>',
  });
  assert.equal(named.status, 200, named.text);
  assert.equal(named.headers.get("x-vitna-test-prompt"), fnv(plain + '<tool_call>\n{"name": "get_weather", "arguments": '), "the opening, up to the arguments");
  assert.deepEqual(named.json.content.map((b) => [b.type, b.name, b.input]), [["tool_use", "get_weather", { city: "Rome" }]]);
  assert.equal(named.json.stop_reason, "tool_use");

  // none: the tools stay in the prompt, so it starts the same, and a call the model writes anyway is text.
  const call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>';
  const none = await post("qwen", "/v1/responses", { input: conversation, tools: RESPONSES_TOOLS, tool_choice: "none", vitna_test_reply: call });
  assert.equal(none.status, 200, none.text);
  assert.equal(none.headers.get("x-vitna-test-prompt"), fnv(rendered({ messages: conversation, tools: TOOLS })));
  assert.deepEqual(none.json.output.map((o) => o.type), ["message"]);
  assert.equal(none.json.output[0].content[0].text, call);

  // As Anthropic's API refuses it: a forced call with thinking.
  const both = await post("qwen", "/v1/messages", {
    messages: conversation,
    tools: MESSAGES_TOOLS,
    tool_choice: { type: "any" },
    thinking: { type: "enabled", budget_tokens: 1024 },
    max_tokens: 64,
  });
  assert.equal(both.status, 400, both.text);
  assert.equal(both.json.type, "error");
  assert.equal(both.json.error.type, "invalid_request_error");
});

test("stop sequences, the token limit and a prefill end or continue a reply in each API's terms", A11, async () => {
  const ask = [{ role: "user", content: "Say hello." }];
  const stopped = await post("qwen", "/v1/messages", { messages: ask, max_tokens: 64, stop_sequences: ["world"], vitna_test_reply: "Hello world. Bye." });
  assert.equal(stopped.status, 200, stopped.text);
  assert.deepEqual(stopped.json.content, [{ type: "text", text: "Hello " }]);
  assert.equal(stopped.json.stop_reason, "stop_sequence");
  assert.equal(stopped.json.stop_sequence, "world");
  const streamedStop = await post("qwen", "/v1/messages", { messages: ask, max_tokens: 64, stop_sequences: ["world"], vitna_test_reply: "Hello world. Bye.", stream: true });
  assert.deepEqual(namedEvents(streamedStop.text).at(-2).data.delta, { stop_reason: "stop_sequence", stop_sequence: "world" });

  const long = "One two three four five six seven eight.";
  const cut = await post("qwen", "/v1/messages", { messages: ask, max_tokens: 2, vitna_test_reply: long });
  assert.equal(cut.json.stop_reason, "max_tokens");
  assert.equal(cut.json.usage.output_tokens, 2);
  const incomplete = await post("qwen", "/v1/responses", { input: ask, max_output_tokens: 2, vitna_test_reply: long });
  assert.equal(incomplete.json.status, "incomplete");
  assert.deepEqual(incomplete.json.incomplete_details, { reason: "max_output_tokens" });
  assert.equal(incomplete.json.max_output_tokens, 2);
  const streamedCut = await post("qwen", "/v1/responses", { input: ask, max_output_tokens: 2, vitna_test_reply: long, stream: true });
  assert.equal(namedEvents(streamedCut.text).at(-1).event, "response.incomplete");
  const length = await post("qwen", "/v1/chat/completions", { messages: ask, max_tokens: 2, vitna_test_reply: long });
  assert.equal(length.json.choices[0].finish_reason, "length");

  // An Anthropic conversation that ends with the assistant is continued from there.
  const prefill = [{ role: "user", content: "Count to three." }, { role: "assistant", content: "One, two," }];
  const cont = await post("qwen", "/v1/messages", { messages: prefill, max_tokens: 16, vitna_test_reply: " three." });
  assert.equal(cont.status, 200, cont.text);
  assert.deepEqual(cont.json.content, [{ type: "text", text: " three." }]);
  assert.equal(cont.json.stop_reason, "end_turn");
  const whole = rendered({ messages: prefill, add_generation_prompt: false });
  assert.ok(whole.endsWith("One, two,<|im_end|>\n"), whole);
  assert.equal(cont.headers.get("x-vitna-test-prompt"), fnv(whole.slice(0, -"<|im_end|>\n".length)), "the last message, left open");
});

test("refusals and errors come in each API's own shape", A11, async () => {
  const ask = [{ role: "user", content: "hi" }];
  const wrongModel = await post("qwen", "/v1/messages", { model: "claude-x", messages: ask, max_tokens: 8 });
  assert.equal(wrongModel.status, 404);
  assert.deepEqual(Object.keys(wrongModel.json), ["type", "error"]);
  assert.equal(wrongModel.json.type, "error");
  assert.equal(wrongModel.json.error.type, "not_found_error");
  const noMax = await post("qwen", "/v1/messages", { messages: ask });
  assert.equal(noMax.status, 400);
  assert.equal(noMax.json.error.type, "invalid_request_error");
  assert.match(noMax.json.error.message, /max_tokens/);
  const image = await post("qwen", "/v1/messages", {
    messages: [{ role: "user", content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "AA==" } }] }],
    max_tokens: 8,
  });
  assert.equal(image.status, 400);
  assert.match(image.json.error.message, /messages\.0\.content\.0/);
  const get = await fetch(servers.qwen.base + "/v1/messages");
  assert.equal(get.status, 405);
  assert.equal((await get.json()).type, "error");

  const stored = await post("qwen", "/v1/responses", { input: "hi", previous_response_id: "resp_1" });
  assert.equal(stored.status, 400);
  assert.equal(stored.json.error.param, "previous_response_id");
  assert.equal(stored.json.error.code, "unsupported_parameter");
  const hosted = await post("qwen", "/v1/responses", { input: "hi", tools: [{ type: "web_search" }] });
  assert.equal(hosted.status, 400);
  assert.equal(hosted.json.error.param, "tools[0]");
  const missingTool = await post("qwen", "/v1/chat/completions", { messages: ask, tools: CHAT_TOOLS, tool_choice: { type: "function", function: { name: "nope" } } });
  assert.equal(missingTool.status, 400);
  assert.equal(missingTool.json.error.param, "tool_choice");
});

test("a ChatML model answers one conversation the same through all three APIs, and refuses tools", A11, async () => {
  const system = "You are a helpful assistant.";
  const user = "What is the capital of France?";
  const chat = await post("chatml", "/v1/chat/completions", { messages: [{ role: "system", content: system }, { role: "user", content: user }], max_tokens: 12, temperature: 0 });
  const messages = await post("chatml", "/v1/messages", { system, messages: [{ role: "user", content: user }], max_tokens: 12, temperature: 0 });
  const responses = await post("chatml", "/v1/responses", { instructions: system, input: user, max_output_tokens: 12, temperature: 0 });
  for (const r of [chat, messages, responses]) assert.equal(r.status, 200, r.text);
  const text = chat.json.choices[0].message.content;
  assert.ok(text.length > 0);
  assert.deepEqual(messages.json.content, [{ type: "text", text }]);
  assert.deepEqual(responses.json.output.map((o) => o.content[0].text), [text]);
  assert.equal(anthropicInput(messages.json.usage), chat.json.usage.prompt_tokens);
  assert.equal(responses.json.usage.input_tokens, chat.json.usage.prompt_tokens);
  const streamed = await post("chatml", "/v1/messages", { system, messages: [{ role: "user", content: user }], max_tokens: 12, temperature: 0, stream: true });
  const ev = namedEvents(streamed.text);
  assert.equal(ev.filter((e) => e.event === "content_block_delta").map((e) => e.data.delta.text).join(""), text);
  const tools = await post("chatml", "/v1/messages", { messages: [{ role: "user", content: user }], tools: MESSAGES_TOOLS, max_tokens: 8 });
  assert.equal(tools.status, 400);
  assert.equal(tools.json.type, "error");
  assert.match(tools.json.error.message, /tool/);
  const health = await (await fetch(servers.chatml.base + "/v1/health")).json();
  assert.equal(health.chat_template, "chatml");
});
