// Gate A12's chat page (engine/web/chat.html), which the server answers GET /
// with from inside the engine (engine/src/chat_page.c, written from the page
// by engine/tools/embed_page.mjs).
//
// The page's script keeps its parts that need no browser apart from the
// ones that do, so these tests run them in Node: Markdown rendered safely,
// server-sent events read across reads, and each chat chunk's effect on the
// reply. A server started on the pinned model must serve the page as it is,
// under a policy that lets it reach its own server alone. Needs a built engine
// and the model files for that last part; skipped with the reason otherwise,
// unless VITNA_REQUIRE_REFERENCE=1 (as in CI), where it fails.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { embedPage, PAGE, SOURCE } from "../engine/tools/embed_page.mjs";
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
const SERVED = { skip: process.env.VITNA_REQUIRE_REFERENCE === "1" ? false : missing ?? false };

const html = readFileSync(PAGE, "utf8");

/** The page's script, run in Node without a document: its pure parts, as the page defines them. */
function pageParts() {
  // The page's one script, between its tags, by position: this is the page's own text, not markup to filter.
  const script = html.slice(html.indexOf("<script>") + "<script>".length, html.lastIndexOf("</script>"));
  const context = vm.createContext({});
  vm.runInContext(script + "\n;({ escapeHtml, renderMarkdown, parseEvents, applyChunk, newReply, statsLine, samplingFor })", context);
  return vm.runInContext("({ escapeHtml, renderMarkdown, parseEvents, applyChunk, newReply, statsLine, samplingFor })", context);
}

test("the page built into the engine is engine/web/chat.html, byte for byte", () => {
  assert.equal(html.includes("\r"), false, "the page keeps LF line endings");
  assert.equal(readFileSync(SOURCE, "utf8"), embedPage(html), "run node engine/tools/embed_page.mjs");
  // MSVC takes a string literal of at most 64 KiB once its pieces are joined.
  assert.ok(Buffer.byteLength(html) < 60000, `${Buffer.byteLength(html)} bytes`);
  assert.equal(html.includes(String.fromCharCode(0x2014)), false, "no em-dashes");
  assert.equal(/https?:\/\/(?!127\.0\.0\.1)/.test(html.replace(/https\?:\\\/\\\//g, "")), false, "the page loads nothing from anywhere");
});

test("Markdown is rendered from escaped text, so nothing a model writes becomes markup", () => {
  const { renderMarkdown } = pageParts();
  assert.equal(renderMarkdown("<script>alert(1)</script>"), "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>");
  assert.equal(renderMarkdown('<img src=x onerror="alert(1)">'), "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt;</p>");
  // Only http and https links are links.
  assert.equal(renderMarkdown("[a](javascript:alert(1))"), "<p>[a](javascript:alert(1))</p>");
  assert.equal(renderMarkdown("[docs](https://example.com/x)"), '<p><a href="https://example.com/x" target="_blank" rel="noopener noreferrer">docs</a></p>');
  // A quote in a link was escaped before the link was made, so it stays inside
  // the href: the anchor's only quotes are its three attributes' own.
  const injected = renderMarkdown('[x](https://e.com/"onmouseover="alert(1))');
  assert.equal(injected, '<p><a href="https://e.com/&quot;onmouseover=&quot;alert(1" target="_blank" rel="noopener noreferrer">x</a>)</p>');
  assert.equal(injected.split('"').length - 1, 6);
  // Nothing inside a code span or block is formatted.
  assert.equal(renderMarkdown("Use `**not bold**` and **bold**."), "<p>Use <code>**not bold**</code> and <strong>bold</strong>.</p>");
  assert.equal(renderMarkdown("```js\nconst a = 1 < 2;\n**x**\n```"), '<pre><code data-lang="js">const a = 1 &lt; 2;\n**x**</code></pre>');
  // A fence still open, as one is while a reply streams, runs to the end.
  assert.equal(renderMarkdown("Here:\n```\nline one"), "<p>Here:</p><pre><code>line one</code></pre>");
});

test("Markdown's lists, headings, quotes, tables and paragraphs", () => {
  const { renderMarkdown } = pageParts();
  assert.equal(renderMarkdown("# Title\nText\nmore\n\nNext"), "<h1>Title</h1><p>Text<br>more</p><p>Next</p>");
  assert.equal(renderMarkdown("- a\n- *b*\n\n1. one\n2) two"), "<ul><li>a</li><li><em>b</em></li></ul><ol><li>one</li><li>two</li></ol>");
  assert.equal(renderMarkdown("> quoted\n> on"), "<blockquote>quoted<br>on</blockquote>");
  assert.equal(
    renderMarkdown("| a | b |\n|---|:---:|\n| 1 | `2` |"),
    "<table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>1</td><td><code>2</code></td></tr></tbody></table>",
  );
  assert.equal(renderMarkdown("| not | a table |"), "<p>| not | a table |</p>");
  assert.equal(renderMarkdown("2 * 3 * 4"), "<p>2 * 3 * 4</p>");
});

test("server-sent events are read across reads, a partial event kept for the next", () => {
  const { parseEvents } = pageParts();
  const a = parseEvents('data: {"a":1}\n\ndata: {"b"');
  assert.deepEqual([...a.events], ['{"a":1}']);
  assert.equal(a.rest, 'data: {"b"');
  const b = parseEvents(a.rest + ':2}\n\ndata: [DONE]\n\n');
  assert.deepEqual([...b.events], ['{"b":2}', "[DONE]"]);
  assert.equal(b.rest, "");
  assert.deepEqual([...parseEvents("event: x\ndata: one\ndata: two\n\n").events], ["one\ntwo"]);
});

test("each chat chunk adds to the reply's reasoning or text, and the last says how it ended", () => {
  const { applyChunk, newReply, statsLine, samplingFor } = pageParts();
  const reply = newReply();
  const chunk = (delta, extra = {}) => JSON.stringify({ choices: [{ index: 0, delta, finish_reason: null, ...extra }] });
  applyChunk(reply, chunk({ role: "assistant", content: "" }));
  applyChunk(reply, chunk({ reasoning_content: "Think" }));
  applyChunk(reply, chunk({ reasoning_content: "ing." }));
  applyChunk(reply, chunk({ content: "Hi" }));
  applyChunk(reply, chunk({ content: " there." }));
  const usage = { prompt_tokens: 1200, completion_tokens: 40, total_tokens: 1240, prompt_tokens_details: { cached_tokens: 1000 } };
  applyChunk(reply, JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage }));
  applyChunk(reply, "[DONE]");
  assert.equal(reply.reasoning, "Thinking.");
  assert.equal(reply.content, "Hi there.");
  assert.equal(reply.finish, "stop");
  assert.equal(reply.done, true);
  assert.equal(statsLine(reply.usage, 2), "40 tokens, 20.0 a second, prompt 1,200 tokens (1,000 reused)");
  // The error event a stream that fails ends with.
  const failed = applyChunk(newReply(), JSON.stringify({ error: { message: "The model failed to run a step." } }));
  assert.equal(failed.error, "The model failed to run a step.");
  // Qwen3's recommended sampling, thinking or not.
  assert.deepEqual({ ...samplingFor("qwen3", true) }, { temperature: 0.6, top_p: 0.95, top_k: 20 });
  assert.deepEqual({ ...samplingFor("qwen3", false) }, { temperature: 0.7, top_p: 0.8, top_k: 20 });
  assert.deepEqual({ ...samplingFor("chatml", true) }, { temperature: 0.7 });
});

let child = null;
let base = null;

before(async () => {
  if (SERVED.skip) return;
  assert.ok(!missing, missing ?? "");
  child = spawn(engine, ["serve", "--model", modelDir, "--model-id", "smollm2-135m", "--port", "0", "--ctx", "512"], { stdio: ["ignore", "pipe", "pipe"] });
  const stderr = keepStderr(child);
  base = await new Promise((resolve, reject) => {
    let out = "";
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
    child.on("close", (code) => reject(new Error(`the server exited with ${code}: ${out}${stderr.text}`)));
  });
});

after(() => {
  if (child) child.kill();
});

test("the server answers / with the page, under a policy that lets it reach this server alone", SERVED, async () => {
  for (const path of ["/", "/chat"]) {
    const res = await fetch(base + path);
    assert.equal(res.status, 200, path);
    assert.equal(res.headers.get("content-type"), "text/html; charset=utf-8");
    assert.equal(await res.text(), html, `${path} is the page, byte for byte`);
    const csp = res.headers.get("content-security-policy");
    assert.match(csp, /default-src 'none'/);
    assert.match(csp, /connect-src 'self'/);
    assert.match(csp, /frame-ancestors 'none'/);
    assert.equal(res.headers.get("x-content-type-options"), "nosniff");
  }
  // HEAD gives the same headers and no body, which Node's parser would refuse.
  const head = await new Promise((resolve, reject) => {
    const url = new URL(base + "/");
    const req = httpRequest({ hostname: url.hostname, port: url.port, path: "/", method: "HEAD" }, (res) => {
      res.resume();
      res.on("end", () => resolve(res));
    });
    req.on("error", reject);
    req.end();
  });
  assert.equal(head.statusCode, 200);
  assert.equal(Number(head.headers["content-length"]), Buffer.byteLength(html));
  // What the page asks of the server, it has: the health it reads its model from.
  const health = await (await fetch(base + "/v1/health")).json();
  assert.equal(health.model, "smollm2-135m");
  assert.equal(health.chat_template, "chatml");
});
