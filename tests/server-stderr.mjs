// What the servers a test file starts print on stderr, shown when a test fails.
//
// A request whose step fails on the server is a 500 that names no cause: the
// cause, a CUDA error on the GPU for one, goes to the server's stderr, which
// nothing reads unless a test asserts on it. So a test file hands each server
// it spawns to keepStderr, and importing this module adds hooks to the file's
// tests: a test that fails has what each server printed on stderr while it
// ran, and whether the server has since exited, added to the message of the
// error it failed with. What a test asserts is unchanged.
//
// node:test tells a hook how its test ended from Node 22.14 (context.passed,
// and the error the test threw as context.error.cause). On an older Node a
// failed test is reported as it was, without the servers' stderr.

import { afterEach, beforeEach } from "node:test";

/** Every server kept, in the order spawned. */
const kept = [];

/** The report keeps the end of what a server printed, at most this much. */
const MOST = 4000;

/** When the running test began: how much each server had printed, and which had exited. */
let from = new Map();
let gone = new Set();

/**
 * Keeps what a spawned server prints on stderr, and how it exits. The report
 * names it by its arguments, but for --model, which every server shares, and
 * by env, the variables it was given beyond the test's own. Returns the
 * record, whose text is everything printed so far.
 */
export function keepStderr(child, env = {}) {
  const args = child.spawnargs.slice(1);
  const model = args.indexOf("--model");
  if (model >= 0) args.splice(model, 2);
  for (const [k, v] of Object.entries(env)) args.push(`${k}=${v}`);
  const server = { name: `pid ${child.pid}: ${args.join(" ")}`, text: "", exited: null };
  child.stderr.on("data", (chunk) => (server.text += chunk));
  child.on("exit", (code, signal) => {
    // A crash on Windows is an NTSTATUS, which reads as one in hex: 0xC0000005 for an access violation.
    server.exited = signal ?? (code > 0xffff ? `${code}, 0x${code.toString(16).toUpperCase()}` : code);
  });
  kept.push(server);
  return server;
}

beforeEach(() => {
  from = new Map(kept.map((s) => [s, s.text.length]));
  gone = new Set(kept.filter((s) => s.exited !== null));
});

afterEach(async (t) => {
  if (t.passed !== false) return;
  const err = t.error?.cause instanceof Error ? t.error.cause : t.error;
  if (!(err instanceof Error)) return;
  await settle();
  // Read the stack before the message changes: V8 writes it out on first read, with the message as it is then.
  const stack = err.stack;
  const message = err.message;
  const report = stderrReport();
  err.message = message + report;
  if (typeof stack === "string") err.stack = message && stack.includes(message) ? stack.replace(message, () => err.message) : stack + report;
});

/** stderr comes on a pipe of its own, so it can trail the response that failed: wait until it stops growing. */
async function settle() {
  let seen = -1;
  for (let i = 0; i < 25; i++) {
    const now = kept.reduce((n, s) => n + s.text.length, 0);
    if (now === seen) return;
    seen = now;
    await new Promise((resolve) => setTimeout(resolve, 40));
  }
}

/** What each server running when the test began, or started since, has printed since then. */
function stderrReport() {
  const lines = [];
  for (const s of kept) {
    if (gone.has(s)) continue;
    let text = s.text.slice(from.get(s) ?? 0);
    const cut = text.length - MOST;
    if (cut > 0) text = `(${cut} earlier characters left out)\n` + text.slice(cut);
    lines.push(`  ${s.name}${s.exited === null ? "" : `, which has exited (${s.exited})`}:`);
    lines.push(text.trim() ? text.trimEnd().split(/\r?\n/).map((l) => `    ${l}`).join("\n") : "    nothing");
  }
  if (lines.length === 0) return "\n\nNo server was running while this test ran.";
  return `\n\nWhat the servers printed on stderr while this test ran:\n${lines.join("\n")}`;
}
