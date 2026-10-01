// A failed test shows what the servers printed on stderr while it ran
// (tests/server-stderr.mjs), and is still reported at its own line.
//
// node --test runs a test file written here, as the suite runs one. It starts
// a stand-in for a server, a node process that says it has started and then
// echoes each line it is sent to its stderr, hands it to keepStderr, and
// fails a test after sending it a line. The failure must carry that line,
// under the stand-in's name, and nothing printed before the test began.
// Needs no engine and no GPU.
//
// node:test tells a hook how its test ended from Node 22.14 on; on an older
// Node the helper adds nothing, and this is skipped with the reason.

import assert from "node:assert/strict";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const helper = new URL("./server-stderr.mjs", import.meta.url).href;
const LINE = "CUDA: the forward pass failed on the device: an illegal memory access was encountered (cudaErrorIllegalAddress)";

const STAND_IN = `process.stderr.write("started\\n");
process.stdin.on("data", (d) => process.stderr.write(d));
`;

const FILE = `import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { keepStderr } from ${JSON.stringify(helper)};

const child = spawn(process.execPath, [fileURLToPath(new URL("./stand-in.mjs", import.meta.url))], { stdio: ["pipe", "pipe", "pipe"] });
const stderr = keepStderr(child, { STAND_IN: "1" });
const printed = async (s) => { while (!stderr.text.includes(s)) await new Promise((r) => setTimeout(r, 10)); };
before(() => printed("started"));
after(() => child.kill());

test("a request that fails", async () => {
  child.stdin.write(${JSON.stringify(LINE + "\n")});
  await printed("cudaErrorIllegalAddress");
  assert.equal(500, 200, "the status");
});

test("a request that succeeds", () => {});
`;

test("a failed test carries what the servers printed on stderr while it ran, and is reported at its own line", (t) => {
  if (!("passed" in t)) return t.skip(`node:test ${process.version} does not tell a hook how its test ended (from Node 22.14)`);
  const dir = mkdtempSync(join(tmpdir(), "vitna-stderr-"));
  try {
    writeFileSync(join(dir, "stand-in.mjs"), STAND_IN);
    writeFileSync(join(dir, "reports.test.mjs"), FILE);
    // Run as a suite of its own: not as a file of this run, which node --test tells its children through NODE_TEST_CONTEXT.
    const { NODE_TEST_CONTEXT, ...env } = process.env;
    const r = spawnSync(process.execPath, ["--test", "--test-reporter=spec", join(dir, "reports.test.mjs")], { encoding: "utf8", env, timeout: 60_000 });
    const out = r.stdout + r.stderr;
    assert.equal(r.status, 1, out);
    assert.match(out, /ℹ pass 1\b/, out);
    assert.match(out, /ℹ fail 1\b/, out);
    const report = /What the servers printed on stderr while this test ran:\s+pid \d+: .*stand-in\.mjs STAND_IN=1:\s+(.*)/.exec(out);
    assert.ok(report, `no report of the stand-in's stderr in:\n${out}`);
    assert.equal(report[1].trim(), LINE);
    assert.doesNotMatch(out, /started/, "what the stand-in printed before the test began is left out");
    assert.match(out, /test at .*reports\.test\.mjs:\d+:\d+/, "the failure is reported at the test's own line");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
