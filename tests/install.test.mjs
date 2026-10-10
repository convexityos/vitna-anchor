// Gate A12's installer: the catalogue of models it may install, the plan the
// engine makes for a machine (vitna-anchor plan), its check of the files
// fetched (vitna-anchor verify), and the two scripts that use them.
//
// The plans are made for machines described in JSON, as `vitna-anchor
// hardware` prints one, so every case runs on any machine; verify runs over
// files these tests write, against a catalogue of their own. Needs a built
// engine for those; skipped with the reason otherwise, unless
// VITNA_REQUIRE_ENGINE=1 (as in CI), where it fails. The scripts' runs, end to
// end on fresh machines, are .github/workflows/install.yml's.

import assert from "node:assert/strict";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { CATALOG, SOURCE, catalogSource, resolveCatalog } from "../engine/tools/gen_catalog.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const ENGINE = { skip: process.env.VITNA_REQUIRE_ENGINE === "1" ? false : engine ? false : "no built engine found" };

const GIB = 2 ** 30;

function run(args) {
  const r = spawnSync(engine, args, { encoding: "utf8", timeout: 60_000 });
  return { status: r.status, stdout: r.stdout, stderr: r.stderr };
}

/** A plan's lines, by their first field; file lines in order. */
function readPlan(text) {
  const plan = { files: [] };
  for (const line of text.trim().split("\n")) {
    const [kind, ...rest] = line.split("\t");
    if (kind === "file") plan.files.push({ url: rest[0], path: rest[1], size: Number(rest[2]), sha256: rest[3] });
    else if (kind === "serve") plan.serve = rest;
    else plan[kind] = rest[0];
  }
  return plan;
}

const scratch = mkdtempSync(join(tmpdir(), "vitna-install-test-"));
process.on("exit", () => rmSync(scratch, { recursive: true, force: true }));

/** The plan for a machine described as `hardware` prints one, models going to an empty directory. */
function planFor(machine, extra = []) {
  const file = join(scratch, `machine-${Math.random().toString(36).slice(2)}.json`);
  writeFileSync(file, JSON.stringify({ os: "linux", arch: "x86_64", cpu_threads: 8, matvec: "avx2+fma", gpus: [], ...machine }));
  return run(["plan", "--dir", join(scratch, "models"), "--hardware", file, ...extra]);
}

const RTX3070 = { name: "NVIDIA GeForce RTX 3070", memory_bytes: 8589410304, free_bytes: 7442792448, compute: "8.6" };

test("the catalogue built into the engine is install/catalog.json with its pins", () => {
  assert.equal(readFileSync(SOURCE, "utf8"), catalogSource(), "run node engine/tools/gen_catalog.mjs");
  const resolved = resolveCatalog();
  assert.deepEqual(resolved.models.map((m) => m.id), ["qwen3-30b-a3b", "smollm2-360m-instruct"]);
  for (const m of resolved.models) {
    for (const f of m.files) {
      // Every file at a revision, never a branch, with a size and a SHA-256.
      assert.match(f.url, /^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\//, f.url);
      assert.ok(f.size > 0 && /^[0-9a-f]{64}$/.test(f.sha256), f.path);
      assert.ok(f.path.startsWith(`${m.id}/`), f.path);
    }
  }
  const catalog = JSON.parse(readFileSync(CATALOG, "utf8"));
  assert.equal(catalog.models.length, resolved.models.length);
});

test("a machine with an NVIDIA GPU of 8 GB and 64 GB of memory gets Qwen3-30B-A3B with a context of 32K, served on the GPU", ENGINE, () => {
  const r = planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] });
  assert.equal(r.status, 0, r.stderr);
  const plan = readPlan(r.stdout);
  assert.equal(plan.model, "qwen3-30b-a3b");
  assert.match(plan.why, /^This machine has an NVIDIA GeForce RTX 3070 with 8\.0 GiB, 64 GiB of memory and 200 GiB free where the models go\.$/);
  assert.deepEqual(plan.files.map((f) => f.path), ["qwen3-30b-a3b/config.json", "qwen3-30b-a3b/tokenizer.json", "qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf"]);
  assert.equal(Number(plan.fetch), plan.files.reduce((s, f) => s + f.size, 0), "nothing there yet, so all of it");
  const dir = join(scratch, "models").replace(/\\/g, "/");
  assert.deepEqual(
    plan.serve.map((a) => a.replace(/\\/g, "/")),
    ["--model", `${dir}/qwen3-30b-a3b`, "--model-id", "qwen3-30b-a3b", "--ctx", "32768", "--weights", `${dir}/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf`, "--device", "cuda"],
  );
});

test("less memory, a smaller context; less than holds the whole file, its experts read from the drive; less than that, the small model, and the plan says why", ENGINE, () => {
  const thirtyTwo = readPlan(planFor({ memory_bytes: 31.9 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] }).stdout);
  assert.equal(thirtyTwo.model, "qwen3-30b-a3b");
  assert.equal(thirtyTwo.serve[thirtyTwo.serve.indexOf("--ctx") + 1], "16384");
  assert.equal(thirtyTwo.serve.includes("--expert-cache"), false, "30 GiB and more holds the whole file");
  assert.doesNotMatch(thirtyTwo.why, /read from the drive/);

  // A machine sold with 16 GB, as Windows counts it once the firmware has kept its share.
  const sixteen = readPlan(planFor({ memory_bytes: 15.9 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] }).stdout);
  assert.equal(sixteen.model, "qwen3-30b-a3b");
  assert.equal(sixteen.serve[sixteen.serve.indexOf("--ctx") + 1], "16384");
  assert.equal(sixteen.serve[sixteen.serve.indexOf("--expert-cache") + 1], "4096");
  assert.equal(sixteen.serve[sixteen.serve.indexOf("--device") + 1], "cuda");
  assert.match(
    sixteen.why,
    /^This machine has an NVIDIA GeForce RTX 3070 with 8\.0 GiB, 16 GiB of memory and 200 GiB free where the models go\. Its experts are read from the drive as they are wanted, into a cache of 4096 MiB in memory, rather than the whole file held there\.$/,
  );
  const below = readPlan(planFor({ memory_bytes: 29.9 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] }).stdout);
  assert.equal(below.serve[below.serve.indexOf("--expert-cache") + 1], "4096", "just under 30 GiB reads from the drive");

  const twelve = readPlan(planFor({ memory_bytes: 12 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] }).stdout);
  assert.equal(twelve.model, "smollm2-360m-instruct");
  assert.match(twelve.why, /^Qwen3-30B-A3B at Q4_K_M needs an NVIDIA GPU with 7\.5 GiB and 14 GiB of memory, and it has 12 GiB of memory\. /);
  assert.equal(twelve.serve.includes("--device"), false, "the small model runs on the CPU");
  assert.equal(twelve.serve.includes("--expert-cache"), false, "the small model has no experts");
  assert.equal(twelve.serve[twelve.serve.indexOf("--ctx") + 1], "4096");
  // Just short of what it needs is said with its tenth, so it never reads as having enough.
  const short = readPlan(planFor({ memory_bytes: 13.9 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] }).stdout);
  assert.equal(short.model, "smollm2-360m-instruct");
  assert.match(short.why, /and 14 GiB of memory, and it has 13\.9 GiB of memory\. /);

  const smallGpu = readPlan(
    planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [{ ...RTX3070, name: "NVIDIA GeForce GTX 1650", memory_bytes: 4 * GIB }] }).stdout,
  );
  assert.equal(smallGpu.model, "smollm2-360m-instruct");
  assert.match(smallGpu.why, /its largest NVIDIA GPU, NVIDIA GeForce GTX 1650, has 4\.0 GiB/);

  const noGpu = readPlan(planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true }).stdout);
  assert.match(noGpu.why, /and no NVIDIA GPU was found\./);
  const cpuBuild = readPlan(planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB, cuda_built: false, gpus: [RTX3070] }).stdout);
  assert.match(cpuBuild.why, /and this engine was built without the CUDA path\./);

  const smallDisk = readPlan(planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 10 * GIB, cuda_built: true, gpus: [RTX3070] }).stdout);
  assert.equal(smallDisk.model, "smollm2-360m-instruct");
  assert.match(smallDisk.why, /it has 10\.0 GiB free where the models go, and the files need 17\.3 GiB more besides 1 GiB to spare/);
});

test("a machine nothing fits is refused, as is a model asked for that does not fit or does not exist", ENGINE, () => {
  const none = planFor({ memory_bytes: 2 * GIB, disk_free_bytes: 200 * GIB, cuda_built: false });
  assert.equal(none.status, 1);
  assert.equal(none.stdout, "");
  assert.match(none.stderr, /^no model in the catalogue fits this machine: .*SmolLM2-360M-Instruct needs 3 GiB of memory, and it has 2 GiB of memory/);
  const asked = planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB, cuda_built: false }, ["--model", "qwen3-30b-a3b"]);
  assert.equal(asked.status, 1);
  assert.match(asked.stderr, /Qwen3-30B-A3B at Q4_K_M needs an NVIDIA GPU with 7\.5 GiB and 14 GiB of memory, and this engine was built without the CUDA path/);
  const smaller = readPlan(planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB, cuda_built: true, gpus: [RTX3070] }, ["--model", "smollm2-360m-instruct"]).stdout);
  assert.equal(smaller.model, "smollm2-360m-instruct");
  const unknown = planFor({ memory_bytes: 64 * GIB, disk_free_bytes: 200 * GIB }, ["--model", "nope"]);
  assert.equal(unknown.status, 1);
  assert.match(unknown.stderr, /the catalogue has no model nope/);
});

test("what is already fetched is not fetched again, and a file longer than it should be is fetched anew", ENGINE, () => {
  // A catalogue of its own, so the files are small; verify and plan read it as they read the built-in one.
  // The folder's name has a letter of Windows' ANSI code page and one outside it,
  // as a user's name often has, which an engine reading paths in that code page
  // could not open: on Windows the engine is a UTF-8 program (engine/windows/utf8.manifest).
  const dir = join(scratch, `fetched-${String.fromCharCode(0xe9, 0x416)}`);
  mkdirSync(join(dir, "tiny"), { recursive: true });
  const content = { "a.bin": Buffer.from("alpha alpha alpha"), "b.bin": Buffer.from("bravo") };
  const sha = (b) => createHash("sha256").update(b).digest("hex");
  const catalog = {
    models: [
      {
        id: "tiny",
        title: "A tiny model",
        note: "For tests.",
        needs: { memory_gib: 0 },
        ctx: [{ memory_gib: 0, ctx: 512 }],
        files: Object.entries(content).map(([name, b]) => ({ url: `https://example.invalid/${name}`, path: `tiny/${name}`, size: b.length, sha256: sha(b) })),
      },
    ],
  };
  const catalogFile = join(scratch, "catalog.json");
  writeFileSync(catalogFile, JSON.stringify(catalog));
  const machine = join(scratch, "tiny-machine.json");
  writeFileSync(machine, JSON.stringify({ memory_bytes: 8 * GIB, disk_free_bytes: 100 * GIB }));
  const verify = () => run(["verify", "--dir", dir, "--model", "tiny", "--catalog", catalogFile]);
  const plan = () => readPlan(run(["plan", "--dir", dir, "--catalog", catalogFile, "--hardware", machine]).stdout);

  let v = verify();
  assert.equal(v.status, 1);
  assert.equal(v.stdout, "bad\ttiny/a.bin\tmissing\nbad\ttiny/b.bin\tmissing\n");
  assert.equal(plan().fetch, "22");

  writeFileSync(join(dir, "tiny", "a.bin"), content["a.bin"].subarray(0, 5));
  writeFileSync(join(dir, "tiny", "b.bin"), Buffer.from("bravo, and more"));
  v = verify();
  assert.equal(v.stdout, "bad\ttiny/a.bin\tshort: 5 of 17 bytes\nbad\ttiny/b.bin\tlong: 15 bytes, pinned 5\n");
  // 12 bytes of a.bin to resume, and all 5 of b.bin, which is fetched anew.
  assert.equal(plan().fetch, "17");

  writeFileSync(join(dir, "tiny", "a.bin"), Buffer.from("alpha alpha ALPHA"));
  writeFileSync(join(dir, "tiny", "b.bin"), content["b.bin"]);
  v = verify();
  assert.equal(v.status, 1);
  assert.match(v.stdout, /^bad\ttiny\/a\.bin\tsha256 [0-9a-f]{64}, pinned [0-9a-f]{64}\nok\ttiny\/b\.bin\n$/);

  writeFileSync(join(dir, "tiny", "a.bin"), content["a.bin"]);
  v = verify();
  assert.equal(v.status, 0, v.stdout);
  assert.equal(v.stdout, "ok\ttiny/a.bin\nok\ttiny/b.bin\n");
  const done = plan();
  assert.equal(done.fetch, "0");
  // And the folder comes back whole in the arguments to serve it with, which the scripts write out.
  assert.equal(done.serve[done.serve.indexOf("--model") + 1].replace(/\\/g, "/"), join(dir, "tiny").replace(/\\/g, "/"));
});

test("the machine as the engine sees it", ENGINE, () => {
  const r = run(["hardware", "--dir", scratch]);
  assert.equal(r.status, 0, r.stderr);
  const hw = JSON.parse(r.stdout);
  assert.ok(["windows", "linux", "macos"].includes(hw.os));
  assert.ok(hw.cpu_threads >= 1);
  assert.ok(hw.memory_bytes > GIB, `${hw.memory_bytes} bytes of memory`);
  assert.ok(hw.disk_free_bytes > 0);
  assert.equal(typeof hw.cuda_built, "boolean");
  assert.ok(Array.isArray(hw.gpus));
});

test("the installer scripts are templates a release fills in, in plain ASCII, and parse", () => {
  const sh = readFileSync(here("../install/install.sh"), "utf8");
  const ps1 = readFileSync(here("../install/install.ps1"), "utf8");
  // Printable ASCII; LF line endings for the shell, and for PowerShell the CRLF .gitattributes checks it out with.
  assert.equal(/[^\x09\x0a\x20-\x7e]/.test(sh), false, "install.sh is printable ASCII with LF line endings");
  assert.equal(/[^\x09\x0a\x0d\x20-\x7e]/.test(ps1), false, "install.ps1 is printable ASCII");
  for (const [name, text] of [["install.sh", sh], ["install.ps1", ps1]]) assert.ok(text.includes("@VERSION@"), name);
  assert.ok(sh.includes("@SHA256_LINUX_X86_64@"));
  assert.ok(ps1.includes("@SHA256_WINDOWS_X64@"));
  // Everything in install.sh runs from main, at the end, so a script cut short in its download runs nothing.
  assert.match(sh, /\nmain "\$@"\n$/);
  const shell = ["/bin/sh", "/usr/bin/sh", "C:/Program Files/Git/usr/bin/sh.exe"].find((p) => existsSync(p));
  if (shell) {
    const r = spawnSync(shell, ["-n", here("../install/install.sh")], { encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr);
  }
  if (process.platform === "win32") {
    const parse = `$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile('${here("../install/install.ps1")}', [ref]$null, [ref]$e); $e.Count`;
    const r = spawnSync("powershell.exe", ["-NoProfile", "-Command", parse], { encoding: "utf8" });
    assert.equal(r.stdout.trim(), "0", r.stderr);
  }
});
