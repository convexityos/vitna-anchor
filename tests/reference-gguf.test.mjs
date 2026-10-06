// Gate A7: quantized weights from a GGUF file, against their reference.
//
// Each fixture in reference/olmoe-1b-7b-gguf/ is gate A5's recording of
// OLMoE-1B-7B (reference/record_moe.py, unchanged) with every weight replaced
// by a GGUF file's, widened to float32 by gguf-py (reference/record_gguf.py).
// The first tests need nothing but the repository: each fixture is the one
// its pins, its inputs and its recorders describe. The rest run the engine
// with --weights <file.gguf> and compare it with the fixture as A5's tests
// compare the BF16 model: the widened weights by digest, the logits and the
// routing with the experts pinned, the greedy tokens pinned and unpinned, and
// the experts read from the drive.
//
// They need the engine, the model directory of gate A5 (for config.json and
// the tokenizer) and the GGUF files (node scripts/fetch-model.mjs
// olmoe-1b-7b-gguf). Without them they are skipped, with the reason, unless
// VITNA_REQUIRE_GGUF=1, where they fail. They run on the CPU, or with
// VITNA_DEVICE=cuda on the GPU, for an engine built with the CUDA path, where
// the digests are of the GPU's widening, read every way its kernels read
// weights; two more run only there, as A5's do. On a GPU they run only with
// VITNA_REQUIRE_GGUF=1 and this file alone, since each engine takes what the
// device has free for its experts.

import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { LOGIT_ATOL, ROUTE_ATOL, compareGreedy, comparePrefill, compareRouting, loadFixture } from "../reference/compare.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const REF = here("../reference/olmoe-1b-7b-gguf/");
const A5 = here("../reference/olmoe-1b-7b/");
const pin = JSON.parse(readFileSync(join(REF, "model.json"), "utf8"));
const a5pin = JSON.parse(readFileSync(join(A5, "model.json"), "utf8"));
const inputs = JSON.parse(readFileSync(join(A5, "prompts.json"), "utf8"));
const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");

// One fixture for each pinned file, named for its quantization.
const quantOf = (file) => file.replace(/\.gguf$/, "").split("-").at(-1);
const FIXTURES = pin.files.map((f) => ({ file: f.path, quant: quantOf(f.path), path: join(REF, `fixture-${quantOf(f.path)}.json`) }));

test("there is a fixture for every pinned GGUF file, and nothing else in the directory", () => {
  const present = readdirSync(REF).filter((n) => n.startsWith("fixture-")).sort();
  assert.deepEqual(present, FIXTURES.map((f) => `fixture-${f.quant}.json`).sort());
});

for (const { file, quant, path } of FIXTURES) {
  const fixture = loadFixture(path);
  const VOCAB = fixture.model.vocab_size;
  const LAYERS = fixture.model.layers;
  const K = fixture.model.experts_per_token;

  test(`${quant}: the fixture is the one its pins, its inputs and its recorders describe`, () => {
    assert.equal(fixture.format, 1);
    // config.json and the tokenizer are gate A5's model's; the weights are the file's.
    assert.equal(fixture.model.repo, a5pin.repo);
    assert.equal(fixture.model.revision, a5pin.revision);
    const entry = pin.files.find((f) => f.path === file);
    assert.deepEqual(
      [fixture.weights.repo, fixture.weights.revision, fixture.weights.file, fixture.weights.size, fixture.weights.sha256],
      [pin.repo, pin.revision, entry.path, entry.size, entry.sha256],
    );
    assert.equal(fixture.weights.widened_by, "gguf 0.19.0, gguf.quants.dequantize");
    assert.equal(fixture.inputs.file, "reference/olmoe-1b-7b/prompts.json");
    assert.equal(fixture.inputs.sha256_lf, sha256Lf(join(A5, "prompts.json")), "prompts.json changed: record the fixture again");
    assert.equal(fixture.recorder.script, "reference/record_gguf.py");
    assert.equal(fixture.recorder.sha256_lf, sha256Lf(here("../reference/record_gguf.py")), "record_gguf.py changed: record the fixture again");
    assert.equal(fixture.recorder.runs.sha256_lf, sha256Lf(here("../reference/record_moe.py")), "record_moe.py changed: record the fixture again");
    assert.equal(fixture.recorder.helpers.sha256_lf, sha256Lf(here("../reference/record.py")), "record.py changed: record the fixture again");
    assert.deepEqual(fixture.software, {
      torch: "2.14.0+cpu",
      transformers: "5.17.0",
      tokenizers: "0.23.2",
      safetensors: "0.8.0",
      numpy: "2.5.3",
      gguf: "0.19.0",
      python: fixture.software.python,
    });
    assert.deepEqual([fixture.settings.dtype, fixture.settings.attn_implementation, fixture.settings.experts_implementation, fixture.settings.threads],
      ["float32", "eager", "eager", 1]);
    assert.ok(readFileSync(path).length < 4 * 1024 * 1024, "the fixture stays under 4 MB");
    assert.deepEqual(fixture.prompts.map((p) => p.text), inputs.prompts.map((p) => p.text));
    for (const p of fixture.prompts) {
      assert.equal(p.positions.length, p.ids.length, p.id);
      assert.equal(p.lastLogits.length, VOCAB, p.id);
      assert.equal(p.greedy.length, inputs.greedy_steps, p.id);
      assert.equal(p.routing.length, p.ids.length, p.id);
      assert.equal(p.greedy_routing.length, p.greedy.length - 1, p.id);
    }
    // Every tensor has a digest: per layer the norms, the four projections,
    // the two query and key norms, the router and the three stacks of experts.
    assert.equal(Object.keys(fixture.weights.sha256_f32).length, 3 + LAYERS * 12);
    assert.ok(Object.values(fixture.weights.sha256_f32).every((h) => /^[0-9a-f]{64}$/.test(h)));
    const a = fixture.against_bf16;
    assert.equal(a.kl_last_position.length, fixture.prompts.length);
    assert.ok(a.top1_positions > 0 && a.top1_agree <= a.top1_positions && a.greedy_agree_before_first_difference <= a.greedy_tokens);
  });

  // ---------------------------------------------------------------------------
  // The engine against the fixture, on the device VITNA_DEVICE names.

  const EXE = process.platform === "win32" ? ".exe" : "";
  const engine = [
    process.env.VITNA_ENGINE,
    here(`../engine/vitna-anchor${EXE}`),
    here(`../engine/build/Release/vitna-anchor${EXE}`),
    here(`../engine/build/vitna-anchor${EXE}`),
  ].filter(Boolean).find((p) => existsSync(p));
  const modelDir = process.env.ANCHOR_MOE_MODEL_DIR || here("../models/olmoe-1b-7b/");
  const ggufPath = join(process.env.ANCHOR_GGUF_DIR || here("../models/olmoe-1b-7b-gguf/"), file);
  const DEVICE = process.env.VITNA_DEVICE ?? "";
  assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
  const GPU = DEVICE === "cuda";
  const onDevice = DEVICE ? ["--device", DEVICE] : [];
  const needs = process.env.VITNA_REQUIRE_GGUF === "1" ? false
    : GPU ? "on a GPU these run only with VITNA_REQUIRE_GGUF=1, and this file alone: each engine takes what the GPU has free"
    : !engine ? "no built engine found"
    : !existsSync(join(modelDir, "config.json")) ? `no config.json in ${modelDir}; run node scripts/fetch-model.mjs olmoe-1b-7b`
    : !existsSync(ggufPath) ? `no ${ggufPath}; run node scripts/fetch-model.mjs olmoe-1b-7b-gguf`
    : false;
  const ENGINE = { skip: needs };
  const ON_GPU = { skip: needs || (!GPU && "VITNA_DEVICE is not cuda") };
  const WEIGHTS = ["--model", modelDir, "--weights", ggufPath];
  const RUN = ["--ctx", "512", ...onDevice];
  const where = DEVICE || "cpu";

  const runEngine = (args, env = {}) => new Promise((resolve, reject) => {
    const child = spawn(engine, args, { env: { ...process.env, ...env } });
    let out = "";
    let err = "";
    child.stdout.setEncoding("utf8").on("data", (d) => (out += d));
    child.stderr.setEncoding("utf8").on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) => (code === 0 ? resolve(out) : reject(new Error(`${engine} ${args.join(" ")} exited ${code}: ${err}`))));
    child.stdin.end();
  });
  const readF32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice(); };
  const readI32 = (p) => { const b = readFileSync(p); return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice(); };
  const writePins = (p, entries) => {
    const buf = Buffer.alloc(entries.length * LAYERS * K * 4);
    entries.forEach((r, t) => r.experts.forEach((ids, l) => ids.slice(0, K).forEach((e, i) => buf.writeInt32LE(e, ((t * LAYERS + l) * K + i) * 4))));
    writeFileSync(p, buf);
  };
  const firstDeparture = (entries, chosen) => {
    for (let t = 0; t < entries.length; t++) {
      for (let l = 0; l < LAYERS; l++) {
        const mine = [...chosen.slice((t * LAYERS + l) * K, (t * LAYERS + l + 1) * K)].sort((a, b) => a - b).join();
        if (mine !== entries[t].experts[l].slice(0, K).sort((a, b) => a - b).join()) return [t, l];
      }
    }
    return null;
  };
  // The CPU path computes on one thread, so the prompts run side by side, each
  // in an engine of its own over the same mapped file. On a GPU they run one
  // after another, since each engine's expert cache takes what the device has free.
  const eachPrompt = async (prefix, run) => {
    const dir = mkdtempSync(join(tmpdir(), prefix));
    const at = (p) => (s) => join(dir, `${p.id}.${s}`);
    try {
      if (!GPU) return await Promise.all(fixture.prompts.map((p) => run(p, at(p))));
      const results = [];
      for (const p of fixture.prompts) results.push(await run(p, at(p)));
      return results;
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  // On a GPU the digests are of the GPU's widening, which weights-sha256
  // --device cuda reads three ways, as the projections, a prompt's matrix
  // product and the embedding read weights, and refuses unless all three agree.
  test(`${quant}: the engine widens every tensor to the bits the reference was recorded on`, ENGINE, async (t) => {
    const mine = JSON.parse(await runEngine(["weights-sha256", ...WEIGHTS, ...onDevice]));
    const want = fixture.weights.sha256_f32;
    const differ = Object.keys(want).filter((k) => mine[k] !== want[k]);
    assert.deepEqual(Object.keys(mine).sort(), Object.keys(want).sort(), "the engine lists other tensors than the reference");
    assert.deepEqual(differ, [], `${differ.length} tensors widen to other bits`);
    t.diagnostic(`on ${where}, all ${Object.keys(want).length} tensors, ${Object.entries(fixture.weights.tensor_types).map(([k, v]) => `${v} ${k}`).join(", ")} in the file, widen to gguf-py's bits`);
  });

  test(`${quant}: with its routing pinned, the engine's logits match the reference within the tolerance, and its router chooses as the rule allows`, ENGINE, async (t) => {
    const results = await eachPrompt("vitna-gguf-logits-", async (p, f) => {
      writePins(f("pin"), p.routing);
      await runEngine(["logits", ...WEIGHTS, "--ids", p.ids.join(","), "--out", f("logits"), ...RUN,
        "--experts-in", f("pin"), "--router-out", f("router"), "--experts-out", f("experts")]);
      const chosen = readI32(f("experts"));
      return {
        p,
        prefill: comparePrefill(fixture, p, readF32(f("logits"))),
        routing: compareRouting(fixture, p.routing, readF32(f("router")), chosen, `${p.id} position`),
        departure: firstDeparture(p.routing, chosen),
      };
    });
    for (const r of results) {
      assert.deepEqual(r.prefill.failures.slice(0, 10), [], r.p.id);
      assert.deepEqual(r.routing.failures.slice(0, 10), [], r.p.id);
    }
    const worst = (k, w) => Math.max(...results.map((r) => r[k].worst[w]));
    const decisions = fixture.prompts.reduce((a, p) => a + p.ids.length * LAYERS, 0);
    const departed = results.filter((r) => r.departure).map((r) => `${r.p.id} position ${r.departure[0]} layer ${r.departure[1]}`);
    t.diagnostic(`on ${where}, largest |engine - reference|: logits ${worst("prefill", "logit").toExponential(2)}, logsumexp ${worst("prefill", "lse").toExponential(2)}, router logits ${worst("routing", "logit").toExponential(2)}; tolerances ${LOGIT_ATOL} and ${ROUTE_ATOL}`);
    t.diagnostic(departed.length ? `the router first chose other experts than the reference at ${departed.join("; ")}` : `the router chose the reference's experts in all ${decisions} decisions`);
  });

  test(`${quant}: with its routing pinned, the engine's greedy decoding matches the reference token for token`, ENGINE, async (t) => {
    const results = await eachPrompt("vitna-gguf-greedy-", async (p, f) => {
      const entries = [...p.routing, ...p.greedy_routing];
      writePins(f("pin"), entries);
      const out = await runEngine(["generate", ...WEIGHTS, "--ids", p.ids.join(","), "--max-new", String(p.greedy_ids.length), "--greedy",
        ...RUN, "--logits-out", f("logits"), "--experts-in", f("pin"), "--router-out", f("router"), "--experts-out", f("experts")]);
      const { ids } = JSON.parse(out);
      return {
        p,
        greedy: compareGreedy(fixture, p, ids, readF32(f("logits"))),
        routing: compareRouting(fixture, entries, readF32(f("router")), readI32(f("experts")), `${p.id} position`),
        tokens: ids.length,
      };
    });
    for (const r of results) {
      assert.deepEqual(r.greedy.failures.slice(0, 10), [], r.p.id);
      assert.deepEqual(r.routing.failures.slice(0, 10), [], r.p.id);
    }
    const tokens = results.reduce((a, r) => a + r.tokens, 0);
    t.diagnostic(`on ${where}, ${tokens} of ${tokens} greedy tokens equal; largest step logit difference ${Math.max(...results.map((r) => r.greedy.worst.logit)).toExponential(2)}`);
  });

  test(`${quant}: unpinned, the engine decodes the reference's tokens until its routing departs at a near-tie, if it ever does`, ENGINE, async (t) => {
    const results = await eachPrompt("vitna-gguf-free-", async (p, f) => {
      const out = await runEngine(["generate", ...WEIGHTS, "--ids", p.ids.join(","), "--max-new", String(p.greedy_ids.length), "--greedy",
        ...RUN, "--logits-out", f("logits"), "--router-out", f("router"), "--experts-out", f("experts")]);
      return { p, ids: JSON.parse(out).ids, logits: readF32(f("logits")), router: readF32(f("router")), chosen: readI32(f("experts")) };
    });
    const E = fixture.model.experts;
    const notes = [];
    for (const { p, ids, logits, router, chosen } of results) {
      const entries = [...p.routing, ...p.greedy_routing];
      const [d, dl] = firstDeparture(entries, chosen) ?? [entries.length, 0];
      const before = compareRouting(fixture, entries.slice(0, d), router.subarray(0, d * LAYERS * E), chosen.subarray(0, d * LAYERS * K), `${p.id} position`);
      assert.deepEqual(before.failures.slice(0, 10), [], p.id);
      if (d < entries.length) {
        const sub = { model: { ...fixture.model, layers: dl + 1 } };
        const entry = { experts: entries[d].experts.slice(0, dl + 1), logits: entries[d].logits.slice(0, dl + 1), lse: entries[d].lse.slice(0, dl + 1) };
        const own = compareRouting(sub, [entry], router.subarray(d * LAYERS * E, (d * LAYERS + dl + 1) * E), chosen.subarray(d * LAYERS * K, (d * LAYERS + dl + 1) * K), `${p.id} position ${d}`);
        assert.deepEqual(own.failures, [], `${p.id}: the routing departed where the rule does not allow it`);
        notes.push(`${p.id} departed at position ${d} layer ${dl}`);
      }
      const steps = Math.max(0, Math.min(p.greedy.length, d - p.ids.length + 1));
      const result = compareGreedy(fixture, { ...p, greedy: p.greedy.slice(0, steps), greedy_ids: p.greedy_ids.slice(0, steps) }, ids.slice(0, steps), logits.subarray(0, steps * VOCAB));
      assert.deepEqual(result.failures.slice(0, 10), [], p.id);
      if (steps === p.greedy.length) assert.deepEqual(ids, p.greedy_ids, p.id);
    }
    t.diagnostic(`on ${where}, ` + (notes.length ? notes.join("; ") : `its routing was the reference's throughout, and all ${fixture.prompts.length * fixture.prompts[0].greedy.length} greedy tokens equal`));
  });

  // Run capital (and code, given more prompts) under each of configs, [name,
  // extra arguments, and optionally the environment]: 16 greedy tokens with
  // their logits, and every position's logits. Every run must give the
  // first's tokens, and its logits byte for byte. On the CPU a prompt's runs
  // go side by side; on a GPU one after another.
  const sameEverywhere = async (configs, ids = ["capital"]) => {
    const prompts = fixture.prompts.filter((p) => ids.includes(p.id));
    const dir = mkdtempSync(join(tmpdir(), "vitna-gguf-same-"));
    try {
      for (const p of prompts) {
        const run = async ([name, extra, env = {}]) => {
          const out = await runEngine(["generate", ...WEIGHTS, "--ids", p.ids.join(","), "--max-new", "16", "--greedy", ...RUN,
            "--logits-out", join(dir, `${p.id}.${name}.greedy`), ...extra], env);
          await runEngine(["logits", ...WEIGHTS, "--ids", p.ids.join(","), "--out", join(dir, `${p.id}.${name}.logits`), ...RUN, ...extra], env);
          return JSON.parse(out).ids;
        };
        const tokens = [];
        if (GPU) for (const c of configs) tokens.push(await run(c));
        else tokens.push(...(await Promise.all(configs.map(run))));
        const [first] = configs;
        for (const [i, [name]] of configs.entries()) {
          assert.deepEqual(tokens[i], tokens[0], `${p.id}: ${name} chose other tokens than ${first[0]}`);
          for (const kind of ["logits", "greedy"]) {
            const a = readFileSync(join(dir, `${p.id}.${first[0]}.${kind}`));
            const b = readFileSync(join(dir, `${p.id}.${name}.${kind}`));
            assert.ok(a.length > 0 && a.equals(b), `${p.id} ${kind}: the ${name} run's ${b.length} bytes are not the ${first[0]} run's ${a.length}`);
          }
        }
      }
      return prompts.map((p) => p.id).join(" and ");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  // Gate A5's step 3 again, for a quantized file: the experts read from the
  // drive into a cache far smaller than they are must give the mapped run's
  // logits bit for bit. A quantized expert's extent is whole rows of blocks,
  // and need not start on a sector.
  test(`${quant}: read from the drive into a small cache, the experts give the mapped run's logits and tokens, bit for bit`, ENGINE, async (t) => {
    const which = await sameEverywhere([["mapped", []], ["streamed", ["--expert-cache", "128"]]]);
    t.diagnostic(`on ${where}, ${which}: every position's logits, and 16 greedy tokens with their logits, equal byte for byte, read through a cache of 128 MiB`);
  });

  // As A5's: on a GPU an expert's arithmetic does not depend on which slot
  // holds it, when it was copied there or from where. Here a layer's down
  // projections may be in another format than the next layer's, so a slot
  // holds experts of different sizes as layers come and go.
  test(`${quant}: on the GPU, an expert cache of any size, fed from either place, gives the same logits and tokens, bit for bit`, ON_GPU, async (t) => {
    const which = await sameEverywhere([
      ["default", []],
      ["small", ["--gpu-expert-cache", "256"]],
      ["both-small", ["--gpu-expert-cache", "256", "--expert-cache", "256"]],
    ], ["capital", "code"]);
    t.diagnostic(`${which}: every position's logits, and 16 greedy tokens with their logits, equal byte for byte with the GPU's expert cache as large as the device allows, at 256 MiB, and at 256 MiB fed from a cache of 256 MiB in memory`);
  });

  // Gate A8 for a quantized file: the experts the device lacks run on the
  // CPU, which widens their blocks as the GPU does (quant.c) and sums in the
  // GPU's order, so the logits are the GPU alone's, byte for byte. As in
  // A5's, every token runs as a step, the prompt's too.
  test(`${quant}: on the GPU, the experts the device lacks run on the CPU, and the logits and tokens are the GPU alone's, byte for byte`, ON_GPU, async (t) => {
    const alone = { VITNA_TEST_NO_ROWS: "1" };
    const small = ["--gpu-expert-cache", "256"];
    const which = await sameEverywhere([
      ["gpu", small, alone],
      ["split", [...small, "--cpu-experts", "4"], alone],
      ["split-streamed", [...small, "--expert-cache", "256", "--cpu-experts", "4"], alone],
    ], ["capital", "code"]);
    t.diagnostic(`${which}: every position's logits, and 16 greedy tokens with their logits, equal the GPU alone's byte for byte, with the experts the device lacked, through 256 MiB of it, run on the CPU on 4 threads, fed from the mapped file and from a cache of 256 MiB in memory`);
  });

  // As A5's: tokens run together as rows (a prompt's, and drafted tokens with
  // --speculate) give a token at a time's logits, byte for byte.
  test(`${quant}: on the GPU, tokens run together as rows give a token at a time's logits and tokens, byte for byte`, ON_GPU, async (t) => {
    const alone = { VITNA_TEST_NO_ROWS: "1" };
    const dir = mkdtempSync(join(tmpdir(), "vitna-gguf-rows-"));
    try {
      const same = (a, b, what) => {
        const x = readFileSync(join(dir, a));
        const y = readFileSync(join(dir, b));
        assert.ok(x.length > 0 && x.equals(y), `${what}: rows gave ${x.length} bytes unlike a token at a time's ${y.length}`);
      };
      const prompts = fixture.prompts.filter((p) => ["capital", "code"].includes(p.id));
      for (const p of prompts) {
        for (const [name, env] of [["rows", {}], ["alone", alone]]) {
          await runEngine(["logits", ...WEIGHTS, "--ids", p.ids.join(","), "--out", join(dir, `${p.id}.${name}.logits`), ...RUN], env);
        }
        const greedy = (name, extra, env) =>
          runEngine(["generate", ...WEIGHTS, "--ids", p.ids.join(","), "--max-new", "16", "--greedy", ...RUN,
            "--logits-out", join(dir, `${p.id}.${name}.greedy`), ...extra], env);
        const drafted = JSON.parse(await greedy("rows", ["--speculate", "4"], {})).ids;
        const stepped = JSON.parse(await greedy("alone", [], alone)).ids;
        assert.deepEqual(drafted, stepped, p.id);
        same(`${p.id}.rows.logits`, `${p.id}.alone.logits`, `${p.id}, every position's logits`);
        same(`${p.id}.rows.greedy`, `${p.id}.alone.greedy`, `${p.id}, 16 greedy tokens drafted 4 at a time`);
      }
      // The longest prompt, through a cache too small to hold a layer's experts for it at once.
      const longest = fixture.prompts.reduce((a, b) => (b.ids.length > a.ids.length ? b : a));
      for (const [name, env] of [["rows", {}], ["alone", alone]]) {
        await runEngine(["logits", ...WEIGHTS, "--ids", longest.ids.join(","), "--out", join(dir, `small.${name}.logits`), ...RUN,
          "--gpu-expert-cache", "256"], env);
      }
      same("small.rows.logits", "small.alone.logits", `${longest.id} through a cache of 256 MiB`);
      t.diagnostic(`${prompts.map((p) => p.id).join(" and ")}: every position's logits, and 16 greedy tokens drafted 4 at a time, equal a token at a time's byte for byte; ${longest.id}'s ${longest.ids.length} positions too, through a cache of 256 MiB`);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}
