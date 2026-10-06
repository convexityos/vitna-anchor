// Gate A9 from Qwen's own GGUF files: Q8_0 and Q4_K_M, against their reference.
//
// Each fixture in reference/qwen3-30b-a3b-gguf/ is recorded by
// reference/record_qwen3.py --file, a layer at a time, with every weight the
// file's, widened to float32 by gguf-py, as gate A7's are for OLMoE. The
// first tests need nothing but the repository: each fixture is the one its
// pins, its inputs and its recorders describe. The rest run the engine with
// --weights <file.gguf> and compare it with the fixture as
// tests/reference-qwen3.test.mjs compares the BF16 model, and the widened
// weights by digest.
//
// They need the engine, gate A9's model directory (for config.json and the
// tokenizer) and the GGUF files (node scripts/fetch-model.mjs
// qwen3-30b-a3b-gguf), and run only with VITNA_REQUIRE_QWEN3_GGUF=1,
// where a missing file fails them, since a file takes much of a machine's
// memory and they most of an hour. They run on the CPU, or with
// VITNA_DEVICE=cuda on the GPU, where the digests are of the GPU's
// widening. The prompts run one after another.

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
const REF = here("../reference/qwen3-30b-a3b-gguf/");
const A9 = here("../reference/qwen3-30b-a3b/");
const pin = JSON.parse(readFileSync(join(REF, "model.json"), "utf8"));
const a9pin = JSON.parse(readFileSync(join(A9, "model.json"), "utf8"));
const inputs = JSON.parse(readFileSync(join(A9, "prompts.json"), "utf8"));
const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");

const quantOf = (file) => file.replace(/\.gguf$/, "").split("-").at(-1).toLowerCase();
const FIXTURES = pin.files.map((f) => ({ file: f.path, quant: quantOf(f.path), path: join(REF, `fixture-${quantOf(f.path)}.json`) }));

test("the pin is Qwen's own GGUF files, Q8_0 and Q4_K_M", () => {
  assert.equal(pin.repo, "Qwen/Qwen3-30B-A3B-GGUF");
  assert.deepEqual(pin.files.map((f) => f.path), ["Qwen3-30B-A3B-Q8_0.gguf", "Qwen3-30B-A3B-Q4_K_M.gguf"]);
});

test("there is a fixture for every pinned GGUF file, and nothing else in the directory", () => {
  const present = readdirSync(REF).filter((n) => n.startsWith("fixture-")).sort();
  assert.deepEqual(present, FIXTURES.map((f) => `fixture-${f.quant}.json`).sort());
});

for (const { file, quant, path } of FIXTURES) {
  const recorded = existsSync(path);
  const fixture = recorded ? loadFixture(path) : null;
  const RECORDED = { skip: !recorded && `reference/qwen3-30b-a3b-gguf/fixture-${quant}.json is not recorded yet` };

  test(`${quant}: the fixture is the one its pins, its inputs and its recorders describe`, RECORDED, () => {
    assert.equal(fixture.format, 1);
    assert.equal(fixture.model.repo, a9pin.repo);
    assert.equal(fixture.model.revision, a9pin.revision);
    const entry = pin.files.find((f) => f.path === file);
    assert.deepEqual(
      [fixture.weights.repo, fixture.weights.revision, fixture.weights.file, fixture.weights.size, fixture.weights.sha256],
      [pin.repo, pin.revision, entry.path, entry.size, entry.sha256],
    );
    assert.equal(fixture.weights.widened_by, "gguf 0.19.0, gguf.quants.dequantize");
    assert.equal(fixture.inputs.file, "reference/qwen3-30b-a3b/prompts.json");
    assert.equal(fixture.inputs.sha256_lf, sha256Lf(join(A9, "prompts.json")), "prompts.json changed: record the fixture again");
    assert.equal(fixture.recorder.script, "reference/record_qwen3.py");
    assert.equal(fixture.recorder.sha256_lf, sha256Lf(here("../reference/record_qwen3.py")), "record_qwen3.py changed: record the fixture again");
    for (const [key, file] of [["helpers", "record.py"], ["moe_helpers", "record_moe.py"], ["gguf_helpers", "record_gguf.py"]]) {
      assert.equal(fixture.recorder[key].sha256_lf, sha256Lf(here(`../reference/${file}`)), `${file} changed: record the fixture again`);
    }
    assert.deepEqual(fixture.software, {
      torch: "2.14.0+cpu",
      transformers: "5.17.0",
      tokenizers: "0.23.2",
      safetensors: "0.8.0",
      numpy: "2.5.3",
      gguf: "0.19.0",
      python: fixture.software.python,
    });
    const s = fixture.settings;
    assert.deepEqual([s.dtype, s.attn_implementation, s.experts_implementation, s.threads], ["float32", "eager", "eager", 1]);
    assert.equal(s.greedy, "proposed, then each confirmed as the reference's own choice at its position");
    assert.deepEqual(fixture.prompts.map((p) => p.text), inputs.prompts.map((p) => p.text));
    for (const p of fixture.prompts) {
      assert.equal(p.positions.length, p.ids.length, p.id);
      assert.equal(p.lastLogits.length, fixture.model.vocab_size, p.id);
      assert.equal(p.greedy.length, inputs.greedy_steps, p.id);
      assert.deepEqual(p.greedy.map((g) => g.id), p.greedy_ids, p.id);
      assert.equal(p.routing.length, p.ids.length, p.id);
      assert.equal(p.greedy_routing.length, p.greedy.length - 1, p.id);
    }
    // Every tensor has a digest: per layer the two norms, the four
    // projections, the query and key norms, the router and the three stacks
    // of experts; and the embedding, the final norm and the output layer.
    assert.equal(Object.keys(fixture.weights.sha256_f32).length, 3 + fixture.model.layers * 12);
    assert.ok(Object.values(fixture.weights.sha256_f32).every((h) => /^[0-9a-f]{64}$/.test(h)));
    const a = fixture.against_bf16;
    assert.equal(a.against, "reference/qwen3-30b-a3b/fixture.json");
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
  const modelDir = process.env.ANCHOR_QWEN3_MODEL_DIR || here("../models/qwen3-30b-a3b/");
  const ggufPath = join(process.env.ANCHOR_QWEN3_GGUF_DIR || here("../models/qwen3-30b-a3b-gguf/"), file);
  const DEVICE = process.env.VITNA_DEVICE ?? "";
  assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
  const GPU = DEVICE === "cuda";
  const onDevice = DEVICE ? ["--device", DEVICE] : [];
  const needs = !recorded ? `reference/qwen3-30b-a3b-gguf/fixture-${quant}.json is not recorded yet`
    : process.env.VITNA_REQUIRE_QWEN3_GGUF === "1" ? false
    : "these run only with VITNA_REQUIRE_QWEN3_GGUF=1: a file takes much of a machine's memory, and they most of an hour";
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
    const { layers, experts_per_token: k } = fixture.model;
    const buf = Buffer.alloc(entries.length * layers * k * 4);
    entries.forEach((r, t) => r.experts.forEach((ids, l) => ids.slice(0, k).forEach((e, i) => buf.writeInt32LE(e, ((t * layers + l) * k + i) * 4))));
    writeFileSync(p, buf);
  };
  const firstDeparture = (entries, chosen) => {
    const { layers, experts_per_token: k } = fixture.model;
    for (let t = 0; t < entries.length; t++) {
      for (let l = 0; l < layers; l++) {
        const mine = [...chosen.slice((t * layers + l) * k, (t * layers + l + 1) * k)].sort((a, b) => a - b).join();
        if (mine !== entries[t].experts[l].slice(0, k).sort((a, b) => a - b).join()) return [t, l];
      }
    }
    return null;
  };
  const eachPrompt = async (prefix, run) => {
    const dir = mkdtempSync(join(tmpdir(), prefix));
    const at = (p) => (s) => join(dir, `${p.id}.${s}`);
    try {
      const results = [];
      for (const p of fixture.prompts) results.push(await run(p, at(p)));
      return results;
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  test(`${quant}: the engine widens every tensor to the bits the reference was recorded on`, ENGINE, async (t) => {
    const mine = JSON.parse(await runEngine(["weights-sha256", ...WEIGHTS, ...onDevice]));
    const want = fixture.weights.sha256_f32;
    const differ = Object.keys(want).filter((k) => mine[k] !== want[k]);
    assert.deepEqual(Object.keys(mine).sort(), Object.keys(want).sort(), "the engine lists other tensors than the reference");
    assert.deepEqual(differ, [], `${differ.length} tensors widen to other bits`);
    t.diagnostic(`on ${where}, all ${Object.keys(want).length} tensors, ${Object.entries(fixture.weights.tensor_types).map(([k, v]) => `${v} ${k}`).join(", ")} in the file, widen to gguf-py's bits`);
  });

  test(`${quant}: with its routing pinned, the engine's logits match the reference within the tolerance, and its router chooses as the rule allows`, ENGINE, async (t) => {
    const results = await eachPrompt("vitna-qwen3-gguf-logits-", async (p, f) => {
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
    const decisions = fixture.prompts.reduce((a, p) => a + p.ids.length * fixture.model.layers, 0);
    const departed = results.filter((r) => r.departure).map((r) => `${r.p.id} position ${r.departure[0]} layer ${r.departure[1]}`);
    t.diagnostic(`on ${where}, largest |engine - reference|: logits ${worst("prefill", "logit").toExponential(2)}, logsumexp ${worst("prefill", "lse").toExponential(2)}, router logits ${worst("routing", "logit").toExponential(2)}; tolerances ${LOGIT_ATOL} and ${ROUTE_ATOL}`);
    t.diagnostic(departed.length ? `the router first chose other experts than the reference at ${departed.join("; ")}` : `the router chose the reference's experts in all ${decisions} decisions`);
  });

  test(`${quant}: with its routing pinned, the engine's greedy decoding matches the reference token for token`, ENGINE, async (t) => {
    const results = await eachPrompt("vitna-qwen3-gguf-greedy-", async (p, f) => {
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
    const { layers, experts: E, experts_per_token: K, vocab_size: V } = fixture.model;
    const results = await eachPrompt("vitna-qwen3-gguf-free-", async (p, f) => {
      const out = await runEngine(["generate", ...WEIGHTS, "--ids", p.ids.join(","), "--max-new", String(p.greedy_ids.length), "--greedy",
        ...RUN, "--logits-out", f("logits"), "--router-out", f("router"), "--experts-out", f("experts")]);
      return { p, ids: JSON.parse(out).ids, logits: readF32(f("logits")), router: readF32(f("router")), chosen: readI32(f("experts")) };
    });
    const notes = [];
    for (const { p, ids, logits, router, chosen } of results) {
      const entries = [...p.routing, ...p.greedy_routing];
      const [d, dl] = firstDeparture(entries, chosen) ?? [entries.length, 0];
      const before = compareRouting(fixture, entries.slice(0, d), router.subarray(0, d * layers * E), chosen.subarray(0, d * layers * K), `${p.id} position`);
      assert.deepEqual(before.failures.slice(0, 10), [], p.id);
      if (d < entries.length) {
        const sub = { model: { ...fixture.model, layers: dl + 1 } };
        const entry = { experts: entries[d].experts.slice(0, dl + 1), logits: entries[d].logits.slice(0, dl + 1), lse: entries[d].lse.slice(0, dl + 1) };
        const own = compareRouting(sub, [entry], router.subarray(d * layers * E, (d * layers + dl + 1) * E), chosen.subarray(d * layers * K, (d * layers + dl + 1) * K), `${p.id} position ${d}`);
        assert.deepEqual(own.failures, [], `${p.id}: the routing departed where the rule does not allow it`);
        notes.push(`${p.id} departed at position ${d} layer ${dl}`);
      }
      const steps = Math.max(0, Math.min(p.greedy.length, d - p.ids.length + 1));
      const result = compareGreedy(fixture, { ...p, greedy: p.greedy.slice(0, steps), greedy_ids: p.greedy_ids.slice(0, steps) }, ids.slice(0, steps), logits.subarray(0, steps * V));
      assert.deepEqual(result.failures.slice(0, 10), [], p.id);
      if (steps === p.greedy.length) assert.deepEqual(ids, p.greedy_ids, p.id);
    }
    t.diagnostic(`on ${where}, ` + (notes.length ? notes.join("; ") : `its routing was the reference's throughout, and all ${fixture.prompts.length * fixture.prompts[0].greedy.length} greedy tokens equal`));
  });

  // Gate A8 on this model: with --cpu-experts, the experts the device lacks
  // are shared between the CPU, in the GPU's arithmetic, and copies to it,
  // and the logits must be the GPU alone's, byte for byte. Through 1 GiB of
  // the device most of a token's experts are missing. Every token runs as a
  // step (VITNA_TEST_NO_ROWS), the prompt's too, since rows copy in what they
  // lack either way.
  test(`${quant}: on the GPU, the experts the device lacks run on the CPU, and the logits and tokens are the GPU alone's, byte for byte`, ON_GPU, async (t) => {
    const alone = { VITNA_TEST_NO_ROWS: "1" };
    const small = ["--gpu-expert-cache", "1024"];
    const configs = [["gpu", small], ["split", [...small, "--cpu-experts", "4"]]];
    const dir = mkdtempSync(join(tmpdir(), "vitna-qwen3-gguf-split-"));
    try {
      const prompts = fixture.prompts.filter((x) => ["capital", "code"].includes(x.id));
      for (const p of prompts) {
        const ids = [];
        for (const [name, extra] of configs) {
          const out = await runEngine(["generate", ...WEIGHTS, "--ids", p.ids.join(","), "--max-new", "16", "--greedy", ...RUN,
            "--logits-out", join(dir, `${p.id}.${name}.greedy`), ...extra], alone);
          ids.push(JSON.parse(out).ids);
          await runEngine(["logits", ...WEIGHTS, "--ids", p.ids.join(","), "--out", join(dir, `${p.id}.${name}.logits`), ...RUN, ...extra], alone);
        }
        assert.deepEqual(ids[1], ids[0], `${p.id}: the split chose other tokens`);
        for (const kind of ["logits", "greedy"]) {
          const a = readFileSync(join(dir, `${p.id}.gpu.${kind}`));
          const b = readFileSync(join(dir, `${p.id}.split.${kind}`));
          assert.ok(a.length > 0 && a.equals(b), `${p.id} ${kind}: the split's ${b.length} bytes are not the GPU alone's ${a.length}`);
        }
      }
      t.diagnostic(`${prompts.map((x) => x.id).join(" and ")}: every position's logits, and 16 greedy tokens with their logits, equal the GPU alone's byte for byte, through 1 GiB of the device, the CPU on 4 threads`);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}
