// Gate A9: Qwen3-30B-A3B, against its reference.
//
// The fixture is recorded by reference/record_qwen3.py, a layer at a time,
// in the fields gate A5's has, so reference/compare.mjs holds the engine to
// it as it holds the engine to OLMoE's. The first tests need nothing but the
// repository: the fixture is the one its pin, its inputs and its recorders
// describe, its routing is complete and consistent, its greedy tokens were
// confirmed as the reference's own, and the comparison accepts the fixture's
// own logits and routing and rejects wrong ones at this vocabulary.
//
// The rest run the engine. Its tokenizer needs only the model's
// tokenizer.json (node scripts/fetch-model.mjs qwen3-30b-a3b --only
// tokenizer.json, as CI fetches it); the others need the 61.1 GB of weights.
// Without them these are skipped, with the reason, unless
// VITNA_REQUIRE_QWEN3=1, or for the tokenizer VITNA_REQUIRE_QWEN3_TOKENIZER=1
// (as in CI), where they fail. They run on the CPU, or with
// VITNA_DEVICE=cuda on the GPU.

import assert from "node:assert/strict";
import test from "node:test";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { LOGIT_ATOL, ROUTE_ATOL, compareGreedy, comparePrefill, compareRouting, compareTokenization, loadFixture } from "../reference/compare.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const REF = here("../reference/qwen3-30b-a3b/");
const FIXTURE = join(REF, "fixture.json");
const pin = JSON.parse(readFileSync(join(REF, "model.json"), "utf8"));
const inputs = JSON.parse(readFileSync(join(REF, "prompts.json"), "utf8"));
const recorded = existsSync(FIXTURE);
const fixture = recorded ? loadFixture(FIXTURE) : null;
const RECORDED = { skip: !recorded && "reference/qwen3-30b-a3b/fixture.json is not recorded yet" };

const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");

test("the pin is Qwen's own checkpoint, every file of it", () => {
  assert.equal(pin.repo, "Qwen/Qwen3-30B-A3B");
  assert.equal(pin.license, "apache-2.0");
  const shards = pin.files.map((f) => f.path).filter((p) => p.endsWith(".safetensors"));
  assert.deepEqual(shards, Array.from({ length: 16 }, (_, i) => `model-${String(i + 1).padStart(5, "0")}-of-00016.safetensors`));
  for (const f of ["config.json", "tokenizer.json", "model.safetensors.index.json"]) assert.ok(pin.files.some((x) => x.path === f), f);
  for (const f of pin.files) assert.match(f.sha256, /^[0-9a-f]{64}$/, f.path);
});

test("the inputs are gate A5's prompts and corpus, and the cases for this model's tokenizer after them", () => {
  const a5 = JSON.parse(readFileSync(here("../reference/olmoe-1b-7b/prompts.json"), "utf8"));
  assert.deepEqual(inputs.prompts, a5.prompts);
  assert.equal(inputs.greedy_steps, a5.greedy_steps);
  assert.deepEqual(inputs.corpus.slice(0, a5.corpus.length), a5.corpus);
  assert.ok(inputs.corpus.length > a5.corpus.length);
});

test("the fixture is the one its pin, its inputs and its recorders describe", RECORDED, () => {
  assert.equal(fixture.format, 1);
  assert.equal(fixture.model.repo, pin.repo);
  assert.equal(fixture.model.revision, pin.revision);
  assert.equal(fixture.inputs.file, "reference/qwen3-30b-a3b/prompts.json");
  assert.equal(fixture.inputs.sha256_lf, sha256Lf(join(REF, "prompts.json")), "prompts.json changed: record the fixture again");
  const scripts = { recorder: "record_qwen3.py", helpers: "record.py", moe_helpers: "record_moe.py", gguf_helpers: "record_gguf.py" };
  for (const [key, file] of Object.entries(scripts)) {
    const entry = key === "recorder" ? fixture.recorder : fixture.recorder[key];
    assert.equal(entry.script, `reference/${file}`);
    assert.equal(entry.sha256_lf, sha256Lf(here(`../reference/${file}`)), `${file} changed: record the fixture again`);
  }
  assert.deepEqual(fixture.software, {
    torch: "2.14.0+cpu",
    transformers: "5.17.0",
    tokenizers: "0.23.2",
    safetensors: "0.8.0",
    numpy: "2.5.3",
    python: fixture.software.python,
  });
  const s = fixture.settings;
  assert.deepEqual([s.dtype, s.attn_implementation, s.experts_implementation, s.threads], ["float32", "eager", "eager", 1]);
  assert.equal(s.greedy, "proposed, then each confirmed as the reference's own choice at its position");
  assert.ok(Number.isInteger(s.passes) && s.passes >= 1 && Number.isInteger(s.proposals_replaced));
  assert.deepEqual(
    [fixture.model.layers, fixture.model.experts, fixture.model.experts_per_token, fixture.model.renormalized, s.router_topk],
    [48, 128, 8, true, 9],
  );

  assert.deepEqual(fixture.prompts.map((p) => p.text), inputs.prompts.map((p) => p.text));
  assert.deepEqual(fixture.corpus.map((c) => c.text), inputs.corpus);
  // transformers runs the model's own tokenizer.json, so the two sets of ids agree on every string.
  const disagree = fixture.corpus.filter((c) => JSON.stringify(c.ids) !== JSON.stringify(c.tokenizer_json_ids)).map((c) => c.text);
  assert.deepEqual(disagree, fixture.tokenizer_disagreements);
  assert.deepEqual(disagree, []);

  for (const p of fixture.prompts) {
    assert.equal(p.positions.length, p.ids.length, p.id);
    assert.equal(p.lastLogits.length, fixture.model.vocab_size, p.id);
    assert.equal(p.greedy.length, inputs.greedy_steps, p.id);
    assert.deepEqual(p.greedy.map((g) => g.id), p.greedy_ids, p.id);
    const last = p.positions.at(-1);
    for (const [id, v] of last.top) assert.equal(p.lastLogits[id], v, `${p.id} id ${id}`);
    assert.equal(p.greedy[0].id, last.top[0][0], p.id);
    // Each greedy token is the largest logit at its step, the lowest id first between equals.
    for (const [s, g] of p.greedy.entries()) {
      const [best, value] = g.top[0];
      assert.equal(g.id, best, `${p.id} step ${s}`);
      assert.ok(g.top.every(([id, v]) => v < value || (v === value && id >= best)), `${p.id} step ${s}`);
    }
  }
});

test("the tokenizer cases this model adds tokenize as its split pattern and added tokens say", RECORDED, () => {
  const ids = (text) => fixture.corpus.find((c) => c.text === text).tokenizer_json_ids;
  // Each number character alone.
  assert.equal(ids("12345").length, 5);
  // Its chat and reasoning markers are added tokens.
  const chat = ids("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
  assert.equal(chat[0], 151644);
  assert.ok(chat.includes(151645));
  const think = ids("Hello<think>reasoning</think>answer");
  assert.ok(think.includes(151667) && think.includes(151668));
});

function* routes(p) {
  for (const [t, r] of p.routing.entries()) yield [`${p.id} position ${t}`, r];
  for (const [s, r] of p.greedy_routing.entries()) yield [`${p.id} greedy ${s}`, r];
}

test("the routing is complete and consistent", RECORDED, () => {
  const { layers, experts, experts_per_token: k } = fixture.model;
  let decisions = 0;
  for (const p of fixture.prompts) {
    assert.equal(p.routing.length, p.ids.length, p.id);
    assert.equal(p.greedy_routing.length, p.greedy.length - 1, p.id);
    for (const [where, r] of routes(p)) {
      assert.equal(r.experts.length, layers, where);
      for (let l = 0; l < layers; l++) {
        const ids = r.experts[l];
        const logits = r.logits[l];
        assert.equal(ids.length, k + 1, where);
        assert.equal(new Set(ids).size, k + 1, `${where} layer ${l}: an expert twice`);
        assert.ok(ids.every((e) => Number.isInteger(e) && e >= 0 && e < experts), `${where} layer ${l}`);
        for (let i = 1; i < k; i++) assert.ok(logits[i] <= logits[i - 1], `${where} layer ${l}: out of order`);
        assert.ok(logits[k] <= logits[k - 1], `${where} layer ${l}: the expert left out scored higher`);
        // Each chosen expert's share of the softmax over all 128; Qwen3
        // then divides each by their sum, so the 8 shares sum to less than 1.
        const sum = logits.slice(0, k).reduce((a, v) => a + Math.exp(v - r.lse[l]), 0);
        assert.ok(sum > 0 && sum < 1, `${where} layer ${l}: shares sum to ${sum}`);
        decisions++;
      }
    }
  }
  const positions = fixture.prompts.reduce((a, p) => a + p.ids.length + p.greedy.length - 1, 0);
  assert.equal(decisions, positions * layers);
});

test("the comparison accepts the reference's own logits and rejects a wrong answer", RECORDED, () => {
  const V = fixture.model.vocab_size;
  for (const p of fixture.prompts) {
    const one = { ...p, ids: [p.ids.at(-1)], positions: [p.positions.at(-1)] };
    const exact = comparePrefill(fixture, one, p.lastLogits.slice());
    assert.deepEqual(exact.failures, [], p.id);
    assert.ok(exact.worst.logit === 0 && exact.worst.lse < 1e-9, p.id);
    assert.deepEqual(comparePrefill(fixture, one, p.lastLogits.map((v) => v + 0.4 * LOGIT_ATOL)).failures, [], p.id);
    const probe = p.lastLogits.slice();
    probe[fixture.probe_ids[7]] += 2 * LOGIT_ATOL;
    assert.ok(comparePrefill(fixture, one, probe).failures.length > 0, `${p.id}: a wrong probe logit passed`);
    const shifted = p.lastLogits.map((v) => v + 2 * LOGIT_ATOL);
    assert.ok(comparePrefill(fixture, one, shifted).failures.some((f) => f.includes("logsumexp")), p.id);
  }
  for (const p of fixture.prompts) {
    const steps = new Float32Array(p.greedy.length * V).fill(-1e4);
    p.greedy.forEach((g, s) => g.top.forEach(([id, v]) => (steps[s * V + id] = v)));
    assert.deepEqual(compareGreedy(fixture, p, p.greedy_ids, steps).failures, [], p.id);
    const changed = p.greedy_ids.slice();
    changed[5] = changed[5] === 0 ? 1 : 0;
    assert.ok(compareGreedy(fixture, p, changed, steps).failures.length > 0, p.id);
  }
});

// ---------------------------------------------------------------------------
// The engine against the fixture.

const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const modelDir = process.env.ANCHOR_QWEN3_MODEL_DIR || here("../models/qwen3-30b-a3b/");
const required = (v) => process.env[v] === "1";
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const GPU = DEVICE === "cuda";
// On a GPU the experts are copied from memory the device has locked, and
// the whole 61.1 GB checkpoint is more than most machines can lock, so they
// come through a cache in memory read from the drive (--expert-cache), of
// VITNA_QWEN3_EXPERT_CACHE MiB, 16 GiB unless it says otherwise.
const RUN = ["--ctx", "512", ...(DEVICE ? ["--device", DEVICE] : []),
  ...(GPU ? ["--expert-cache", process.env.VITNA_QWEN3_EXPERT_CACHE || "16384"] : [])];

function missing(files, fetch) {
  if (!recorded) return "reference/qwen3-30b-a3b/fixture.json is not recorded yet";
  if (!engine) return "no built engine found";
  const absent = files.find((f) => !existsSync(join(modelDir, f)));
  return absent ? `no ${absent} in ${modelDir}; run ${fetch}` : false;
}
const TOKENIZER = {
  skip: (required("VITNA_REQUIRE_QWEN3") || required("VITNA_REQUIRE_QWEN3_TOKENIZER")) && recorded ? false
    : missing(["tokenizer.json"], "node scripts/fetch-model.mjs qwen3-30b-a3b --only tokenizer.json"),
};
// The model takes most of the machine's memory, so these run one engine at
// a time, and on a GPU only when asked for, with this file alone.
const MODEL = {
  skip: required("VITNA_REQUIRE_QWEN3") && recorded ? false
    : GPU ? "on a GPU these run only with VITNA_REQUIRE_QWEN3=1, and this file alone"
    : missing(pin.files.map((f) => f.path), "node scripts/fetch-model.mjs qwen3-30b-a3b"),
};

function runEngine(args, input = "") {
  assert.ok(engine, "no built engine found");
  return new Promise((resolve, reject) => {
    const child = spawn(engine, args);
    let out = "";
    let err = "";
    child.stdout.setEncoding("utf8").on("data", (d) => (out += d));
    child.stderr.setEncoding("utf8").on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) => (code === 0 ? resolve(out) : reject(new Error(`${engine} ${args.join(" ")} exited ${code}: ${err}`))));
    child.stdin.end(input);
  });
}
const readF32 = (path) => {
  const b = readFileSync(path);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
};
const readI32 = (path) => {
  const b = readFileSync(path);
  return new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
};

function writePins(path, entries) {
  const { layers, experts_per_token: k } = fixture.model;
  const buf = Buffer.alloc(entries.length * layers * k * 4);
  entries.forEach((r, t) => r.experts.forEach((ids, l) => ids.slice(0, k).forEach((e, i) => buf.writeInt32LE(e, ((t * layers + l) * k + i) * 4))));
  writeFileSync(path, buf);
}

function firstDeparture(entries, chosen) {
  const { layers, experts_per_token: k } = fixture.model;
  for (let t = 0; t < entries.length; t++) {
    for (let l = 0; l < layers; l++) {
      const mine = [...chosen.slice((t * layers + l) * k, (t * layers + l + 1) * k)].sort((a, b) => a - b).join();
      if (mine !== entries[t].experts[l].slice(0, k).sort((a, b) => a - b).join()) return [t, l];
    }
  }
  return null;
}

async function eachPrompt(prefix, run) {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  const at = (p) => (s) => join(dir, `${p.id}.${s}`);
  try {
    const results = [];
    for (const p of fixture.prompts) results.push(await run(p, at(p)));
    return results;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("the engine's tokenizer gives the ids of the model's own tokenizer.json", TOKENIZER, async (t) => {
  const cases = [...fixture.corpus, ...fixture.prompts];
  const out = await runEngine(["tokenize", "--model", modelDir], cases.map((c) => JSON.stringify(c.text)).join("\n") + "\n");
  const lines = out.trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, cases.length);
  const byText = new Map(cases.map((c, i) => [c.text, lines[i]]));
  assert.deepEqual(compareTokenization(fixture, (text) => byText.get(text)), []);
  t.diagnostic(`${cases.length} of ${cases.length} strings tokenized exactly, Qwen's split pattern and added tokens included`);
});

test("with its routing pinned, the engine's logits match the reference within the tolerance, and its router chooses as the rule allows", MODEL, async (t) => {
  const results = await eachPrompt("vitna-qwen3-logits-", async (p, f) => {
    writePins(f("pin"), p.routing);
    await runEngine(["logits", "--model", modelDir, "--ids", p.ids.join(","), "--out", f("logits"), ...RUN,
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
  t.diagnostic(`on ${DEVICE || "cpu"}, largest |engine - reference|: logits ${worst("prefill", "logit").toExponential(2)}, logsumexp ${worst("prefill", "lse").toExponential(2)}, router logits ${worst("routing", "logit").toExponential(2)}; tolerances ${LOGIT_ATOL} and ${ROUTE_ATOL}`);
  t.diagnostic(departed.length ? `the router first chose other experts than the reference at ${departed.join("; ")}` : `the router chose the reference's experts in all ${decisions} decisions`);
});

test("with its routing pinned, the engine's greedy decoding matches the reference token for token", MODEL, async (t) => {
  const results = await eachPrompt("vitna-qwen3-greedy-", async (p, f) => {
    const entries = [...p.routing, ...p.greedy_routing];
    writePins(f("pin"), entries);
    const out = await runEngine(["generate", "--model", modelDir, "--ids", p.ids.join(","), "--max-new", String(p.greedy_ids.length), "--greedy",
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
  t.diagnostic(`on ${DEVICE || "cpu"}, ${tokens} of ${tokens} greedy tokens equal; largest step logit difference ${Math.max(...results.map((r) => r.greedy.worst.logit)).toExponential(2)}, router logits ${Math.max(...results.map((r) => r.routing.worst.logit)).toExponential(2)}`);
});

test("unpinned, the engine decodes the reference's tokens until its routing departs at a near-tie, if it ever does", MODEL, async (t) => {
  const { layers, experts: E, experts_per_token: K } = fixture.model;
  const results = await eachPrompt("vitna-qwen3-free-", async (p, f) => {
    const out = await runEngine(["generate", "--model", modelDir, "--ids", p.ids.join(","), "--max-new", String(p.greedy_ids.length), "--greedy",
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
    const V = fixture.model.vocab_size;
    const result = compareGreedy(fixture, { ...p, greedy: p.greedy.slice(0, steps), greedy_ids: p.greedy_ids.slice(0, steps) }, ids.slice(0, steps), logits.subarray(0, steps * V));
    assert.deepEqual(result.failures.slice(0, 10), [], p.id);
    if (steps === p.greedy.length) assert.deepEqual(ids, p.greedy_ids, p.id);
  }
  t.diagnostic(`on ${DEVICE || "cpu"}, ` + (notes.length ? notes.join("; ") : `its routing was the reference's throughout, and all ${fixture.prompts.length * fixture.prompts[0].greedy.length} greedy tokens equal`));
});
