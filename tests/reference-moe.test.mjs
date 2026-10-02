// Gate A5: a mixture of experts, against its reference.
//
// OLMoE-1B-7B's fixture, recorded by reference/record_moe.py as gate A1's is
// recorded by record.py, with each layer's routing added. The first tests
// need nothing but the repository: the fixture is the one its pin, its inputs
// and its recorders describe, its routing is complete and consistent, and the
// comparison in reference/compare.mjs accepts the fixture's own logits and
// routing and rejects wrong ones at this vocabulary too.
//
// The rest are steps 2 and 3: the engine runs the model, its experts mapped
// or read from the drive, and is compared with the fixture. Its tokenizer
// needs only the model's tokenizer.json (node scripts/fetch-model.mjs
// olmoe-1b-7b --only tokenizer.json, as CI fetches it); the others need the
// 13.8 GB of weights. Without them these are skipped, with the reason,
// unless VITNA_REQUIRE_MOE=1, or for the tokenizer
// VITNA_REQUIRE_MOE_TOKENIZER=1 (as in CI), where they fail. They run on the
// CPU, or with VITNA_DEVICE=cuda on the GPU (--device cuda), for an engine
// built with the CUDA path, as gate A4's do; two more run only there. On a
// GPU they need VITNA_REQUIRE_MOE=1 and this file run alone (below).

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
const REF = here("../reference/olmoe-1b-7b/");
const fixture = loadFixture(join(REF, "fixture.json"));
const pin = JSON.parse(readFileSync(join(REF, "model.json"), "utf8"));
const inputs = JSON.parse(readFileSync(join(REF, "prompts.json"), "utf8"));
const VOCAB = fixture.model.vocab_size;

const sha256Lf = (path) => createHash("sha256").update(readFileSync(path, "latin1").replaceAll("\r\n", "\n"), "latin1").digest("hex");

test("the fixture is the one its pin, its inputs and its recorders describe", () => {
  assert.equal(fixture.format, 1);
  assert.equal(fixture.model.repo, pin.repo);
  assert.equal(fixture.model.revision, pin.revision);
  assert.equal(fixture.inputs.file, "reference/olmoe-1b-7b/prompts.json");
  assert.equal(fixture.inputs.sha256_lf, sha256Lf(join(REF, "prompts.json")), "prompts.json changed: record the fixture again");
  assert.equal(fixture.recorder.script, "reference/record_moe.py");
  assert.equal(fixture.recorder.sha256_lf, sha256Lf(here("../reference/record_moe.py")), "record_moe.py changed: record the fixture again");
  // record_moe.py takes its helpers and its pinned versions from record.py.
  assert.equal(fixture.recorder.helpers.script, "reference/record.py");
  assert.equal(fixture.recorder.helpers.sha256_lf, sha256Lf(here("../reference/record.py")), "record.py changed: record this fixture again");
  assert.deepEqual(fixture.software, {
    torch: "2.14.0+cpu",
    transformers: "5.17.0",
    tokenizers: "0.23.2",
    safetensors: "0.8.0",
    numpy: "2.5.3",
    python: fixture.software.python,
  });
  assert.equal(fixture.settings.dtype, "float32");
  assert.equal(fixture.settings.attn_implementation, "eager");
  assert.equal(fixture.settings.experts_implementation, "eager");
  assert.equal(fixture.settings.threads, 1);
  assert.ok(readFileSync(join(REF, "fixture.json")).length < 4 * 1024 * 1024, "the fixture stays under 4 MB");

  // Every file the model is made of is pinned, the three weight shards included.
  const shards = JSON.parse(readFileSync(join(REF, "model.json"), "utf8")).files.map((f) => f.path).filter((p) => p.endsWith(".safetensors"));
  assert.deepEqual(shards, ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]);

  assert.deepEqual(fixture.prompts.map((p) => p.text), inputs.prompts.map((p) => p.text));
  assert.deepEqual(fixture.corpus.map((c) => c.text), inputs.corpus);

  // The prompts and the first 42 corpus strings are gate A1's.
  const a1 = JSON.parse(readFileSync(here("../reference/smollm2-135m/prompts.json"), "utf8"));
  assert.deepEqual(inputs.prompts, a1.prompts);
  assert.deepEqual(inputs.corpus.slice(0, a1.corpus.length), a1.corpus);
  assert.equal(inputs.greedy_steps, a1.greedy_steps);

  // For this model, transformers runs the model's own tokenizer.json, so the
  // two sets of ids agree on every string.
  const disagree = fixture.corpus.filter((c) => JSON.stringify(c.ids) !== JSON.stringify(c.tokenizer_json_ids)).map((c) => c.text);
  assert.deepEqual(disagree, fixture.tokenizer_disagreements);
  assert.deepEqual(disagree, []);

  for (const p of fixture.prompts) {
    assert.equal(p.positions.length, p.ids.length, p.id);
    assert.equal(p.lastLogits.length, VOCAB, p.id);
    assert.equal(p.greedy.length, inputs.greedy_steps, p.id);
    assert.deepEqual(p.greedy.map((g) => g.id), p.greedy_ids, p.id);
    const last = p.positions.at(-1);
    for (const [id, v] of last.top) assert.equal(p.lastLogits[id], v, `${p.id} id ${id}`);
    assert.equal(p.greedy[0].id, last.top[0][0], p.id);
  }
});

test("the tokenizer cases this model adds tokenize as its added tokens and NFC say", () => {
  const ids = (text) => fixture.corpus.find((c) => c.text === text).tokenizer_json_ids;
  assert.deepEqual(ids("|||IP_ADDRESS|||"), [0]);
  assert.deepEqual(ids("<|padding|>"), [1]);
  assert.deepEqual(ids("|||EMAIL_ADDRESS|||"), [50277]);
  assert.deepEqual(ids("|||PHONE_NUMBER|||"), [50278]);
  assert.deepEqual(ids("<|endoftext|>"), [50279]);
  // Runs of spaces are added tokens, 24 spaces (50254) down to 2 (50276),
  // taken longest first before the byte-level split.
  assert.deepEqual(ids(" ".repeat(25)), [50254, 209]);
  assert.deepEqual(ids(`x${" ".repeat(30)}y`), [89, 50254, 50272, 90]);
  assert.deepEqual(ids("  hello  world  "), [50276, 25521, 50276, 10186, 50276]);
  // NFC first: three spellings of the same letter, and a decomposed syllable,
  // come out as one; a ligature, which only NFKC would change, does not.
  const angstrom = fixture.corpus.find((c) => c.text === "Å Å Å");
  assert.equal(angstrom.decoded, "Å Å Å");
  assert.equal(angstrom.tokenizer_json_ids[1], angstrom.tokenizer_json_ids[2]);
  assert.equal(fixture.corpus.find((c) => c.text === "ﬁ fi").decoded, "ﬁ fi");
});

// The routing: for every layer of every prompt position, and of every token
// greedy decoding fed back, the 8 experts chosen in the router's order, then
// the closest expert not chosen; their router logits; and the logsumexp of
// all of them.
function* routes(p) {
  for (const [t, r] of p.routing.entries()) yield [`${p.id} position ${t}`, r];
  for (const [s, r] of p.greedy_routing.entries()) yield [`${p.id} greedy ${s}`, r];
}

test("the routing is complete and consistent", () => {
  const { layers, experts, experts_per_token: k } = fixture.model;
  assert.deepEqual([layers, experts, k, fixture.settings.router_topk], [16, 64, 8, 9]);
  let decisions = 0;
  for (const p of fixture.prompts) {
    assert.equal(p.routing.length, p.ids.length, p.id);
    assert.equal(p.greedy_routing.length, p.greedy.length - 1, p.id);
    for (const [where, r] of routes(p)) {
      assert.equal(r.experts.length, layers, where);
      assert.equal(r.logits.length, layers, where);
      assert.equal(r.lse.length, layers, where);
      for (let l = 0; l < layers; l++) {
        const ids = r.experts[l];
        const logits = r.logits[l];
        assert.equal(ids.length, k + 1, where);
        assert.equal(new Set(ids).size, k + 1, `${where} layer ${l}: an expert twice`);
        assert.ok(ids.every((e) => Number.isInteger(e) && e >= 0 && e < experts), `${where} layer ${l}`);
        // The router's order is by weight, so by logit; the expert left out
        // scored no higher than any chosen one.
        for (let i = 1; i < k; i++) assert.ok(logits[i] <= logits[i - 1], `${where} layer ${l}: out of order`);
        assert.ok(logits[k] <= logits[k - 1], `${where} layer ${l}: the expert left out scored higher`);
        // Each weight is exp(logit - lse). OLMoE does not renormalize the 8,
        // so they sum to less than 1.
        const sum = logits.slice(0, k).reduce((a, v) => a + Math.exp(v - r.lse[l]), 0);
        assert.ok(sum > 0 && sum < 1, `${where} layer ${l}: weights sum to ${sum}`);
        decisions++;
      }
    }
  }
  const positions = fixture.prompts.reduce((a, p) => a + p.ids.length + p.greedy.length - 1, 0);
  assert.equal(decisions, positions * layers);
});

test("the routing figures reference/README.md and compare.mjs quote are this fixture's", () => {
  const k = fixture.model.experts_per_token;
  const margins = [];
  const kept = [];
  for (const p of fixture.prompts) {
    for (const [where, r] of routes(p)) {
      r.logits.forEach((lg, l) => {
        margins.push([Math.min(...lg.slice(0, k)) - lg[k], `${where} layer ${l}`]);
        for (const v of lg) kept.push(Math.abs(v));
      });
    }
  }
  margins.sort((a, b) => a[0] - b[0]);
  const within = (bound) => margins.filter(([m]) => m < bound).length;
  assert.equal(margins.length, 6848);
  assert.deepEqual([2e-2, 1e-2, 2e-3, 1e-3, 1e-4, 1e-5].map(within), [1416, 774, 159, 85, 9, 2]);
  assert.deepEqual(margins.slice(0, 2).map(([m, at]) => [m.toExponential(2), at]), [
    ["5.45e-6", "unicode greedy 26 layer 2"],
    ["6.62e-6", "code greedy 10 layer 5"],
  ]);
  kept.sort((a, b) => a - b);
  assert.equal(kept[kept.length >> 1].toFixed(2), "0.42");
  assert.equal(kept.at(-1).toFixed(1), "5.7");
});

// The comparison, on the one full row the fixture keeps for each prompt, as in
// tests/reference.test.mjs.
function lastPositionOnly(p) {
  return { ...p, ids: [p.ids.at(-1)], positions: [p.positions.at(-1)] };
}

test("the comparison accepts the reference's own logits and rejects a wrong answer", () => {
  for (const p of fixture.prompts) {
    const one = lastPositionOnly(p);
    const exact = comparePrefill(fixture, one, p.lastLogits.slice());
    assert.deepEqual(exact.failures, [], p.id);
    assert.ok(exact.worst.logit === 0 && exact.worst.lse < 1e-9, p.id);

    const close = p.lastLogits.map((v) => v + 0.4 * LOGIT_ATOL);
    assert.deepEqual(comparePrefill(fixture, one, close).failures, [], p.id);

    const probe = p.lastLogits.slice();
    probe[fixture.probe_ids[7]] += 2 * LOGIT_ATOL;
    assert.ok(comparePrefill(fixture, one, probe).failures.length > 0, `${p.id}: a wrong probe logit passed`);

    // A logit the fixture keeps only in the full row: the output layer's last
    // row, past the tokenizer's 50,280 ids, which no token maps to.
    const kept = new Set([...one.positions[0].top.map(([id]) => id), ...fixture.probe_ids]);
    const other = [...Array(VOCAB).keys()].reverse().find((i) => !kept.has(i));
    assert.ok(other >= 50280, `${p.id}: ${other}`);
    const tail = p.lastLogits.slice();
    tail[other] -= 2 * LOGIT_ATOL;
    assert.ok(comparePrefill(fixture, one, tail).failures.length > 0, `${p.id}: a wrong tail logit passed`);

    const shifted = p.lastLogits.map((v) => v + 2 * LOGIT_ATOL);
    assert.ok(comparePrefill(fixture, one, shifted).failures.some((f) => f.includes("logsumexp")), p.id);
  }

  for (const p of fixture.prompts) {
    const steps = new Float32Array(p.greedy.length * VOCAB).fill(-1e4);
    p.greedy.forEach((g, s) => g.top.forEach(([id, v]) => (steps[s * VOCAB + id] = v)));
    assert.deepEqual(compareGreedy(fixture, p, p.greedy_ids, steps).failures, [], p.id);
    const changed = p.greedy_ids.slice();
    changed[5] = changed[5] === 0 ? 1 : 0;
    assert.ok(compareGreedy(fixture, p, changed, steps).failures.length > 0, p.id);
  }

  const fileIds = (text) => {
    const c = [...fixture.corpus, ...fixture.prompts].find((x) => x.text === text);
    return c.tokenizer_json_ids ?? c.ids;
  };
  assert.deepEqual(compareTokenization(fixture, fileIds), []);
  assert.ok(compareTokenization(fixture, () => [0]).length > 0);
});

// Router logits and choices as an engine would report them, built from the
// fixture: the 9 experts it keeps at their recorded logits, the other 55 one
// below the runner-up, and the reference's 8 chosen.
function asReported(entries) {
  const { layers, experts, experts_per_token: k } = fixture.model;
  const logits = new Float32Array(entries.length * layers * experts);
  const chosen = new Int32Array(entries.length * layers * k);
  entries.forEach((r, t) => {
    for (let l = 0; l < layers; l++) {
      const o = (t * layers + l) * experts;
      logits.fill(r.logits[l][k] - 1, o, o + experts);
      r.experts[l].forEach((e, i) => (logits[o + e] = r.logits[l][i]));
      chosen.set(r.experts[l].slice(0, k), (t * layers + l) * k);
    }
  });
  return { logits, chosen };
}

test("the routing comparison accepts the reference's routing, and a runner-up only at a near-tie", () => {
  const { layers, experts, experts_per_token: k } = fixture.model;
  for (const p of fixture.prompts) {
    for (const [entries, label] of [[p.routing, `${p.id} position`], [p.greedy_routing, `${p.id} greedy`]]) {
      const { logits, chosen } = asReported(entries);
      const exact = compareRouting(fixture, entries, logits, chosen, label);
      assert.deepEqual(exact.failures, [], label);
      assert.equal(exact.worst.logit, 0, label);
      const close = logits.map((v) => v + 0.4 * ROUTE_ATOL);
      assert.deepEqual(compareRouting(fixture, entries, close, chosen, label).failures, [], label);
    }
  }

  // Every decision, with how close the reference's runner-up came.
  const decisions = [];
  for (const p of fixture.prompts) {
    for (const [key, entries] of [["routing", p.routing], ["greedy_routing", p.greedy_routing]]) {
      entries.forEach((r, t) => r.logits.forEach((lg, l) => decisions.push({ p, key, t, l, margin: Math.min(...lg.slice(0, k)) - lg[k] })));
    }
  }
  decisions.sort((a, b) => a.margin - b.margin);
  const run = ({ p, key }) => p[key];
  const at = ({ t, l }) => (t * layers + l) * k;

  // The closest call in the fixture: the runner-up may take the 8th's place,
  // with each of the two logits moved by less than ROUTE_ATOL.
  const nearest = decisions[0];
  assert.ok(nearest.margin < 2 * ROUTE_ATOL);
  {
    const entries = run(nearest);
    const { logits, chosen } = asReported(entries);
    const [ids, ref] = [entries[nearest.t].experts[nearest.l], entries[nearest.t].logits[nearest.l]];
    const o = (nearest.t * layers + nearest.l) * experts;
    logits[o + ids[k - 1]] = ref[k - 1] - 0.6 * nearest.margin;
    logits[o + ids[k]] = ref[k] + 0.6 * nearest.margin;
    chosen[at(nearest) + k - 1] = ids[k];
    assert.deepEqual(compareRouting(fixture, entries, logits, chosen, "nearest").failures, [], "a near-tie decided the other way");
  }

  // Where the reference's 7th and 8th both came within 2 * ROUTE_ATOL of the
  // runner-up, the runner-up may stand in for either, but for one only.
  const double = decisions.find((d) => {
    const lg = run(d)[d.t].logits[d.l];
    return lg[k - 2] - lg[k] < 2 * ROUTE_ATOL;
  });
  {
    const entries = run(double);
    const ids = entries[double.t].experts[double.l];
    const outside = [...Array(experts).keys()].find((e) => !ids.includes(e));
    let { logits, chosen } = asReported(entries);
    chosen[at(double) + k - 2] = ids[k];
    assert.deepEqual(compareRouting(fixture, entries, logits, chosen, "double").failures, [], "the runner-up in the 7th's place");
    ({ logits, chosen } = asReported(entries));
    chosen[at(double) + k - 2] = ids[k];
    chosen[at(double) + k - 1] = outside;
    assert.ok(compareRouting(fixture, entries, logits, chosen, "double").failures.length > 0, "two stand-ins passed");
  }

  // A wide one: the same stand-in fails.
  const wide = decisions.find((d) => d.margin > 0.1);
  {
    const entries = run(wide);
    const { logits, chosen } = asReported(entries);
    chosen[at(wide) + k - 1] = entries[wide.t].experts[wide.l][k];
    assert.ok(compareRouting(fixture, entries, logits, chosen, "wide").failures.length > 0, "a runner-up 0.1 behind stood in");
  }

  // At the closest call too, an expert the reference ranked below its
  // runner-up may not stand in; nor may an expert be chosen twice, or a
  // router logit be off by twice the tolerance.
  {
    const d = nearest;
    const entries = run(d);
    const ids = entries[d.t].experts[d.l];
    const outside = [...Array(experts).keys()].find((e) => !ids.includes(e));

    let { logits, chosen } = asReported(entries);
    chosen[at(d) + k - 1] = outside;
    assert.ok(compareRouting(fixture, entries, logits, chosen, "outside").failures.length > 0, "an expert outside the 9 passed");

    ({ logits, chosen } = asReported(entries));
    chosen[at(d) + k - 1] = chosen[at(d)];
    assert.ok(compareRouting(fixture, entries, logits, chosen, "twice").failures.length > 0, "an expert chosen twice passed");

    ({ logits, chosen } = asReported(entries));
    logits[(d.t * layers + d.l) * experts + ids[3]] += 2 * ROUTE_ATOL;
    assert.ok(compareRouting(fixture, entries, logits, chosen, "logit").failures.length > 0, "a router logit off by 2e-3 passed");

    // A report for one position more than the run has.
    ({ logits, chosen } = asReported(entries));
    const longer = new Float32Array(logits.length + layers * experts);
    longer.set(logits);
    assert.ok(compareRouting(fixture, entries, longer, chosen, "longer").failures.length > 0, "a report of the wrong length passed");
  }
});

// ---------------------------------------------------------------------------
// The engine against the fixture: gate A5, step 2, on the CPU.

const EXE = process.platform === "win32" ? ".exe" : "";
const engine = [
  process.env.VITNA_ENGINE,
  here(`../engine/vitna-anchor${EXE}`),
  here(`../engine/build/Release/vitna-anchor${EXE}`),
  here(`../engine/build/vitna-anchor${EXE}`),
].filter(Boolean).find((p) => existsSync(p));
const modelDir = process.env.ANCHOR_MOE_MODEL_DIR || here("../models/olmoe-1b-7b/");
const LAYERS = fixture.model.layers;
const K = fixture.model.experts_per_token;
const required = (v) => process.env[v] === "1";
const DEVICE = process.env.VITNA_DEVICE ?? "";
assert.ok(["", "cpu", "cuda"].includes(DEVICE), `VITNA_DEVICE must be cpu or cuda, not ${DEVICE}`);
const onDevice = DEVICE ? ["--device", DEVICE] : [];
const GPU = DEVICE === "cuda";

function missing(files, fetch) {
  if (!engine) return "no built engine found";
  const absent = files.find((f) => !existsSync(join(modelDir, f)));
  return absent ? `no ${absent} in ${modelDir}; run ${fetch}` : false;
}
const TOKENIZER = {
  skip: required("VITNA_REQUIRE_MOE") || required("VITNA_REQUIRE_MOE_TOKENIZER") ? false
    : missing(["tokenizer.json"], "node scripts/fetch-model.mjs olmoe-1b-7b --only tokenizer.json"),
};
// On a GPU each engine takes what the device has free for its experts, so
// beside the other files' servers, which VITNA_DEVICE=cuda also puts there,
// one of them would find no memory. There these run only when asked for,
// with VITNA_REQUIRE_MOE=1, and with this file alone.
const MODEL = {
  skip: required("VITNA_REQUIRE_MOE") ? false
    : GPU ? "on a GPU these run only with VITNA_REQUIRE_MOE=1, and this file alone: each engine takes what the GPU has free"
    : missing(pin.files.map((f) => f.path), "node scripts/fetch-model.mjs olmoe-1b-7b"),
};

/** Run the engine. Resolves with its stdout; rejects with its stderr if it exits other than 0. */
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

/** The fixture's experts for a run of positions, as --experts-in reads them: positions x layers x 8, int32 LE. */
function writePins(path, entries) {
  const buf = Buffer.alloc(entries.length * LAYERS * K * 4);
  entries.forEach((r, t) => r.experts.forEach((ids, l) => ids.slice(0, K).forEach((e, i) => buf.writeInt32LE(e, ((t * LAYERS + l) * K + i) * 4))));
  writeFileSync(path, buf);
}

/** The first decision, as [position, layer], where the experts chosen are not the reference's 8, or null. */
function firstDeparture(entries, chosen) {
  for (let t = 0; t < entries.length; t++) {
    for (let l = 0; l < LAYERS; l++) {
      const mine = [...chosen.slice((t * LAYERS + l) * K, (t * LAYERS + l + 1) * K)].sort((a, b) => a - b).join();
      if (mine !== entries[t].experts[l].slice(0, K).sort((a, b) => a - b).join()) return [t, l];
    }
  }
  return null;
}

// Every run of the model: --ctx 512, on the device VITNA_DEVICE names. The
// CPU path runs a step at a time on one thread, so there the six prompts run
// side by side, each in an engine of its own over the same mapped files. On
// a GPU they run one after another, since each engine's expert cache takes
// what the device has free.
const RUN = ["--ctx", "512", ...onDevice];
async function eachPrompt(prefix, run) {
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
}

test("the engine's tokenizer gives the ids of the model's own tokenizer.json", TOKENIZER, async (t) => {
  const cases = [...fixture.corpus, ...fixture.prompts];
  const out = await runEngine(["tokenize", "--model", modelDir], cases.map((c) => JSON.stringify(c.text)).join("\n") + "\n");
  const lines = out.trim().split("\n").map((l) => JSON.parse(l));
  assert.equal(lines.length, cases.length);
  const byText = new Map(cases.map((c, i) => [c.text, lines[i]]));
  assert.deepEqual(compareTokenization(fixture, (text) => byText.get(text)), []);
  t.diagnostic(`${cases.length} of ${cases.length} strings tokenized exactly, NFC and the added tokens included`);
});

// With its routing pinned to the reference's (--experts-in), the engine's
// logits are compared as gate A2 compares SmolLM2's, so that a near-tie it
// decides the other way cannot fail the comparison of everything after it.
// Its own router still runs, and what it chose is compared under the
// routing rule. Where it chose the reference's experts, the pinned run is
// the run it would make unpinned, bit for bit.
test("with its routing pinned, the engine's logits match the reference within the tolerance, and its router chooses as the rule allows", MODEL, async (t) => {
  const results = await eachPrompt("vitna-moe-logits-", async (p, f) => {
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
  const decisions = fixture.prompts.reduce((a, p) => a + p.ids.length * LAYERS, 0);
  const departed = results.filter((r) => r.departure).map((r) => `${r.p.id} position ${r.departure[0]} layer ${r.departure[1]}`);
  t.diagnostic(`on ${DEVICE || "cpu"}, largest |engine - reference|: logits ${worst("prefill", "logit").toExponential(2)}, logsumexp ${worst("prefill", "lse").toExponential(2)}, router logits ${worst("routing", "logit").toExponential(2)}; tolerances ${LOGIT_ATOL} and ${ROUTE_ATOL}`);
  t.diagnostic(departed.length ? `the router first chose other experts than the reference at ${departed.join("; ")}` : `the router chose the reference's experts in all ${decisions} decisions`);
});

test("with its routing pinned, the engine's greedy decoding matches the reference token for token", MODEL, async (t) => {
  const results = await eachPrompt("vitna-moe-greedy-", async (p, f) => {
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

// The tests above pin the experts the router chooses anyway, so they would
// pass with the pins ignored. Here one decision is pinned the other way, as
// reference/routing_sensitivity.py forces it: the prompt's closest call at a
// prompt position, its 8th expert replaced by the runner-up. The positions
// before it must not move, and the logits from it on must, past the
// tolerance.
test("pinned experts are the ones a token goes through", MODEL, async (t) => {
  const p = fixture.prompts.find((x) => x.id === "capital");
  const [, at, layer] = p.routing.flatMap((r, pos) => r.logits.map((lg, l) => [Math.min(...lg.slice(0, K)) - lg[K], pos, l])).sort((a, b) => a[0] - b[0])[0];
  const swapped = p.routing.map((r, pos) => (pos !== at ? r : {
    ...r,
    experts: r.experts.map((ids, l) => (l !== layer ? ids : [...ids.slice(0, K - 1), ids[K], ids[K - 1]])),
  }));
  const dir = mkdtempSync(join(tmpdir(), "vitna-moe-swap-"));
  try {
    writePins(join(dir, "pin"), swapped);
    await runEngine(["logits", "--model", modelDir, "--ids", p.ids.join(","), "--out", join(dir, "logits"), ...RUN, "--experts-in", join(dir, "pin")]);
    const rows = readF32(join(dir, "logits"));
    const V = fixture.model.vocab_size;
    const moved = p.positions.map((pos, i) =>
      Math.max(...[...pos.top, ...fixture.probe_ids.map((id, k) => [id, pos.probe[k]])].map(([id, v]) => Math.abs(rows[i * V + id] - v))));
    assert.ok(moved.slice(0, at).every((d) => d <= LOGIT_ATOL), `positions before ${at} moved: ${moved}`);
    assert.ok(Math.max(...moved.slice(at)) > LOGIT_ATOL, `the pinned swap at position ${at} moved nothing past the tolerance: ${moved}`);
    t.diagnostic(`on ${DEVICE || "cpu"}, ${p.id} position ${at} layer ${layer}, the runner-up pinned in place of the 8th expert: the logits after it moved by up to ${Math.max(...moved).toExponential(2)}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// Unpinned, the engine routes on its own. Everything is compared as above up
// to the first decision where it chose other experts than the reference,
// which must be one the rule allows, a near-tie; after it the engine is on a
// path the fixture does not describe, and nothing more is compared.
test("unpinned, the engine decodes the reference's tokens until its routing departs at a near-tie, if it ever does", MODEL, async (t) => {
  const results = await eachPrompt("vitna-moe-free-", async (p, f) => {
    const out = await runEngine(["generate", "--model", modelDir, "--ids", p.ids.join(","), "--max-new", String(p.greedy_ids.length), "--greedy",
      ...RUN, "--logits-out", f("logits"), "--router-out", f("router"), "--experts-out", f("experts")]);
    return { p, ids: JSON.parse(out).ids, logits: readF32(f("logits")), router: readF32(f("router")), chosen: readI32(f("experts")) };
  });
  const E = fixture.model.experts;
  const notes = [];
  for (const { p, ids, logits, router, chosen } of results) {
    const entries = [...p.routing, ...p.greedy_routing];
    const at = firstDeparture(entries, chosen) ?? [entries.length, 0];
    const [d, dl] = at;
    // Whole positions before the departure, then its own position's layers up to it.
    const before = compareRouting(fixture, entries.slice(0, d), router.subarray(0, d * LAYERS * E), chosen.subarray(0, d * LAYERS * K), `${p.id} position`);
    assert.deepEqual(before.failures.slice(0, 10), [], p.id);
    if (d < entries.length) {
      const sub = { model: { ...fixture.model, layers: dl + 1 } };
      const entry = { experts: entries[d].experts.slice(0, dl + 1), logits: entries[d].logits.slice(0, dl + 1), lse: entries[d].lse.slice(0, dl + 1) };
      const own = compareRouting(sub, [entry], router.subarray(d * LAYERS * E, (d * LAYERS + dl + 1) * E), chosen.subarray(d * LAYERS * K, (d * LAYERS + dl + 1) * K), `${p.id} position ${d}`);
      assert.deepEqual(own.failures, [], `${p.id}: the routing departed where the rule does not allow it`);
      notes.push(`${p.id} departed at position ${d} layer ${dl}`);
    }
    // Step s takes its logits from position ids.length - 1 + s, so the steps before the departure.
    const steps = Math.max(0, Math.min(p.greedy.length, d - p.ids.length + 1));
    const V = fixture.model.vocab_size;
    const result = compareGreedy(fixture, { ...p, greedy: p.greedy.slice(0, steps), greedy_ids: p.greedy_ids.slice(0, steps) }, ids.slice(0, steps), logits.subarray(0, steps * V));
    assert.deepEqual(result.failures.slice(0, 10), [], p.id);
    if (steps === p.greedy.length) assert.deepEqual(ids, p.greedy_ids, p.id);
  }
  t.diagnostic(`on ${DEVICE || "cpu"}, ` + (notes.length ? notes.join("; ") : `its routing was the reference's throughout, and all ${fixture.prompts.length * fixture.prompts[0].greedy.length} greedy tokens equal`));
});

// Run capital and code under each of configs, [name, extra arguments]: 16
// greedy tokens with their logits, and every position's logits. Every run
// must give the first's tokens, and its logits byte for byte. On the CPU the
// runs of a prompt go side by side; on a GPU one after another.
async function sameEverywhere(configs) {
  const prompts = fixture.prompts.filter((p) => ["capital", "code"].includes(p.id));
  const dir = mkdtempSync(join(tmpdir(), "vitna-moe-same-"));
  try {
    for (const p of prompts) {
      const run = async ([name, extra]) => {
        const out = await runEngine(["generate", "--model", modelDir, "--ids", p.ids.join(","), "--max-new", "16", "--greedy", ...RUN,
          "--logits-out", join(dir, `${p.id}.${name}.greedy`), ...extra]);
        await runEngine(["logits", "--model", modelDir, "--ids", p.ids.join(","), "--out", join(dir, `${p.id}.${name}.logits`), ...RUN, ...extra]);
        return JSON.parse(out).ids;
      };
      const ids = [];
      if (GPU) for (const c of configs) ids.push(await run(c));
      else ids.push(...(await Promise.all(configs.map(run))));
      const [first] = configs;
      for (const [i, [name]] of configs.entries()) {
        assert.deepEqual(ids[i], ids[0], `${p.id}: ${name} chose other tokens than ${first[0]}`);
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
}

// Gate A5, step 3: the experts read from the drive (--expert-cache), with
// direct I/O, into a cache in memory far smaller than they are, so that
// nearly every one is read when a layer wants it or a prefetch guessed it.
// What is computed must not change at all: the same bytes, run through the
// same arithmetic, give the mapped run's logits bit for bit. On a GPU the
// experts are copied to the device from that cache instead of from the
// mapped checkpoint, which must not change them either.
test("read from the drive into a small cache, the experts give the mapped run's logits and tokens, bit for bit", MODEL, async (t) => {
  const which = await sameEverywhere([["mapped", []], ["streamed", ["--expert-cache", "256"]]]);
  t.diagnostic(`on ${DEVICE || "cpu"}, ${which}: every position's logits, and 16 greedy tokens with their logits, equal byte for byte, read through a cache of 256 MiB`);
});

// On a GPU the experts live in a cache on the device, filled as layers want
// them and as the next layer's router guesses. An expert's arithmetic does
// not depend on which slot holds it, when it was copied there or from where,
// so a cache of 21 slots, refilled at nearly every layer, must give the
// logits of one holding nearly half of them, bit for bit, and so must one
// fed from a cache in memory of 21 slots too.
test("on the GPU, an expert cache of any size, fed from either place, gives the same logits and tokens, bit for bit", {
  skip: MODEL.skip || (!GPU && "VITNA_DEVICE is not cuda"),
}, async (t) => {
  const which = await sameEverywhere([
    ["default", []],
    ["small", ["--gpu-expert-cache", "256"]],
    ["both-small", ["--gpu-expert-cache", "256", "--expert-cache", "256"]],
  ]);
  t.diagnostic(`${which}: every position's logits, and 16 greedy tokens with their logits, equal byte for byte with the GPU's expert cache as large as the device allows, at 256 MiB, and at 256 MiB fed from a cache of 256 MiB in memory`);
});

test("an expert cache too small for twice the experts a token goes through is refused, saying how large it must be", MODEL, async () => {
  await assert.rejects(
    runEngine(["logits", "--model", modelDir, "--ids", "510", "--out", join(tmpdir(), "vitna-moe-unused.f32"), ...RUN, "--expert-cache", "64"]),
    /expert cache must hold twice the experts a token goes through: \d+ MiB or more/,
  );
});

test("on the GPU, an expert cache there too small for twice the experts a token goes through is refused, saying how large it must be", {
  skip: MODEL.skip || (!GPU && "VITNA_DEVICE is not cuda"),
}, async () => {
  await assert.rejects(
    runEngine(["logits", "--model", modelDir, "--ids", "510", "--out", join(tmpdir(), "vitna-moe-unused.f32"), ...RUN, "--gpu-expert-cache", "100"]),
    /--device cuda: the GPU's expert cache must hold twice the experts a token goes through: 192 MiB or more/,
  );
});
