// Gate A5, step 1: the reference for a mixture of experts.
//
// OLMoE-1B-7B's fixture, recorded by reference/record_moe.py as gate A1's is
// recorded by record.py, with each layer's routing added. These tests need
// nothing but the repository: the fixture is the one its pin, its inputs and
// its recorders describe, its routing is complete and consistent, and the
// comparison in reference/compare.mjs accepts the fixture's own logits and
// rejects wrong ones at this vocabulary too. The engine cannot run this model
// yet, so the comparisons with it are todo until step 2.

import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
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

// Gate A5 step 2: the engine runs this model on the CPU and matches the
// fixture under reference/compare.mjs, routing included.
test.todo("the engine's tokenizer gives the ids of the model's own tokenizer.json");
test.todo("the engine routes each token to the reference's experts");
test.todo("the engine's logits match the reference within the stated tolerance");
test.todo("the engine's greedy decoding matches the reference token for token");
