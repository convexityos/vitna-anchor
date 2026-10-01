// How the engine's output is compared with the reference fixture.
//
// The tolerances below were fixed in gate A1, before the engine could produce
// a single logit, so gate A2 is held to a line drawn in advance. Loosening one
// is a visible change to this file, with its reason in the pull request.
//
//   Token ids              exactly equal to the tokenizers library's, running
//                          the model's own tokenizer.json, for every prompt
//                          and corpus string. First written as "the
//                          reference's ids". Comparing the engine with them
//                          showed that transformers 5.17 drops this model's
//                          Digits pre-tokenizer step, so on one corpus string
//                          its ids are not the file's. The fixture keeps both,
//                          and names that string in tokenizer_disagreements.
//   Logits                 |engine - reference| <= LOGIT_ATOL, for the 16
//                          largest and 64 probe logits at every position of
//                          every prompt, and for all 49152 at the last one
//   logsumexp of a row     |difference| <= LOGIT_ATOL at every position
//   Argmax at a position   the reference's, unless the reference's top two
//                          are within 2 * LOGIT_ATOL of each other, where
//                          either of the two is accepted
//   Greedy decoding        the same token ids for all steps, token for token,
//                          and the top 8 logits at each step within LOGIT_ATOL
//
// Why 1e-2: the reference is computed in float32, and so is the engine's
// float path. Summing in a different order, or rounding exp and sin
// differently, moves logits by far less than 1e-2. A real defect, such as the
// wrong rotary convention, a missing norm, or a position off by one, moves
// them by 0.1 or more, and usually changes the argmax too.
//
// A mixture of experts adds one more, set in gate A5 before the engine could
// route a single token:
//
//   Routing                at every layer of every position: the router
//                          logits of the 9 experts the fixture keeps within
//                          ROUTE_ATOL of the reference's, and the 8 experts
//                          the token goes through the reference's 8, except
//                          that one whose router logit the reference puts
//                          within 2 * ROUTE_ATOL of the runner-up's may give
//                          way to the runner-up
//
// Why a band: a token goes through the 8 experts its router scores highest,
// so two scores that nearly tie can trade places when the same sums are
// added in another order, and everything after then moves by far more than
// rounding: by 0.011 to 0.21 when each prompt's closest call was forced the
// other way with every other decision held, and by up to 0.39 when the
// others were left to the router. Why 1e-3: run in two other float32
// orders, OLMoE's reference moved its router logits by at most 1.05e-5 and
// changed no routing decision, so the band is about a hundred times that,
// and a tenth of LOGIT_ATOL for logits that are small (those the fixture
// keeps have a median magnitude of 0.42, and none exceeds 5.7). A band of
// 2e-2 would leave 1,416 of the fixture's 6,848 routing decisions free to go
// either way; 2e-3 leaves 159. The measurements come from
// reference/routing_sensitivity.py.

import { readFileSync } from "node:fs";

export const LOGIT_ATOL = 1e-2;
export const ROUTE_ATOL = 1e-3;

/**
 * Read the fixture and decode each prompt's last row of logits. Logits are
 * written as the shortest decimal that reads back as the float32, so each is
 * rounded back to that float32 here (Math.fround) to compare exactly.
 */
export function loadFixture(path) {
  const fixture = JSON.parse(readFileSync(path, "utf8"));
  const f32pairs = (pairs) => pairs.map(([id, v]) => [id, Math.fround(v)]);
  for (const p of fixture.prompts) {
    const bytes = Buffer.from(p.last_logits_f32le_b64, "base64");
    p.lastLogits = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    for (const pos of p.positions) {
      pos.top = f32pairs(pos.top);
      pos.probe = pos.probe.map(Math.fround);
      pos.max = Math.fround(pos.max);
      pos.margin = Math.fround(pos.margin);
    }
    for (const g of p.greedy) {
      g.top = f32pairs(g.top);
      g.margin = Math.fround(g.margin);
    }
    // A mixture of experts' fixture keeps router logits too.
    for (const r of [...(p.routing ?? []), ...(p.greedy_routing ?? [])]) {
      r.logits = r.logits.map((row) => row.map(Math.fround));
    }
  }
  return fixture;
}

function logsumexp(row) {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  let sum = 0;
  for (const v of row) sum += Math.exp(v - max);
  return max + Math.log(sum);
}

function argmax(row) {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/**
 * Compare token ids with the model's own tokenizer.json, as the tokenizers
 * library runs it. (A prompt's ids are the same under both tokenizers; the
 * recorder refuses a prompt where they are not.) `idsFor(text)` returns the
 * engine's ids for a string. Returns the mismatches, each { text, expected, actual }.
 */
export function compareTokenization(fixture, idsFor) {
  const mismatches = [];
  const cases = [...fixture.corpus, ...fixture.prompts];
  for (const c of cases) {
    const expected = c.tokenizer_json_ids ?? c.ids;
    const actual = idsFor(c.text);
    if (actual.length !== expected.length || actual.some((id, i) => id !== expected[i])) {
      mismatches.push({ text: c.text, expected, actual });
    }
  }
  return mismatches;
}

/**
 * Compare the logits for one prompt's prefill. `rows` holds T rows of V
 * float32 logits, one per position. Returns { worst, failures }.
 */
export function comparePrefill(fixture, prompt, rows, vocab = fixture.model.vocab_size) {
  const failures = [];
  const worst = { logit: 0, lse: 0 };
  if (rows.length !== prompt.ids.length * vocab) {
    return { worst, failures: [`${prompt.id}: expected ${prompt.ids.length} x ${vocab} logits, got ${rows.length}`] };
  }
  for (let t = 0; t < prompt.ids.length; t++) {
    const row = rows.subarray(t * vocab, (t + 1) * vocab);
    const ref = prompt.positions[t];
    const check = (id, expected, what) => {
      const d = Math.abs(row[id] - expected);
      worst.logit = Math.max(worst.logit, d);
      if (!(d <= LOGIT_ATOL)) failures.push(`${prompt.id} position ${t} ${what} id ${id}: ${row[id]} vs ${expected}`);
    };
    for (const [id, v] of ref.top) check(id, v, "top");
    fixture.probe_ids.forEach((id, k) => check(id, ref.probe[k], "probe"));

    const dl = Math.abs(logsumexp(row) - ref.lse);
    worst.lse = Math.max(worst.lse, dl);
    if (!(dl <= LOGIT_ATOL)) failures.push(`${prompt.id} position ${t}: logsumexp ${logsumexp(row)} vs ${ref.lse}`);

    const got = argmax(row);
    const [first, second] = ref.top;
    const accepted = ref.margin < 2 * LOGIT_ATOL ? [first[0], second[0]] : [first[0]];
    if (!accepted.includes(got)) failures.push(`${prompt.id} position ${t}: argmax ${got}, reference ${first[0]} (margin ${ref.margin})`);
  }
  const last = rows.subarray((prompt.ids.length - 1) * vocab);
  for (let i = 0; i < vocab; i++) {
    const d = Math.abs(last[i] - prompt.lastLogits[i]);
    worst.logit = Math.max(worst.logit, d);
    if (!(d <= LOGIT_ATOL)) {
      failures.push(`${prompt.id} last position id ${i}: ${last[i]} vs ${prompt.lastLogits[i]}`);
      break;
    }
  }
  return { worst, failures };
}

/**
 * Compare greedy decoding for one prompt. `ids` are the engine's chosen ids,
 * `steps` its logits at each step (steps x V float32). Returns { worst, failures }.
 */
export function compareGreedy(fixture, prompt, ids, steps, vocab = fixture.model.vocab_size) {
  const failures = [];
  const worst = { logit: 0 };
  const expected = prompt.greedy_ids;
  const firstDiff = expected.findIndex((id, i) => ids[i] !== id);
  if (ids.length !== expected.length || firstDiff !== -1) {
    failures.push(`${prompt.id}: greedy ids first differ at step ${firstDiff}: ${JSON.stringify(ids)} vs ${JSON.stringify(expected)}`);
  }
  const n = Math.min(prompt.greedy.length, steps.length / vocab);
  for (let s = 0; s < n; s++) {
    const row = steps.subarray(s * vocab, (s + 1) * vocab);
    for (const [id, v] of prompt.greedy[s].top) {
      const d = Math.abs(row[id] - v);
      worst.logit = Math.max(worst.logit, d);
      if (!(d <= LOGIT_ATOL)) failures.push(`${prompt.id} step ${s} id ${id}: ${row[id]} vs ${v}`);
    }
  }
  return { worst, failures };
}

/**
 * Compare routing with a mixture-of-experts fixture's, for a run of positions:
 * a prompt's `routing`, or its `greedy_routing`. Each entry keeps, per layer,
 * the 8 experts the reference chose and then its runner-up, with their router
 * logits. `routerLogits` holds the engine's router logits for every expert
 * (positions x layers x experts float32) and `chosen` the experts it sent
 * each position through (positions x layers x 8). `label` names the run in
 * failures. Returns { worst, failures }.
 */
export function compareRouting(fixture, entries, routerLogits, chosen, label) {
  const { layers, experts, experts_per_token: k } = fixture.model;
  const failures = [];
  const worst = { logit: 0 };
  if (routerLogits.length !== entries.length * layers * experts || chosen.length !== entries.length * layers * k) {
    const want = `${entries.length} x ${layers} x ${experts} router logits and ${entries.length} x ${layers} x ${k} experts`;
    return { worst, failures: [`${label}: expected ${want}, got ${routerLogits.length} and ${chosen.length}`] };
  }
  entries.forEach((entry, t) => {
    for (let l = 0; l < layers; l++) {
      const at = `${label} ${t} layer ${l}`;
      const row = routerLogits.subarray((t * layers + l) * experts, (t * layers + l + 1) * experts);
      const ids = entry.experts[l];
      const ref = entry.logits[l];
      ids.forEach((e, i) => {
        const d = Math.abs(row[e] - ref[i]);
        worst.logit = Math.max(worst.logit, d);
        if (!(d <= ROUTE_ATOL)) failures.push(`${at} expert ${e}: router logit ${row[e]} vs ${ref[i]}`);
      });
      const mine = [...chosen.slice((t * layers + l) * k, (t * layers + l + 1) * k)];
      const theirs = ids.slice(0, k);
      const left = theirs.filter((e) => !mine.includes(e));
      const added = mine.filter((e) => !theirs.includes(e));
      if (new Set(mine).size !== k) {
        failures.push(`${at}: chose ${JSON.stringify(mine)}, not ${k} different experts`);
      } else if (left.length > 0) {
        // The only stand-in allowed is the runner-up, for one expert the
        // reference scored within 2 * ROUTE_ATOL of it.
        const close = left.length === 1 && added[0] === ids[k] && ref[theirs.indexOf(left[0])] - ref[k] < 2 * ROUTE_ATOL;
        if (!close) failures.push(`${at}: experts ${JSON.stringify(mine)}, reference ${JSON.stringify(theirs)}, runner-up ${ids[k]}`);
      }
    }
  });
  return { worst, failures };
}
