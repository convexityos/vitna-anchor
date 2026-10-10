// bench/history.json, the engine's measured history, held to the README it
// indexes. Every figure an entry prints must be in the README section the
// entry names, as the README prints it, so neither can change without the
// other; and a figure measured from 2026-10-10 keeps its runs, from which its
// statistic is recomputed here at the precision it is printed. Nothing here
// runs the engine: it reads the two files.

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { median, mean, printedDecimals, printedValue, summarize } from "../bench/stats.mjs";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const history = JSON.parse(readFileSync(here("../bench/history.json"), "utf8"));
const readme = readFileSync(here("../README.md"), "utf8");

/** The README's text under the heading `title` (any level), to the next heading. */
function section(title) {
  const lines = readme.split("\n");
  const start = lines.findIndex((line) => /^#{2,4} /.test(line) && line.replace(/^#+ /, "").trim() === title);
  if (start < 0) return null;
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((line) => /^#{2,4} /.test(line));
  return (end < 0 ? rest : rest.slice(0, end)).join("\n");
}

const figuresOf = (entry) => [entry.result, ...(entry.also ?? []), ...(entry.compare ?? [])];
const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/** A figure as a whole number in text: "5.1" is not found in "15.1" or "5.12". */
const asFigure = (printed) => new RegExp(`(?<![\\d.,])${escape(printed)}(?![\\d]|[.,]\\d)`);

// From this date an entry keeps each run's value, so its spread can be shown.
const RUNS_KEPT_FROM = "2026-10-10";

test("every entry says when, on what, with which command, and where the README publishes it", () => {
  assert.ok(history.entries.length >= 10, `${history.entries.length} entries, so these checks would pass on almost nothing`);
  for (const e of history.entries) {
    for (const key of ["id", "date", "gate", "what", "machine", "model", "weights", "device", "command", "readme", "result"]) {
      assert.ok(e[key], `${e.id ?? "an entry"} has no ${key}`);
    }
    assert.match(e.date, /^\d{4}-\d{2}-\d{2}$/, `${e.id}: date`);
    const machine = history.machines[e.machine];
    assert.ok(machine, `${e.id} names a machine the history does not describe: ${e.machine}`);
    if (e.drive) assert.ok(machine.drives?.[e.drive], `${e.id} names a drive ${e.machine} does not have: ${e.drive}`);
    assert.ok(["cpu", "cuda"].includes(e.device), `${e.id}: device ${e.device}`);
    assert.ok(section(e.readme) !== null, `${e.id} names a README section that does not exist: ${e.readme}`);
    for (const f of figuresOf(e)) {
      assert.equal(typeof f.printed, "string", `${e.id}: a figure without its printed form`);
      assert.ok(f.unit, `${e.id}: ${f.printed} has no unit`);
      assert.ok(Number.isInteger(f.n) && f.n >= 1, `${e.id}: ${f.printed} does not say how many runs it summarizes`);
      assert.ok(typeof f.summary === "string" && f.summary.length > 0, `${e.id}: ${f.printed} does not say which statistic it is`);
    }
    for (const c of e.compare ?? []) {
      assert.ok(["before", "beside", "reference"].includes(c.kind), `${e.id}: compare kind ${c.kind}`);
      assert.ok(c.label, `${e.id}: ${c.printed} is compared without saying with what`);
    }
  }
});

test("entries are oldest first, and their ids unique", () => {
  const ids = history.entries.map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length, "an id appears twice");
  const dates = history.entries.map((e) => e.date);
  assert.deepEqual(dates, [...dates].sort(), "entries are not oldest first");
});

test("every figure is in the README section its entry names, as the README prints it", () => {
  for (const e of history.entries) {
    const text = section(e.readme);
    for (const f of figuresOf(e)) {
      assert.match(text, asFigure(f.printed), `${e.id}: "${f.printed}" (${f.unit}) is not in the README's "${e.readme}"`);
    }
    if (e.factor) assert.ok(text.includes(e.factor), `${e.id}: the factor "${e.factor}" is not in the README's "${e.readme}"`);
  }
});

test("a figure that keeps its runs is their median or mean, at the precision it is printed, and every figure from 2026-10-10 keeps them", () => {
  let checked = 0;
  for (const e of history.entries) {
    for (const f of figuresOf(e)) {
      if (e.date >= RUNS_KEPT_FROM) assert.ok(Array.isArray(f.runs), `${e.id}: "${f.printed}" was measured on ${e.date} and keeps no runs`);
      if (!f.runs) continue;
      assert.ok(["median", "mean"].includes(f.summary), `${e.id}: "${f.printed}" keeps runs but its summary is "${f.summary}", not median or mean`);
      assert.equal(f.runs.length, f.n, `${e.id}: "${f.printed}" says ${f.n} runs and keeps ${f.runs.length}`);
      assert.ok(f.runs.every((v) => typeof v === "number" && Number.isFinite(v)), `${e.id}: "${f.printed}" keeps a run that is not a number`);
      const s = summarize(f.runs, f.summary);
      const decimals = printedDecimals(f.printed);
      assert.equal(Number(s.value.toFixed(decimals)), printedValue(f.printed), `${e.id}: the ${f.summary} of its runs is ${s.value}, printed as ${f.printed}`);
      checked++;
    }
  }
  assert.ok(checked >= 15, `only ${checked} figures keep their runs, so this check would pass on almost nothing`);
});

test("the statistics are what they say", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(mean([1, 2, 6]), 3);
  const s = summarize([58.951, 59.2591, 59.2826]);
  assert.equal(s.n, 3);
  assert.equal(s.value, 59.2591);
  assert.equal(s.min, 58.951);
  assert.equal(s.max, 59.2826);
  assert.ok(Math.abs(s.spread - (59.2826 - 58.951) / 59.2591) < 1e-12);
  assert.equal(summarize([5]).spread, 0);
  assert.throws(() => summarize([]));
  assert.throws(() => summarize([1], "mode"));
  assert.equal(printedValue("1,140.1"), 1140.1);
  assert.equal(printedValue("1.03e-3"), 0.00103);
  assert.equal(printedValue("234 of 239"), 234);
  assert.equal(printedDecimals("59.3"), 1);
  assert.equal(printedDecimals("11,249"), 0);
});
