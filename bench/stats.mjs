// The statistics bench/history.json's entries summarize their runs with, and
// the spread a reader needs beside them: how many runs, the least and the
// most, and the spread as a share of the figure. tests/bench-history.test.mjs
// recomputes each entry's figure with these, and anything that shows the
// history can print the spread from them rather than work it out again.

/** The median of values: the middle one, or the mean of the middle two. */
export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function mean(values) {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * A figure's runs summarized: n, the statistic it names (median or mean),
 * the least and the most, and the spread, (most - least) / statistic, as a
 * fraction. With one run the spread is 0 and says nothing.
 */
export function summarize(runs, stat = "median") {
  if (!Array.isArray(runs) || runs.length === 0) throw new Error("summarize needs at least one run");
  if (stat !== "median" && stat !== "mean") throw new Error(`no statistic ${stat}`);
  const value = stat === "median" ? median(runs) : mean(runs);
  const min = Math.min(...runs);
  const max = Math.max(...runs);
  return { n: runs.length, stat, value, min, max, spread: value ? (max - min) / value : 0 };
}

/** A printed figure as a number: "1,140.1" is 1140.1, "1.03e-3" is 0.00103. */
export function printedValue(printed) {
  const m = /^-?[\d,]+(\.\d+)?(e-?\d+)?/.exec(printed);
  return m ? Number(m[0].replace(/,/g, "")) : NaN;
}

/** How many decimals a printed figure carries: "59.3" has 1, "11,249" has 0. */
export function printedDecimals(printed) {
  const m = /^-?[\d,]+(?:\.(\d+))?/.exec(printed);
  return m && m[1] ? m[1].length : 0;
}
