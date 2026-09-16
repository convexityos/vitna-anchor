import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluateSmartOrderRoute,
  resolveModelFamily,
  calculateCallCost,
} from "../runtime/arbitrage.mjs";

test("resolveModelFamily matches model aliases correctly", () => {
  assert.equal(resolveModelFamily("meta-llama/llama-3.3-70b-instruct"), "llama-3.3-70b");
  assert.equal(resolveModelFamily("deepseek/deepseek-chat"), "deepseek-v3");
  assert.equal(resolveModelFamily("qwen/qwen-2.5-72b-instruct"), "qwen-2.5-72b");
  assert.equal(resolveModelFamily("unknown/custom-model"), null);
});

test("calculateCallCost accurately computes token cost in USD", () => {
  // 1,000 input tokens @ $0.50/M + 500 output tokens @ $1.00/M
  // inCost = 0.0005, outCost = 0.0005 -> total = 0.001
  const cost = calculateCallCost(1000, 500, 0.50, 1.00);
  assert.equal(cost, 0.001);
});

test("evaluateSmartOrderRoute selects the lowest-cost provider and computes savings", () => {
  const result = evaluateSmartOrderRoute({
    model: "meta-llama/llama-3.3-70b-instruct",
    inputTokens: 2000,
    outputTokens: 500,
  });

  assert.equal(result.ok, true);
  assert.equal(result.family, "llama-3.3-70b");
  assert.equal(result.chosenProvider, "deepinfra", "cheapest provider selected");
  assert.ok(result.costUsd > 0, "positive cost computed");
  assert.ok(result.medianMarketCostUsd > result.costUsd, "cheaper than market median");
  assert.ok(result.savingsUsd > 0, "positive net savings calculated");
  assert.ok(result.savingsPct > 40, `savings percentage should be >40%, got ${result.savingsPct}%`);
  assert.ok(result.fallbackChain.length >= 3, "fallback chain holds alternate sellers");
  assert.equal(result.fallbackChain[0].provider, "together", "second cheapest is first fallback");
});

test("evaluateSmartOrderRoute respects quantization constraints", () => {
  const result = evaluateSmartOrderRoute({
    model: "meta-llama/llama-3.3-70b-instruct",
    inputTokens: 1000,
    outputTokens: 200,
    allowedQuantizations: ["bf16"], // Only full precision admitted
  });

  assert.equal(result.ok, true);
  assert.equal(result.quantization, "bf16");
  assert.equal(result.chosenProvider, "openrouter");
});
