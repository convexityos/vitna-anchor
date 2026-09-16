import assert from "node:assert/strict";
import test from "node:test";

import {
  SpeculativeDrafter,
  runSpeculativeGeneration,
} from "../runtime/draft.mjs";

const SpeculativeDrafterSim = SpeculativeDrafter;

test("speculative drafting proposes exactly lookahead_window candidate tokens", () => {
  const drafter = new SpeculativeDrafterSim(4);
  const candidates = drafter.propose(100);
  assert.equal(candidates.length, 4);
  assert.equal(drafter.totalDrafted, 4);
  for (const c of candidates) {
    assert.ok(c.tokenId > 100);
    assert.ok(c.draftProb >= 0.75 && c.draftProb <= 0.95);
    assert.equal(c.accepted, false);
  }
});

test("speculative rejection sampling accepts candidates when p_target >= p_draft", () => {
  const drafter = new SpeculativeDrafterSim(4);
  const candidates = [
    { tokenId: 101, draftProb: 0.80, targetProb: 0.90, accepted: false },
    { tokenId: 102, draftProb: 0.70, targetProb: 0.85, accepted: false },
    { tokenId: 103, draftProb: 0.75, targetProb: 0.80, accepted: false },
    { tokenId: 104, draftProb: 0.60, targetProb: 0.65, accepted: false },
  ];

  const result = drafter.verify(candidates);
  assert.equal(result.acceptedCount, 4);
  assert.equal(result.emittedTokenId, 104);
  for (const c of candidates) {
    assert.equal(c.accepted, true);
  }
});

test("rejection terminates the candidate prefix and emits correction token", () => {
  const drafter = new SpeculativeDrafterSim(4);
  const candidates = [
    { tokenId: 201, draftProb: 0.80, targetProb: 0.90, accepted: false }, // accepted (ratio >= 1)
    { tokenId: 202, draftProb: 0.99, targetProb: 0.0001, accepted: false }, // rejected (ratio near 0)
    { tokenId: 203, draftProb: 0.80, targetProb: 0.90, accepted: false }, // discarded
    { tokenId: 204, draftProb: 0.80, targetProb: 0.90, accepted: false }, // discarded
  ];

  const result = drafter.verify(candidates);
  assert.equal(result.acceptedCount, 1, "only first candidate accepted before rejection");
  assert.equal(candidates[0].accepted, true);
  assert.equal(candidates[1].accepted, false);
  assert.equal(candidates[2].accepted, false);
  assert.notEqual(result.emittedTokenId, 201);
  assert.notEqual(result.emittedTokenId, 202);
});

test("multi-turn speculative execution achieves >2.0x empirical speedup over baseline", () => {
  const drafter = new SpeculativeDrafterSim(4);
  let currentToken = 500;

  for (let step = 0; step < 50; step++) {
    const candidates = drafter.propose(currentToken);
    const result = drafter.verify(candidates);
    currentToken = result.emittedTokenId;
  }

  const rate = drafter.acceptanceRate();
  const speedup = drafter.speedupFactor();

  assert.ok(rate > 0.65, `acceptance rate should be >65%, got ${(rate * 100).toFixed(1)}%`);
  assert.ok(speedup > 2.0, `speculative speedup should be >2.0x, got ${speedup.toFixed(2)}x`);
});

test("runSpeculativeGeneration produces multi-step verification telemetry and speedup metrics", () => {
  const sim = runSpeculativeGeneration("Explain NVMe DMA slab slicing", {
    lookaheadWindow: 4,
    turns: 6,
  });

  assert.equal(sim.lookaheadWindow, 4);
  assert.equal(sim.steps.length, 6);
  assert.ok(sim.totalDrafted >= 24);
  assert.ok(sim.totalAccepted > 0);
  assert.ok(sim.speedupFactor >= 1.5);
  assert.ok(sim.generatedText.length > 0);

  for (const step of sim.steps) {
    assert.equal(step.candidates.length, 4);
    assert.ok(step.emittedTokens.length > 0);
  }
});
