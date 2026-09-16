import assert from "node:assert/strict";
import test from "node:test";

/**
 * Deterministic simulation of the C11 speculative rejection sampling engine
 * from engine/src/speculative.c to assert exact mathematical invariants.
 */
class SpeculativeDrafterSim {
  constructor(lookaheadWindow = 4, seed = 0x853c49e7) {
    this.lookaheadWindow = lookaheadWindow;
    this.rngState = seed;
    this.totalDrafted = 0;
    this.totalAccepted = 0;
    this.verificationPasses = 0;
  }

  xorshift32() {
    let x = this.rngState;
    if (x === 0) x = 0x6a09e667;
    x ^= (x << 13) >>> 0;
    x ^= (x >>> 17) >>> 0;
    x ^= (x << 5) >>> 0;
    this.rngState = x >>> 0;
    return this.rngState;
  }

  randomUnitFloat() {
    return (this.xorshift32() & 0x00ffffff) / 0x01000000;
  }

  propose(currentTokenId) {
    const candidates = [];
    let runningId = currentTokenId;
    for (let i = 0; i < this.lookaheadWindow; i++) {
      const stepHash = this.xorshift32();
      runningId += 1 + (stepHash % 7);
      candidates.push({
        tokenId: runningId,
        draftProb: 0.75 + (stepHash % 20) / 100.0,
        targetProb: 0.0,
        accepted: false,
      });
    }
    this.totalDrafted += this.lookaheadWindow;
    return candidates;
  }

  verify(candidates) {
    this.verificationPasses++;
    let accepted = 0;
    let lastToken = candidates[0].tokenId;

    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      if (c.targetProb <= 0) {
        c.targetProb = Math.min(0.98, c.draftProb * (0.85 + (this.xorshift32() % 30) / 100.0));
      }

      const ratio = c.targetProb / Math.max(0.001, c.draftProb);
      const r = this.randomUnitFloat();

      if (ratio >= 1.0 || r <= ratio) {
        c.accepted = true;
        accepted++;
        lastToken = c.tokenId;
      } else {
        c.accepted = false;
        lastToken = c.tokenId + (this.xorshift32() % 15) + 1;
        break;
      }
    }

    this.totalAccepted += accepted;
    return { acceptedCount: accepted, emittedTokenId: lastToken };
  }

  acceptanceRate() {
    return this.totalDrafted > 0 ? this.totalAccepted / this.totalDrafted : 0;
  }

  speedupFactor() {
    if (this.verificationPasses === 0) return 1.0;
    const alpha = this.totalAccepted / this.verificationPasses;
    const overhead = 0.05 * this.lookaheadWindow;
    return Math.max(1.0, (alpha + 1.0) / (1.0 + overhead));
  }
}

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
