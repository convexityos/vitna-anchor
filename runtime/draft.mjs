// draft.mjs - Speculative MoE drafting and parallel verification engine.
//
// Fast autoregressive draft prediction combined with parallel target verification
// and rejection sampling. Mirrors the C11 engine contract in engine/src/speculative.c.
//
// Rules:
// - Zero external runtime dependencies
// - Strictly zero em-dashes anywhere in comments, code, or strings
// - Deterministic 32-bit xorshift RNG matching engine/src/speculative.c

const DEFAULT_DRAFT_WINDOW = 4;
const MAX_SPECULATIVE_LOOKAHEAD = 8;

/**
 * Fast deterministic 32-bit xorshift RNG matching C11 core.
 */
export class XorShift32 {
  constructor(seed = 0x853c49e7) {
    this.state = (seed === 0 ? 0x6a09e667 : seed) >>> 0;
  }

  next() {
    let x = this.state;
    x ^= (x << 13) >>> 0;
    x ^= (x >>> 17) >>> 0;
    x ^= (x << 5) >>> 0;
    this.state = x >>> 0;
    return this.state;
  }

  unitFloat() {
    return (this.next() & 0x00ffffff) / 0x01000000;
  }
}

/**
 * Speculative drafter and rejection sampling coordinator.
 */
export class SpeculativeDrafter {
  /**
   * @param {{
   *   lookaheadWindow?: number,
   *   seed?: number,
   * }} options
   */
  constructor(options = {}, maybeSeed) {
    const isNum = typeof options === "number";
    let window = isNum ? options : (options.lookaheadWindow || DEFAULT_DRAFT_WINDOW);
    if (window > MAX_SPECULATIVE_LOOKAHEAD) {
      window = MAX_SPECULATIVE_LOOKAHEAD;
    }
    const seed = isNum ? (maybeSeed || 0x853c49e7) : (options.seed || 0x853c49e7);
    this.lookaheadWindow = window;
    this.rng = new XorShift32(seed);
    this.totalDrafted = 0;
    this.totalAccepted = 0;
    this.verificationPasses = 0;
  }

  /**
   * Propose K candidate tokens from the lightweight draft model.
   * @param {number} currentTokenId
   * @returns {Array<{ tokenId: number, draftProb: number, targetProb: number, accepted: boolean }>}
   */
  propose(currentTokenId) {
    const window = this.lookaheadWindow;
    let runningId = currentTokenId;
    const candidates = [];

    for (let i = 0; i < window; i++) {
      const stepHash = this.rng.next();
      runningId = runningId + 1 + (stepHash % 7);
      const draftProb = 0.75 + (stepHash % 20) / 100.0;

      candidates.push({
        tokenId: runningId,
        draftProb,
        targetProb: 0.0,
        accepted: false,
      });
    }

    this.totalDrafted += window;
    return candidates;
  }

  /**
   * Parallel verification and rejection sampling pass against target model distribution.
   * @param {Array<{ tokenId: number, draftProb: number, targetProb?: number, accepted?: boolean }>} candidates
   * @returns {{
   *   acceptedCount: number,
   *   acceptedTokens: number[],
   *   correctionToken: number,
   *   rejectionIndex: number,
   *   emittedTokens: number[],
   * }}
   */
  verify(candidates) {
    if (!candidates || candidates.length === 0) {
      return {
        acceptedCount: 0,
        acceptedTokens: [],
        correctionToken: 0,
        rejectionIndex: -1,
        emittedTokens: [],
      };
    }

    this.verificationPasses++;
    let nAccepted = 0;
    let rejectionIndex = -1;
    let correctionToken = 0;
    const acceptedTokens = [];

    let lastEmittedToken = candidates[0].tokenId;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      let pTarget = c.targetProb || 0.0;
      if (pTarget <= 0.0) {
        pTarget = c.draftProb * (0.85 + (this.rng.next() % 30) / 100.0);
        if (pTarget > 1.0) pTarget = 0.98;
        c.targetProb = pTarget;
      }

      // Speculative rejection criterion: min(1, p_target / p_draft)
      const ratio = pTarget / (c.draftProb > 0.001 ? c.draftProb : 0.001);
      const r = this.rng.unitFloat();

      if (ratio >= 1.0 || r <= ratio) {
        c.accepted = true;
        nAccepted++;
        acceptedTokens.push(c.tokenId);
        lastEmittedToken = c.tokenId;
      } else {
        c.accepted = false;
        rejectionIndex = i;
        const correctionDelta = (this.rng.next() % 15) + 1;
        correctionToken = c.tokenId + correctionDelta;
        lastEmittedToken = correctionToken;
        break;
      }
    }

    this.totalAccepted += nAccepted;

    const emittedTokens = [...acceptedTokens];
    if (rejectionIndex >= 0) {
      emittedTokens.push(correctionToken);
    }

    return {
      acceptedCount: nAccepted,
      acceptedTokens,
      correctionToken,
      rejectionIndex,
      emittedTokens,
      emittedTokenId: lastEmittedToken,
    };
  }

  /**
   * Empirical acceptance rate (accepted tokens / total drafted tokens).
   * @returns {number}
   */
  acceptanceRate() {
    if (this.totalDrafted === 0) return 0.0;
    return Number((this.totalAccepted / this.totalDrafted).toFixed(4));
  }

  /**
   * Effective speedup factor over baseline autoregressive decoding.
   * @returns {number}
   */
  speedupFactor() {
    if (this.verificationPasses === 0) return 1.0;
    const alpha = this.totalAccepted / this.verificationPasses;
    const draftOverhead = 0.05 * this.lookaheadWindow;
    const speedup = (alpha + 1.0) / (1.0 + draftOverhead);
    return Number(Math.max(1.0, speedup).toFixed(2));
  }
}

/**
 * Vocabulary mock table for deterministic interactive demo token text.
 */
const SAMPLE_VOCAB = [
  " sovereign", " inference", " hardware", " direct", " NVMe", " DMA",
  " unbuffered", " sector", " 4KB", " alignment", " low-latency", " MoE",
  " expert", " routing", " zero-copy", " slab", " cache", " throughput",
  " attestation", " proof", " bare-metal", " execution", " pipeline", " parallel",
  " acceleration", " speedup", " memory", " bandwidth", " stream", " tensor"
];

/**
 * Map token id to sample readable token text for terminal display.
 * @param {number} tokenId
 * @returns {string}
 */
export function decodeMockToken(tokenId) {
  const idx = Math.abs(tokenId) % SAMPLE_VOCAB.length;
  return SAMPLE_VOCAB[idx];
}

/**
 * Run a multi-turn speculative generation simulation.
 *
 * @param {string} prompt
 * @param {{
 *   targetModel?: string,
 *   draftModel?: string,
 *   lookaheadWindow?: number,
 *   turns?: number,
 *   seed?: number,
 * }} options
 * @returns {{
 *   prompt: string,
 *   targetModel: string,
 *   draftModel: string,
 *   lookaheadWindow: number,
 *   steps: Array<{
 *     turn: number,
 *     candidates: Array<{ tokenId: number, text: string, draftProb: number, targetProb: number, accepted: boolean }>,
 *     acceptedCount: number,
 *     emittedTokens: Array<{ tokenId: number, text: string }>,
 *     rejectionIndex: number,
 *   }>,
 *   totalDrafted: number,
 *   totalAccepted: number,
 *   acceptanceRate: number,
 *   verificationPasses: number,
 *   speedupFactor: number,
 *   generatedText: string,
 * }}
 */
export function runSpeculativeGeneration(prompt, options = {}) {
  const targetModel = options.targetModel || "vitna/anchor-moe-70b";
  const draftModel = options.draftModel || "vitna/anchor-draft-1b";
  const lookaheadWindow = options.lookaheadWindow || DEFAULT_DRAFT_WINDOW;
  const turns = options.turns || 5;

  const drafter = new SpeculativeDrafter({
    lookaheadWindow,
    seed: options.seed || 0x853c49e7,
  });

  let currentTokenId = 1000;
  const steps = [];
  const emittedAll = [];

  for (let turn = 0; turn < turns; turn++) {
    const candidates = drafter.propose(currentTokenId);
    const result = drafter.verify(candidates);

    const stepCandidates = candidates.map((c) => ({
      tokenId: c.tokenId,
      text: decodeMockToken(c.tokenId),
      draftProb: Number(c.draftProb.toFixed(3)),
      targetProb: Number(c.targetProb.toFixed(3)),
      accepted: c.accepted,
    }));

    const stepEmitted = result.emittedTokens.map((id) => ({
      tokenId: id,
      text: decodeMockToken(id),
    }));

    steps.push({
      turn: turn + 1,
      candidates: stepCandidates,
      acceptedCount: result.acceptedCount,
      emittedTokens: stepEmitted,
      rejectionIndex: result.rejectionIndex,
    });

    for (const em of stepEmitted) {
      emittedAll.push(em.text);
    }

    if (result.emittedTokens.length > 0) {
      currentTokenId = result.emittedTokens[result.emittedTokens.length - 1];
    }
  }

  return {
    prompt,
    targetModel,
    draftModel,
    lookaheadWindow,
    steps,
    totalDrafted: drafter.totalDrafted,
    totalAccepted: drafter.totalAccepted,
    acceptanceRate: drafter.acceptanceRate(),
    verificationPasses: drafter.verificationPasses,
    speedupFactor: drafter.speedupFactor(),
    generatedText: emittedAll.join("").trim(),
  };
}
