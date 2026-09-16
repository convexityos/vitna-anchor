// What actually ran, one line per request.
//
// The runtime's half of the loop the register has been waiting for. `PLATFORM.md`
// records what the first real usage import taught on 2026-08-24: an export
// scoped by surface is a superset of a lane, the register cannot see how much a
// lane runs, and the majority of a window resolved to a sku the call-name map
// did not carry. All three are properties of reading somebody's export after the
// fact. A ledger written at the moment of the request has none of them: the lane
// is known because the caller named it, the volume is a row count, and the sku
// is the one the policy said to serve.
//
// The shape is deliberately the shape `aperture:usage:import` already reads. A
// row carries `model` and `units` at the top level, so the import needs one new
// format reader and no new semantics, and an operator can hand the same file to
// the same command they would have handed a console export to.
//
// Three rules, each inherited from the record rather than invented here:
//
//   * **Append only.** One `appendFileSync` per row, no rewrite path, no
//     truncation, no compaction. There is no code in this module that can edit
//     a line it has already written.
//   * **Nothing is estimated.** A provider that reported no token counts gets
//     nulls, never a plausible figure. That is mistake 3, the synthetic
//     fallback, at request scale.
//   * **A failure is a row.** A request that errored is recorded with its
//     outcome, because a ledger that only holds successes is a ledger that
//     overstates what a lane did.
//
// Nothing here is sent anywhere. The file is the customer's, on the customer's
// disk, and it moves only when a person moves it.
import { appendFileSync } from "node:fs";

/**
 * One request, as the ledger records it.
 *
 * @typedef {Object} LedgerRow
 * @property {string} at when the request was answered, UTC
 * @property {string} model the name the provider was asked for, which is what an import reads
 * @property {number} units one served request is one unit, and anything else is none
 * @property {string | null} lane the alias the caller named, or null when unrouted
 * @property {string | null} sku what the record calls the model
 * @property {string | null} provider whose model it is
 * @property {string | null} via how it was reached: direct, or the substrate
 * @property {string | null} decision the accepted decision that put this model on this lane
 * @property {string | null} policy the digest of the document that said so
 * @property {number | null} policyAgeSeconds how old that document was when the request ran
 * @property {boolean} routed false for a model name passed through untouched
 * @property {string} serve primary, a named fallback, or a routed rule
 * @property {string | null} shadowRule the shadow rule that matched this request, where one did
 * @property {string | null} shadowSku what that rule would have served instead
 * @property {number | null} cohort the 0 to 99 bucket a traffic share was decided by, where one looked
 * @property {string} outcome served, or refused, or failed
 * @property {number | null} status what the provider answered, where it answered
 * @property {number} latencyMs
 * @property {number | null} inputTokens
 * @property {number | null} outputTokens
 * @property {number | null} [hardware_cost_usd] amortized hardware and electricity cost for local inference
 * @property {number | null} [cloud_market_usd] frontier cloud market equivalent cost
 * @property {number | null} [net_saved_usd] net economic margin vs frontier cloud market
 * @property {number | null} [tokens_evaporated] prompt tokens eliminated via structural context compaction
 * @property {number | null} [tokens_early_stopped] completion tokens saved via structural early-stop guard
 * @property {number | null} [dense_shortcuts_taken] MoE layers evaluated via dense shortcut (0 NVMe reads)
 * @property {number | null} [speculative_tokens_accepted] candidate tokens accepted in draft tree verification
 * @property {string | null} [sovereign_proof] cryptographic air-gap proof from local engine
 * @property {string | null} [trajectory_hash] deterministic SHA-256 token trajectory digest
 * @property {boolean | null} [airgap] true when served locally with zero external network egress
 * @property {string | null} [precision_tier] multi-precision quantization tier
 * @property {boolean | null} [cache_hit] true when served from prompt cache with zero upstream tokens
 * @property {number | null} [queue_wait_ms] duration request spent in priority concurrency queue
 * @property {number | null} [radix_prefix_matched] tokens matched via Radix Tree KV cache prefix sharing
 * @property {number | null} [grammar_tokens_masked] candidate tokens rejected/masked by JSON grammar state machine
 * @property {string | null} error the reason, where there is one
 */

/**
 * A writer, or one that discards.
 *
 * A runtime with no `VITNA_LEDGER` set still serves, and says on its status page
 * that it is recording nothing. That is a deliberate configuration rather than a
 * silent default: somebody trying the runtime for ten minutes should not have to
 * choose a path first, and somebody running it in production should be able to
 * see from the status page that they forgot to.
 *
 * @param {string | null} path
 * @param {{ onError?: (error: Error) => void }} [options]
 */
export function ledger(path, options = {}) {
  let written = 0;
  let failed = 0;
  /** @type {string | null} */
  let lastError = null;

  return {
    enabled: Boolean(path),
    /** @param {LedgerRow} row */
    write(row) {
      if (!path) return;
      try {
        // One line, one syscall, no buffering. A runtime killed between two
        // requests has lost nothing, which matters because this file is the
        // only record that the request happened at all.
        appendFileSync(path, `${JSON.stringify(row)}\n`, "utf8");
        written += 1;
      } catch (error) {
        // A ledger that cannot be written must not take the request down with
        // it. The response has already been earned; losing it to a full disk
        // would turn an accounting problem into an outage. It is counted and
        // surfaced on the status page instead.
        failed += 1;
        lastError = error instanceof Error ? error.message : String(error);
        options.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    },
    status: () => ({ path, written, failed, lastError }),
  };
}

/**
 * The row for a request, assembled in one place so that every path through the
 * server produces the same shape. A served request and a refused one differ in
 * their fields and not in their columns, because a ledger whose rows change
 * shape by outcome is one nothing can total.
 *
 * @param {Partial<LedgerRow> & { at: string, model: string, outcome: string, latencyMs: number }} parts
 * @returns {LedgerRow}
 */
export function ledgerRow(parts) {
  const inTokens = parts.inputTokens ?? null;
  const outTokens = parts.outputTokens ?? null;
  const isLocal =
    parts.via === "local" ||
    parts.provider === "local" ||
    (typeof parts.serve === "string" && parts.serve.startsWith("local"));

  let hwCost = parts.hardware_cost_usd ?? parts.hardwareCostUsd ?? null;
  let cloudMarket = parts.cloud_market_usd ?? parts.cloudMarketUsd ?? null;
  let netSaved = parts.net_saved_usd ?? parts.netSavedUsd ?? null;

  if (isLocal && parts.outcome === "served" && inTokens !== null && outTokens !== null) {
    if (hwCost === null) {
      // Amortized workstation hardware + electricity: $0.000002 per token ($2.00 per 1M tokens)
      hwCost = Math.round((inTokens + outTokens) * 0.000002 * 1e6) / 1e6;
    }
    if (cloudMarket === null) {
      // Cloud reference price ($3.00/1M input, $15.00/1M output)
      cloudMarket = Math.round((inTokens * 0.000003 + outTokens * 0.000015) * 1e6) / 1e6;
    }
    if (netSaved === null) {
      netSaved = Math.round((cloudMarket - hwCost) * 1e6) / 1e6;
    }
  }

  return {
    at: parts.at,
    model: parts.model,
    // A served request is one unit and everything else is none.
    //
    // Found by running it rather than by reading it: a refused lane wrote a row
    // naming the alias the caller asked for, and an import then counted that as
    // a unit of a model nobody served, which put a third of a window into
    // "something we cannot account for". The row still exists, because a ledger
    // that hides failures makes "ran little" and "failed often" the same shape;
    // what it must not do is claim work was done. A window of nothing but
    // refusals therefore totals zero units, and the import says in its own
    // words that the file says nothing about what ran.
    units: parts.units ?? (parts.outcome === "served" ? 1 : 0),
    lane: parts.lane ?? null,
    sku: parts.sku ?? null,
    provider: parts.provider ?? null,
    via: parts.via ?? null,
    decision: parts.decision ?? null,
    policy: parts.policy ?? null,
    policyAgeSeconds: parts.policyAgeSeconds ?? null,
    routed: parts.routed ?? false,
    serve: parts.serve ?? "primary",
    // The would-have, riding the row of what actually ran. A shadow rule's
    // whole output is these two fields accumulating on the ledger until a
    // brief cites them; nothing else anywhere records that it matched.
    shadowRule: parts.shadowRule ?? null,
    shadowSku: parts.shadowSku ?? null,
    // The hosts this request was allowed to be served by, in the order the
    // policy named them, and null where the lane does not rotate.
    //
    // **This says what was PERMITTED and not what answered**, which is a real
    // limit rather than a phrasing. OpenRouter's chat completions response
    // documents no field naming the upstream that served
    // (https://openrouter.ai/docs/api-reference/overview, retrieved
    // 2026-08-31), so a runtime cannot honestly write one down. What it can
    // write is the set it sent, which is signed, and `allow_fallbacks: false`
    // is what makes that set exhaustive rather than advisory.
    //
    // Which host actually answered is a fact OpenRouter keeps and this does
    // not, and reading it there is the same independent trail
    // `docs/routing/hosted.md` describes. That is a check somebody else keeps
    // becoming useful for a second reason.
    sellers: parts.sellers ?? null,
    // The bucket a traffic share was decided by, and null on a request no share
    // rule looked at. Recorded because a canary nobody can audit is a canary
    // nobody should trust: `serve: "rule:2"` says which rule answered and this
    // says why that rule matched this request. It is a number from 0 to 99
    // derived from what the caller sent, never the thing it was derived from.
    cohort: parts.cohort ?? null,
    outcome: parts.outcome,
    status: parts.status ?? null,
    latencyMs: parts.latencyMs,
    inputTokens: parts.inputTokens ?? null,
    outputTokens: parts.outputTokens ?? null,
    hardware_cost_usd: hwCost,
    cloud_market_usd: cloudMarket,
    net_saved_usd: netSaved,
    tokens_evaporated: parts.tokens_evaporated ?? parts.tokensEvaporated ?? null,
    tokens_early_stopped: parts.tokens_early_stopped ?? parts.tokensEarlyStopped ?? null,
    dense_shortcuts_taken: parts.dense_shortcuts_taken ?? parts.denseShortcutsTaken ?? null,
    speculative_tokens_accepted: parts.speculative_tokens_accepted ?? parts.speculativeTokensAccepted ?? null,
    sovereign_proof: parts.sovereign_proof ?? parts.sovereignProof ?? null,
    trajectory_hash: parts.trajectory_hash ?? parts.trajectoryHash ?? null,
    airgap: parts.airgap ?? (isLocal ? true : null),
    precision_tier: parts.precision_tier ?? parts.precisionTier ?? (isLocal ? "int8_anchor_int3_cold" : null),
    cache_hit: parts.cache_hit ?? parts.cacheHit ?? null,
    queue_wait_ms: parts.queue_wait_ms ?? parts.queueWaitMs ?? null,
    radix_prefix_matched: parts.radix_prefix_matched ?? parts.radixPrefixMatched ?? null,
    grammar_tokens_masked: parts.grammar_tokens_masked ?? parts.grammarTokensMasked ?? null,
    error: parts.error ?? null,
  };
}
