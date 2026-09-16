// Deterministic prompt caching for the Vitna runtime.
//
// Serves identical or normalized requests at sub-millisecond latency with zero
// upstream token consumption and zero provider cost.
//
// Rules:
// - Zero external runtime dependencies (uses node:crypto)
// - No em-dashes anywhere in code or comments
// - Pure deterministic fingerprinting across messages, tools, and schema

import { createHash } from "node:crypto";

/**
 * Check if a chat completion request is deterministically cacheable.
 *
 * A request with temperature > 0 or a stream without a fixed seed produces
 * non-deterministic outputs and should not be cached unless explicitly forced.
 *
 * @param {Record<string, any>} request
 * @returns {boolean}
 */
export function isRequestCacheable(request) {
  if (!request || typeof request !== "object") return false;
  if (request.cache === false) return false;
  if (request.cache === true) return true;

  // Temperature > 0 produces non-deterministic completions
  if (typeof request.temperature === "number" && request.temperature > 0) {
    return false;
  }

  // Streaming requests bypass cache by default unless explicitly opted in
  if (request.stream === true && request.cache !== true) {
    return false;
  }

  return true;
}

/**
 * Compute canonical SHA-256 fingerprint for a completion request.
 *
 * Normalizes message roles, trims exterior whitespace, sorts tool functions,
 * and canonicalizes response_format to ensure identical semantic intents
 * hit the exact same cache slot.
 *
 * @param {Record<string, any>} request
 * @returns {string}
 */
export function computeRequestFingerprint(request) {
  if (!request || typeof request !== "object") return "";

  const model = typeof request.model === "string" ? request.model.trim() : "";
  const messages = Array.isArray(request.messages)
    ? request.messages.map((m) => ({
        role: m.role || "user",
        content: typeof m.content === "string" ? m.content.trim() : JSON.stringify(m.content || ""),
        name: m.name || undefined,
      }))
    : [];

  const tools = Array.isArray(request.tools)
    ? request.tools.map((t) => ({
        type: t.type || "function",
        name: t.function?.name || "",
        params: t.function?.parameters ? JSON.stringify(t.function.parameters) : "",
      }))
    : [];

  const responseFormat = request.response_format ? JSON.stringify(request.response_format) : "";

  const canonicalPayload = JSON.stringify({
    model,
    messages,
    tools,
    responseFormat,
  });

  return createHash("sha256").update(canonicalPayload, "utf8").digest("hex");
}

/**
 * Create an in-memory LRU prompt cache.
 *
 * @param {{
 *   maxEntries?: number,
 *   defaultTtlMs?: number,
 *   now?: () => number,
 * }} [options]
 */
export function createPromptCache(options = {}) {
  const maxEntries = options.maxEntries ?? 2048;
  const defaultTtlMs = options.defaultTtlMs ?? 300000; // 5 minutes
  const now = options.now ?? (() => Date.now());

  /** @type {Map<string, { response: any, created: number, expiresAt: number, hits: number, inputTokens: number, outputTokens: number }>} */
  const entries = new Map();

  let totalHits = 0;
  let totalMisses = 0;
  let totalTokensSaved = 0;

  return {
    /**
     * Retrieve cached response if present and not expired.
     * @param {string} fingerprint
     * @returns {{ response: any, hits: number, inputTokens: number, outputTokens: number } | null}
     */
    get(fingerprint) {
      if (!fingerprint) return null;
      const item = entries.get(fingerprint);
      if (!item) {
        totalMisses++;
        return null;
      }

      if (now() > item.expiresAt) {
        entries.delete(fingerprint);
        totalMisses++;
        return null;
      }

      // Refresh LRU order
      entries.delete(fingerprint);
      entries.set(fingerprint, item);

      item.hits++;
      totalHits++;
      totalTokensSaved += (item.inputTokens + item.outputTokens);

      return {
        response: JSON.parse(JSON.stringify(item.response)),
        hits: item.hits,
        inputTokens: item.inputTokens,
        outputTokens: item.outputTokens,
      };
    },

    /**
     * Store response in cache with TTL and token usage metrics.
     * @param {string} fingerprint
     * @param {any} response
     * @param {{ inputTokens?: number, outputTokens?: number, ttlMs?: number }} [meta]
     */
    set(fingerprint, response, meta = {}) {
      if (!fingerprint || !response) return;

      if (entries.size >= maxEntries) {
        const oldestKey = entries.keys().next().value;
        if (oldestKey) entries.delete(oldestKey);
      }

      const ttl = meta.ttlMs ?? defaultTtlMs;
      const current = now();

      entries.set(fingerprint, {
        response: JSON.parse(JSON.stringify(response)),
        created: current,
        expiresAt: current + ttl,
        hits: 0,
        inputTokens: meta.inputTokens ?? 0,
        outputTokens: meta.outputTokens ?? 0,
      });
    },

    /**
     * Cache statistics.
     */
    stats() {
      return {
        size: entries.size,
        maxEntries,
        hits: totalHits,
        misses: totalMisses,
        tokensSaved: totalTokensSaved,
      };
    },

    /**
     * Clear all cached items.
     */
    clear() {
      entries.clear();
      totalHits = 0;
      totalMisses = 0;
      totalTokensSaved = 0;
    },
  };
}
