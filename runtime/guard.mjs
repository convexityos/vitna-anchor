/**
 * guard.mjs - Structural Early-Stop Guard
 *
 * Prevents output token waste by detecting structural completion
 * (e.g. balanced JSON braces, closing code fences) and stopping generation
 * before the model generates trailing conversational apologies or filler.
 *
 * Rules:
 * - Pure deterministic state machine with zero external dependencies
 * - No em-dashes anywhere in comments, code, or strings
 */

/**
 * Creates a structural delimiter tracker for streaming completions.
 *
 * @param {object} [options]
 * @param {boolean} [options.stopOnJsonRootClose=true] Stop when top-level JSON object/array closes
 * @param {string[]} [options.stopDelimiters=[]] Specific substrings that trigger immediate stop
 */
export function createStructuralStopGuard(options = {}) {
  const stopOnJson = options.stopOnJsonRootClose !== false;
  const customDelimiters = Array.isArray(options.stopDelimiters) ? options.stopDelimiters : [];

  let buffer = "";
  let inString = false;
  let escapeNext = false;
  let braceDepth = 0;
  let bracketDepth = 0;
  let hasEnteredStructure = false;
  let completed = false;
  let stopReason = null;

  return {
    /**
     * Feed a streaming text chunk into the tracker.
     * @param {string} delta
     * @returns {{ shouldStop: boolean, reason: string | null }}
     */
    feedChunk(delta) {
      if (completed || typeof delta !== "string") {
        return { shouldStop: completed, reason: stopReason };
      }

      buffer += delta;

      // Check custom stop delimiters
      for (const delim of customDelimiters) {
        if (buffer.includes(delim)) {
          completed = true;
          stopReason = `custom_delimiter: "${delim}"`;
          return { shouldStop: true, reason: stopReason };
        }
      }

      if (!stopOnJson) {
        return { shouldStop: false, reason: null };
      }

      // Track JSON structure balance
      for (let i = buffer.length - delta.length; i < buffer.length; i++) {
        const ch = buffer[i];

        if (escapeNext) {
          escapeNext = false;
          continue;
        }

        if (ch === "\\") {
          escapeNext = true;
          continue;
        }

        if (ch === '"') {
          inString = !inString;
          continue;
        }

        if (inString) continue;

        if (ch === "{") {
          braceDepth++;
          hasEnteredStructure = true;
        } else if (ch === "}") {
          if (braceDepth > 0) braceDepth--;
          if (hasEnteredStructure && braceDepth === 0 && bracketDepth === 0) {
            completed = true;
            stopReason = "json_root_object_closed";
            return { shouldStop: true, reason: stopReason };
          }
        } else if (ch === "[") {
          bracketDepth++;
          hasEnteredStructure = true;
        } else if (ch === "]") {
          if (bracketDepth > 0) bracketDepth--;
          if (hasEnteredStructure && braceDepth === 0 && bracketDepth === 0) {
            completed = true;
            stopReason = "json_root_array_closed";
            return { shouldStop: true, reason: stopReason };
          }
        }
      }

      return { shouldStop: false, reason: null };
    },

    isComplete() {
      return completed;
    },

    getReason() {
      return stopReason;
    },

    getBuffer() {
      return buffer;
    },
  };
}

/**
 * Trim trailing conversational filler after a valid closing JSON object.
 *
 * @param {string} text
 * @returns {string}
 */
export function trimTrailingChatter(text) {
  if (typeof text !== "string") return text;
  const trimmed = text.trim();

  // If text starts with '{' and has a matching '}', prune anything following it
  if (trimmed.startsWith("{")) {
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = 0; i < trimmed.length; i++) {
      const ch = trimmed[i];
      if (esc) { esc = false; continue; }
      if (ch === "\\") { esc = true; continue; }
      if (ch === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          return trimmed.slice(0, i + 1);
        }
      }
    }
  }
  return text;
}
