/**
 * compaction.mjs - Context Evaporator and Prompt Compaction
 *
 * Slashing billable token waste and latency before requests reach models:
 * 1. Deduplicates redundant system/instruction turns
 * 2. Compresses verbose JSON schemas and tool definitions
 * 3. Strips repetitive whitespace padding and conversational boilerplate
 *
 * Rules:
 * - Pure deterministic functions with zero external dependencies
 * - No em-dashes anywhere in comments, code, or strings
 * - Never modifies user code blocks, strings, or semantic intents
 */

/**
 * Compact a single text string by collapsing redundant multi-newline runs
 * and stripping edge whitespace outside of code blocks.
 *
 * @param {string} text
 * @returns {string}
 */
export function compactText(text) {
  if (typeof text !== "string") return text;
  if (!text.includes("```")) {
    // Normal prose: collapse 3+ newlines into 2, strip trailing whitespace per line
    return text
      .split("\n")
      .map((line) => line.trimEnd())
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  // Text contains markdown code fences: preserve contents inside fences intact
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts
    .map((part) => {
      if (part.startsWith("```")) return part; // inside code fence: keep verbatim
      return part
        .split("\n")
        .map((line) => line.trimEnd())
        .join("\n")
        .replace(/\n{3,}/g, "\n\n");
    })
    .join("")
    .trim();
}

/**
 * Compact a tool definition schema by stripping empty descriptions and redundant fields.
 *
 * @param {Record<string, unknown>} tool
 * @returns {Record<string, unknown>}
 */
export function compactToolDefinition(tool) {
  if (!tool || typeof tool !== "object") return tool;
  const clone = JSON.parse(JSON.stringify(tool));

  function cleanSchema(obj) {
    if (!obj || typeof obj !== "object") return;
    for (const key of Object.keys(obj)) {
      if (key === "description" && typeof obj[key] === "string" && !obj[key].trim()) {
        delete obj[key];
      } else if (key === "additionalProperties" && obj[key] === false) {
        delete obj[key];
      } else if (typeof obj[key] === "object") {
        cleanSchema(obj[key]);
      }
    }
  }

  if (clone.function) {
    cleanSchema(clone.function);
  } else {
    cleanSchema(clone);
  }
  return clone;
}

/**
 * Pre-process incoming messages array, evaporating syntactic bloat while
 * strictly preserving conversation semantics and roles.
 *
 * @param {Array<Record<string, unknown>>} messages
 * @returns {Array<Record<string, unknown>>}
 */
export function compactMessages(messages) {
  if (!Array.isArray(messages)) return messages;

  const compacted = [];
  let lastRole = null;
  let lastContent = null;

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;

    let content = msg.content;
    if (typeof content === "string") {
      content = compactText(content);
    } else if (Array.isArray(content)) {
      content = content.map((part) => {
        if (part?.type === "text" && typeof part.text === "string") {
          return { ...part, text: compactText(part.text) };
        }
        return part;
      });
    }

    // Deduplicate identical consecutive messages
    if (msg.role === lastRole && typeof content === "string" && content === lastContent) {
      continue;
    }

    // Merge different consecutive system messages into a single consolidated system turn
    if (msg.role === "system" && compacted.length > 0 && compacted[compacted.length - 1].role === "system") {
      const prev = compacted[compacted.length - 1];
      if (typeof prev.content === "string" && typeof content === "string") {
        prev.content = `${prev.content}\n\n${content}`.trim();
        lastContent = prev.content;
        continue;
      }
    }

    lastRole = msg.role;
    lastContent = typeof content === "string" ? content : null;

    compacted.push({
      ...msg,
      content,
    });
  }

  return compacted;
}

/**
 * Compute character and token savings from compaction.
 *
 * @param {unknown} original
 * @param {unknown} compacted
 * @returns {{ originalChars: number, compactedChars: number, charsSaved: number, tokensSaved: number, savingsPct: number }}
 */
export function calculateCompactionSavings(original, compacted) {
  const origStr = typeof original === "string" ? original : JSON.stringify(original || "");
  const compStr = typeof compacted === "string" ? compacted : JSON.stringify(compacted || "");

  const originalChars = origStr.length;
  const compactedChars = compStr.length;
  const charsSaved = Math.max(0, originalChars - compactedChars);
  const tokensSaved = Math.round(charsSaved / 4);
  const savingsPct = originalChars > 0 ? Number(((charsSaved / originalChars) * 100).toFixed(1)) : 0;

  return {
    originalChars,
    compactedChars,
    charsSaved,
    tokensSaved,
    savingsPct,
  };
}
