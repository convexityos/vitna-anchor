// Turns a log from scripts/record-api-requests.mjs into a recorded-shapes
// fixture (reference/api-shapes/): each request's method, path, describing
// headers and body, with the content taken out and the shape kept.
//
//   node scripts/api-shapes.mjs <log.jsonl> <client> <version> > reference/api-shapes/<client>-<version>.json
//
// Every string longer than 32 characters becomes "<n chars>": prompts, tool
// descriptions, file contents and the client's own instructions, which are
// its vendor's text and not this repository's to copy. Shorter strings, which
// are the shape (types, roles, tool and property names, enums, ids), stay. The
// fields that identify a person, a machine or a session (Anthropic's
// metadata, Codex's client_metadata and prompt_cache_key) are dropped whole.

import fs from "node:fs";

const [, , logFile, client, version] = process.argv;
if (!version) {
  console.error("usage: node scripts/api-shapes.mjs <log.jsonl> <client> <version>");
  process.exit(2);
}

const DROP = new Set(["metadata", "client_metadata", "prompt_cache_key"]);
const LONGEST = 32;

function shape(v) {
  if (typeof v === "string") return v.length > LONGEST ? `<${v.length} chars>` : v;
  if (Array.isArray(v)) return v.map(shape);
  if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, shape(x)]));
  return v;
}

const requests = fs
  .readFileSync(logFile, "utf8")
  .trim()
  .split("\n")
  .map((l) => JSON.parse(l))
  .filter((r) => r.method)
  .map((r) => {
    const body = r.body && Object.fromEntries(Object.entries(r.body).filter(([k]) => !DROP.has(k)));
    return { method: r.method, path: r.url, headers: r.headers, body: body ? shape(body) : null };
  });

const out = {
  format: "vitna-anchor recorded API shapes, version 1",
  client,
  version,
  recorded: new Date().toISOString().slice(0, 10),
  how: `Recorded with scripts/record-api-requests.mjs between ${client} ${version} and this server, and made into this file by scripts/api-shapes.mjs: strings longer than ${LONGEST} characters are replaced by their length, and metadata, client_metadata and prompt_cache_key are dropped.`,
  requests,
};
process.stdout.write(JSON.stringify(out, null, 2) + "\n");
