// Writes engine/src/chat_page.c from engine/web/chat.html: the chat page the
// server answers GET / with, built into the engine so that it needs no file
// beside it. Edit the page, then run this; tests/chat-page.test.mjs fails if
// the two differ.
//
//   node engine/tools/embed_page.mjs           # write chat_page.c
//   node engine/tools/embed_page.mjs --check   # exit 1 if chat_page.c is not the page's

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ENGINE = join(dirname(fileURLToPath(import.meta.url)), "..");
export const PAGE = join(ENGINE, "web", "chat.html");
export const SOURCE = join(ENGINE, "src", "chat_page.c");

/* A line of the page as a C string literal: a backslash, a quote and a
 * question mark (which could begin a trigraph) escaped, every byte outside
 * printable ASCII in octal, and the line's newline kept. */
function literal(bytes, newline) {
  let s = '    "';
  for (const b of bytes) {
    if (b === 0x5c) s += "\\\\";
    else if (b === 0x22) s += '\\"';
    else if (b === 0x3f) s += "\\?";
    else if (b === 0x09) s += "\\t";
    else if (b >= 0x20 && b < 0x7f) s += String.fromCharCode(b);
    else s += "\\" + b.toString(8).padStart(3, "0");
  }
  return s + (newline ? "\\n" : "") + '"';
}

/** text as C string literals, a line of it to each, which C joins into one string. */
export function cStringLines(text) {
  const bytes = Buffer.from(text, "utf8");
  const lines = [];
  let start = 0;
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] === 0x0a) {
      lines.push(literal(bytes.subarray(start, i), true));
      start = i + 1;
    }
  }
  if (start < bytes.length) lines.push(literal(bytes.subarray(start), false));
  return lines.join("\n");
}

export function embedPage(html) {
  return [
    "/**",
    " * chat_page.c - The chat page the server answers GET / with: engine/web/chat.html,",
    " * written here by engine/tools/embed_page.mjs. Edit the page and run that; do",
    " * not edit this file.",
    " */",
    "",
    "#include <stddef.h>",
    "",
    "const char vitna_chat_page[] =",
    cStringLines(html) + ";",
    "",
    "const size_t vitna_chat_page_len = sizeof(vitna_chat_page) - 1;",
    "",
  ].join("\n");
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const html = readFileSync(PAGE, "utf8");
  if (html.includes("\r")) {
    console.error(`${PAGE} has carriage returns; the page is kept with LF line endings`);
    process.exit(1);
  }
  const want = embedPage(html);
  if (process.argv.includes("--check")) {
    let have = "";
    try {
      have = readFileSync(SOURCE, "utf8");
    } catch {}
    if (have !== want) {
      console.error(`${SOURCE} is not ${PAGE}: run node engine/tools/embed_page.mjs`);
      process.exit(1);
    }
    console.log("chat_page.c is the page");
  } else {
    writeFileSync(SOURCE, want);
    console.log(`wrote ${SOURCE}: ${Buffer.byteLength(html)} bytes of page`);
  }
}
