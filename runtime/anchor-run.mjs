#!/usr/bin/env node
// Command line for vitna-anchor.
//
// No model runs yet. `serve` answers every generation request with 501 and
// `chat` says so and exits. The other commands work on files: they parse,
// slice, stripe and quantize checkpoints, and compile JSON Schemas.
//
// Nothing this command line prints is a performance figure or a claim about
// the machine. A figure appears only once a gate in the README measures it.
//
// Zero external dependencies.

import { createServer } from "node:http";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { runModelPull } from "./ingest.mjs";
import { quantizeSlabFile } from "./quantize.mjs";
import { sliceStripedDmaSlabs, formatStripeSummary } from "./stripe.mjs";
import { compileJsonSchemaToPda, formatGrammarSummary } from "./grammar.mjs";

export const NO_MODEL_MESSAGE =
  "No model runs yet. This build of vitna-anchor has no forward pass, so it cannot generate text. " +
  "See the gate ladder in the README.";

// Commands earlier versions offered, and why each one went. Calling one prints
// its reason rather than a bare "unknown command".
export const REMOVED_COMMANDS = {
  probe: "it printed projected tokens per second and an air-gap result that nothing had measured",
  route: "it ranked cloud providers by a price table with no source",
  draft: "it simulated speculative decoding without a model to draft from",
  bench: "it computed prefetch and striping throughput from constants rather than measuring it",
  tune: "it recommended settings from reads of a file it had just written, which the page cache serves",
  registry: "it listed models as verified that nothing had verified",
  models: "it listed models as verified that nothing had verified",
};

const ANSI = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  amber: "\x1b[38;2;245;158;11m",
  green: "\x1b[38;2;34;197;94m",
  gray: "\x1b[38;2;156;163;175m",
  hairline: "\x1b[38;2;75;85;99m",
};

function rule(title = "") {
  const width = 74;
  if (!title) return ANSI.hairline + "─".repeat(width) + ANSI.reset;
  const prefix = `── [ ${title} ] `;
  const rem = Math.max(0, width - prefix.length);
  return ANSI.hairline + prefix + "─".repeat(rem) + ANSI.reset;
}

function printUsage() {
  console.log("\n" + ANSI.bold + "vitna-anchor" + ANSI.reset);
  console.log(ANSI.dim + NO_MODEL_MESSAGE + "\n" + ANSI.reset);
  console.log("Usage:");
  console.log("  vitna-anchor pull     <model-id-or-path> [--out <dir>] [--dry-run]");
  console.log("  vitna-anchor quantize <checkpoint> [--bits 4|8] [--out <dir>] [--json]");
  console.log("  vitna-anchor stripe   <file> --drives <d1,d2,...> [--chunk-kb 4] [--json]");
  console.log("  vitna-anchor schema   <schema.json | inline JSON> [--json]");
  console.log("  vitna-anchor serve    [--port <port>] [--host <ip>]");
  console.log("  vitna-anchor chat\n");
  console.log("Commands:");
  console.log("  pull      Read a SafeTensors or GGUF checkpoint, from disk or the Hugging Face Hub,");
  console.log("            and rewrite it with every tensor at a 4096-byte offset");
  console.log("  quantize  Quantize a checkpoint's tensors to block-wise INT4 or INT8");
  console.log("  stripe    Split a file round-robin into chunks across several directories");
  console.log("  schema    Compile a JSON Schema into the automaton a constrained decoder would use");
  console.log("  serve     Start an HTTP server that answers generation requests with 501");
  console.log("  chat      Say that no model runs yet, and exit\n");
  console.log("Environment variables:");
  console.log("  VITNA_ANCHOR_PORT    Server listen port (default 8765)");
  console.log("  VITNA_ANCHOR_HOST    Server listen host (default 127.0.0.1)\n");
}

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

/**
 * Start the local HTTP server. It states that no model runs: generation
 * endpoints answer 501, and the model list is empty.
 * @param {{ host?: string, port?: number, quiet?: boolean }} options
 */
export function startAnchorServer({ host = "127.0.0.1", port = 8765, quiet = false } = {}) {
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");

    // Read and discard any request body before answering, so the client
    // gets the response rather than a reset connection.
    req.resume();
    req.on("end", () => {
      if (req.method === "GET" && (url.pathname === "/health" || url.pathname === "/v1/health")) {
        sendJson(res, 200, {
          ok: true,
          engine: "vitna-anchor",
          model: null,
          generation: false,
          message: NO_MODEL_MESSAGE,
        });
        return;
      }

      if (req.method === "GET" && url.pathname === "/v1/models") {
        sendJson(res, 200, { object: "list", data: [] });
        return;
      }

      if (
        req.method === "POST" &&
        (url.pathname === "/v1/chat/completions" ||
          url.pathname === "/v1/completions" ||
          url.pathname === "/v1/embeddings")
      ) {
        sendJson(res, 501, {
          error: { message: NO_MODEL_MESSAGE, type: "not_implemented", code: "no_model" },
        });
        return;
      }

      sendJson(res, 404, { error: { message: "Not found", type: "invalid_request_error", code: "not_found" } });
    });
  });

  server.listen(port, host, () => {
    if (quiet) return;
    const { port: boundPort } = server.address();
    console.log("\n" + rule("VITNA ANCHOR SERVER"));
    console.log(`  Listening on      : ${ANSI.bold}http://${host}:${boundPort}${ANSI.reset}`);
    console.log(`  Model             : ${ANSI.amber}none${ANSI.reset}`);
    console.log(`  Generation        : answers 501 on /v1/chat/completions and /v1/completions\n`);
    console.log(ANSI.dim + NO_MODEL_MESSAGE + "\n" + ANSI.reset);
  });

  return server;
}

/**
 * Quantize a checkpoint's tensors into 4096-byte aligned INT4 or INT8 slabs.
 * @param {string} inputPath
 * @param {{ bits?: number, out?: string, isJson?: boolean }} options
 */
export function runQuantizeCli(inputPath, options = {}) {
  if (!inputPath) {
    console.error(ANSI.amber + "Error: a checkpoint path is required. Usage: vitna-anchor quantize <checkpoint> [--bits 4|8]" + ANSI.reset);
    process.exit(1);
  }

  const bits = Number(options.bits || 4);
  const outDir = options.out || "./models";
  const isJson = Boolean(options.isJson);

  if (!isJson) {
    console.log("\n" + rule("QUANTIZE"));
    console.log(`  Input File        : ${ANSI.bold}${inputPath}${ANSI.reset}`);
    console.log(`  Target Precision  : INT${bits}, block-wise symmetric scales`);
    console.log(`  Tensor Alignment  : 4096-byte offsets`);
    console.log(`  Output Directory  : ${ANSI.dim}${outDir}${ANSI.reset}\n`);
  }

  const result = quantizeSlabFile(inputPath, outDir, {
    bits,
    onProgress: isJson ? undefined : ({ current, total, tensorName, ratio }) => {
      const pct = ((current / total) * 100).toFixed(0);
      process.stdout.write(`\r  [${pct}%] INT${bits}: ${ANSI.dim}${tensorName.slice(0, 35)}${ANSI.reset} (${ratio}x)   `);
    },
  });

  if (isJson) {
    console.log(JSON.stringify(result, null, 2));
    return result;
  }

  console.log("\n");
  console.log(`  Status            : ${ANSI.green}written${ANSI.reset}`);
  console.log(`  Total Tensors     : ${ANSI.bold}${result.totalTensors}${ANSI.reset}`);
  console.log(`  Original Size     : ${(result.originalTotalBytes / (1024 * 1024)).toFixed(1)} MB`);
  console.log(`  Quantized Size    : ${ANSI.bold}${(result.quantizedTotalBytes / (1024 * 1024)).toFixed(1)} MB${ANSI.reset}`);
  console.log(`  Size Ratio        : ${result.netCompressionRatio}x`);
  console.log(`  Mean Tensor SNR   : ${result.avgSnrDb} dB (reconstruction against these weights, not model quality)`);
  console.log(`  Quantized Slab    : ${ANSI.bold}${result.quantizedSlabPath}${ANSI.reset}`);
  console.log(`  Index Manifest    : ${ANSI.bold}${result.manifestPath}${ANSI.reset}`);
  console.log(`  SHA-256           : ${result.sha256}\n`);
  console.log(rule() + "\n");
  return result;
}

function readSchemaArgument(target) {
  if (!target) {
    throw new Error("a schema file path or inline JSON is required");
  }
  if (existsSync(target)) {
    return JSON.parse(readFileSync(target, "utf8"));
  }
  try {
    return JSON.parse(target);
  } catch {
    throw new Error(`"${target}" is neither an existing file nor valid JSON`);
  }
}

// CLI flag and command processing
export function runCli(argv = process.argv.slice(2)) {
  const env = process.env;
  const defaultPort = Number(env.VITNA_ANCHOR_PORT || 8765);
  const defaultHost = env.VITNA_ANCHOR_HOST || "127.0.0.1";

  const command = argv[0];

  if (!command || command === "--help" || command === "-h" || command === "help") {
    printUsage();
    process.exit(0);
  }

  const namedArgs = {};
  const switchArgs = new Set();
  const positionalArgs = [];

  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
        namedArgs[key] = argv[++i];
      } else {
        switchArgs.add(key);
      }
    } else {
      positionalArgs.push(arg);
    }
  }

  switch (command) {
    case "pull": {
      const target = positionalArgs[0] || namedArgs.model;
      if (!target) {
        console.error(ANSI.amber + "Error: a model id or checkpoint path is required. Usage: vitna-anchor pull <model-id-or-path>" + ANSI.reset);
        process.exit(1);
      }
      const outDir = namedArgs.out || "./models";
      const dryRun = switchArgs.has("dry-run");
      const format = namedArgs.format;

      runModelPull(target, { outDir, dryRun, format }).catch((err) => {
        console.error(ANSI.amber + `Pull failed: ${err.message}` + ANSI.reset);
        process.exit(1);
      });
      break;
    }

    case "serve": {
      startAnchorServer({
        host: namedArgs.host || defaultHost,
        port: Number(namedArgs.port || defaultPort),
      });
      break;
    }

    case "chat": {
      console.error(NO_MODEL_MESSAGE);
      process.exit(1);
      break;
    }

    case "quantize": {
      const target = positionalArgs[0] || namedArgs.model;
      const bits = Number(namedArgs.bits || 4);
      const outDir = namedArgs.out || "./models";
      const isJson = switchArgs.has("json");
      runQuantizeCli(target, { bits, out: outDir, isJson });
      break;
    }

    case "stripe": {
      const target = positionalArgs[0] || namedArgs.model;
      if (!target) {
        console.error(ANSI.amber + "Error: a file path is required for striping." + ANSI.reset);
        console.log("Usage: vitna-anchor stripe <file> --drives /mnt/nvme0,/mnt/nvme1 [--chunk-kb 4]");
        process.exit(1);
      }
      const rawDrives = namedArgs.drives || namedArgs.drive || "";
      const driveList = rawDrives.split(",").map((d) => d.trim()).filter(Boolean);
      if (driveList.length < 2) {
        console.error(ANSI.amber + "Error: at least 2 directories are required (e.g. --drives /mnt/nvme0,/mnt/nvme1)." + ANSI.reset);
        process.exit(1);
      }
      const chunkKb = Number(namedArgs["chunk-kb"] || 4);
      const isJson = switchArgs.has("json");
      try {
        const manifest = sliceStripedDmaSlabs(target, driveList, { chunkSizeBytes: chunkKb * 1024 });
        if (isJson) {
          console.log(JSON.stringify(manifest, null, 2));
        } else {
          console.log("\n" + formatStripeSummary(manifest, ANSI) + "\n");
        }
      } catch (err) {
        console.error(ANSI.amber + `Stripe failed: ${err.message}` + ANSI.reset);
        process.exit(1);
      }
      break;
    }

    case "schema": {
      const target = positionalArgs[0] || namedArgs.schema || namedArgs.file;
      let schemaObj;
      try {
        schemaObj = readSchemaArgument(target);
      } catch (err) {
        console.error(ANSI.amber + `Schema failed: ${err.message}` + ANSI.reset);
        process.exit(1);
      }
      const pda = compileJsonSchemaToPda(schemaObj);
      if (switchArgs.has("json")) {
        console.log(JSON.stringify(pda, null, 2));
      } else {
        console.log("\n" + formatGrammarSummary(pda, ANSI) + "\n");
      }
      break;
    }

    default: {
      if (Object.hasOwn(REMOVED_COMMANDS, command)) {
        console.error(`"${command}" was removed: ${REMOVED_COMMANDS[command]}.`);
        process.exit(1);
      }
      console.error(ANSI.amber + `Unknown command: "${command}"` + ANSI.reset + "\n");
      printUsage();
      process.exit(1);
    }
  }
}

// Dispatch only when this file is the entry point (`node runtime/anchor-run.mjs`).
// bin/vitna-anchor.mjs imports runCli and calls it itself.
function isEntryPoint() {
  if (!process.argv[1]) return false;
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isEntryPoint()) {
  runCli();
}
