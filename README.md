# vitna-anchor

[![CI](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml/badge.svg)](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)

An inference engine in C, with a Node.js command line, being built one gate at a time.

**What runs today.** The engine runs one small dense model, [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), on a CPU in float32. Its token ids, logits and greedy output match a pinned reference implementation, and CI checks that on Linux, macOS and Windows. It does not serve the model yet: the HTTP servers answer generation requests with `501 Not Implemented`. Nothing has been published to npm, so there is no install command: build from source.

Earlier versions of this README described an engine that streams experts from NVMe at a stated line rate, drafts tokens speculatively for a speedup, answers from a prefix cache in under a millisecond, guarantees schema-valid JSON, certifies an air gap, and routes to the cheapest cloud provider for a stated saving. None of that was measured, and most of it had not been built. The v0.1.0 release binaries are that earlier simulator.

## The gate ladder

Each gate has a pass condition that a test checks. Nothing is claimed here, in the command line's output or anywhere else in this repository before the gate that proves it has passed.

| Gate | Passes when | Status |
|---|---|---|
| A0, honesty | This README, the command line and both servers state what runs today, and print no figure nobody measured | Passed, [#2](https://github.com/convexityos/vitna-anchor/pull/2) |
| A1, a reference | One small dense open model is pinned by revision and file hash, token ids and logits for fixed prompts are recorded from a pinned reference implementation, and a test compares against them | Passed, [#4](https://github.com/convexityos/vitna-anchor/pull/4). See [`reference/`](reference/README.md) |
| A2, a forward pass on a CPU | Tokenizer, embeddings, RMSNorm, attention over a real key-value cache, MLP and sampling in C. Logits match A1 within a stated tolerance, and greedy output matches token for token | Passed, [#5](https://github.com/convexityos/vitna-anchor/pull/5). CI checks it on every push |
| A3, serving | An OpenAI-compatible `/v1` with streaming, and usage counted from the tokens actually produced | Not started |
| A4, one GPU | The A2 comparison passes on CUDA | Not started |
| A5, experts from a drive | A mixture-of-experts checkpoint streams from NVMe with direct I/O and prefetch, and tokens per second are published only as measured, with the hardware named | Not started |
| A6, reuse and constraints | Prefix reuse over real key-value tensors, and constrained decoding that masks real logits | Not started |

## What is in the repository

"Matches the reference" means `tests/reference.test.mjs` compares it with the fixture in [`reference/`](reference/README.md). "Tested" means the repository's own tests exercise it, on inputs they build themselves: the C unit tests in `engine/tests/` check each part against published test vectors, a formula evaluated in double precision, or an input worked by hand.

| Component | Where | State |
|---|---|---|
| Llama forward pass in float32: embeddings, RMSNorm, grouped-query attention over a key-value cache, half-split rotary embeddings, SwiGLU MLP, tied output layer | `engine/src/model.c`, `engine/src/ops.c` | Matches the reference: every compared logit within the stated tolerance of 1e-2, and 192 of 192 greedy tokens equal. The largest difference on the machine that recorded the reference was 2.2e-4 |
| Byte-level BPE tokenizer read from `tokenizer.json` | `engine/src/tokenizer.c`, `engine/src/unicode.c` | Matches the model's own `tokenizer.json` on all 48 reference strings, and tested on a vocabulary worked by hand. Refuses tokenizer features it does not implement |
| Greedy decoding and seeded temperature, top-k and top-p sampling | `engine/src/sampler.c` | Greedy matches the reference. Sampling is tested for its proportions and its seed |
| Float32 matrix-vector product over F32, BF16 or F16 weights, with scalar, AVX2 and NEON paths | `engine/src/ops.c` | Tested against double precision |
| SafeTensors reader in C | `engine/src/safetensors.c`, `engine/src/json.c` | Tested: the `__metadata__` block is skipped, byte ranges are checked, and malformed files are refused. Reads the model in the A2 comparison |
| RMSNorm and SwiGLU | `engine/src/kernels.c` | Tested against double precision. RMSNorm is used by the forward pass |
| int2/3/4/8 matrix-vector products, with scalar, AVX2 and NEON paths | `engine/src/kernels.c` | Tested against their documented packing formats. No model uses them yet |
| Interleaved rotary position embedding | `engine/src/kv_cache.c` | Tested against its formula. Unused: Hugging Face Llama checkpoints need the half-split form in `ops.c` |
| Paged key-value cache | `engine/src/kv_cache.c` | Unused, untested. Known defect: all layers share one block pool sized for a single layer. The forward pass uses a contiguous cache |
| SHA-256 | `engine/src/crypto.c` | Tested against the FIPS 180-2 examples |
| Top-k softmax routing over experts | `engine/src/router.c` | Untested |
| Expert store and asynchronous reads | `engine/src/expert_store.c`, `engine/src/async_io.c` | Untested. Shards are opened with direct I/O off |
| Prefix tree over token ids | `engine/src/radix_kv.c` | Untested. It holds no key-value tensors |
| JSON Schema to pushdown automaton compiler | `runtime/grammar.mjs`, `engine/src/grammar.c` | The JavaScript half is tested. Neither masks logits yet |
| HTTP server | `engine/src/server.c`, `runtime/anchor-run.mjs` | Tested: health says no model is served, and generation answers 501 |
| SafeTensors and GGUF v2/v3 header parsers | `runtime/ingest.mjs` | Tested on synthetic files |
| Checkpoint rewrite with every tensor at a 4096-byte offset | `runtime/ingest.mjs` | Tested on synthetic files |
| Block-wise INT4 and INT8 quantizer | `runtime/quantize.mjs` | Tested for round-trip error on random tensors |
| Round-robin striping of a file across directories | `runtime/stripe.mjs` | Tested for layout. Read throughput has not been measured |

## Build the engine and run the model

The engine is C11 with no dependencies beyond the C library.

Linux and macOS:

```bash
cd engine
make           # the engine
make test      # the unit tests
```

Windows, with CMake and Visual Studio:

```powershell
cmake -B engine/build -S engine
cmake --build engine/build --config Release
.\engine\build\Release\vitna-anchor-tests.exe
```

Windows, with any clang, gcc or `zig cc`. Set `VITNA_CC` to the compiler's path if it is not on `PATH`:

```powershell
.\engine\build.ps1 -Tests
```

Fetch the model, 269 MB, at its pinned revision, with every file checked against its SHA-256. Then run it:

```bash
node scripts/fetch-model.mjs
./engine/vitna-anchor run --model models/smollm2-135m --prompt "The capital of France is" --max-new 20
```

| Command | What it does |
|---|---|
| `run --model <dir> --prompt <text>` | Prints the prompt's continuation as it is generated |
| `generate --model <dir> (--prompt <text> \| --ids <a,b,...>)` | Prints JSON: the prompt's ids, the new ids and their text. `--logits-out <file>` writes each step's logits |
| `logits --model <dir> (--prompt <text> \| --ids <a,b,...>) --out <file>` | Writes the logits at every position of the prompt, as float32 |
| `tokenize --model <dir> [--text <text>]` | Prints token ids as JSON |
| `info --model <file.safetensors>` | Lists the tensors in a SafeTensors file |
| `serve [--port <port>]` | Starts the HTTP server, whose generation endpoints answer 501 |
| `bench` | Times the int4 matrix-vector kernel on synthetic weights. That describes one kernel on one machine, not a model |

Decoding is greedy unless `--temperature <t>` is given, with optional `--top-k`, `--top-p` and `--seed`. The model directory needs `config.json`, `tokenizer.json` and a single `model.safetensors`. A config that asks for something the engine does not implement is refused on load, and the error says what it was: rope scaling, attention or MLP biases, another activation, or a sharded checkpoint.

## The Node.js command line

From a clone, with Node.js 22 or later:

```bash
node bin/vitna-anchor.mjs --help
```

| Command | What it does |
|---|---|
| `pull <model-id-or-path> [--out <dir>] [--dry-run]` | Reads a SafeTensors or GGUF checkpoint, from disk or the Hugging Face Hub, and rewrites it with every tensor at a 4096-byte offset |
| `quantize <checkpoint> [--bits 4\|8]` | Quantizes a checkpoint's tensors to block-wise INT4 or INT8 |
| `stripe <file> --drives <d1,d2,...>` | Splits a file round-robin into chunks across several directories |
| `schema <schema.json>` | Compiles a JSON Schema into the automaton a constrained decoder would use |
| `serve [--port <port>]` | Starts an HTTP server whose generation endpoints answer 501 |
| `chat` | Says that no model is served yet, and exits |

`probe`, `route`, `draft`, `bench`, `tune` and `registry` were removed. Calling one prints why.

## Tests

```bash
npm test
```

`tests/engine-server.test.mjs` and `tests/reference.test.mjs` run the built engine when they find one, and the reference comparison also needs the model files. Without them those tests are skipped, and say why. CI builds the engine and runs the unit tests on Linux, macOS and Windows, fetches the model, and requires both engine tests to pass.

## Measurements

A performance figure appears in this repository only with the hardware and the command that produced it, and it says whether it was measured or estimated. The engine is not yet measured for speed: tokens per second belong to gate A5.

## License

Apache License 2.0. See [LICENSE](LICENSE).
