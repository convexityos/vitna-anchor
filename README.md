# vitna-anchor

[![CI](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml/badge.svg)](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)

An inference engine in C, with a Node.js command line, being built one gate at a time.

**No model runs yet.** There is no tokenizer and no attention, so nothing here can generate text. The HTTP server answers generation requests with `501 Not Implemented`. Nothing has been published to npm, so there is no install command: build from source.

Earlier versions of this README described an engine that streams experts from NVMe at a stated line rate, drafts tokens speculatively for a speedup, answers from a prefix cache in under a millisecond, guarantees schema-valid JSON, certifies an air gap, and routes to the cheapest cloud provider for a stated saving. None of that was measured, and most of it had not been built. The v0.1.0 release binaries are that earlier simulator.

## The gate ladder

Each gate has a pass condition that a test checks. Nothing is claimed here, in the command line's output or anywhere else in this repository before the gate that proves it has passed.

| Gate | Passes when | Status |
|---|---|---|
| A0, honesty | This README, the command line and both servers state what runs today, and print no figure nobody measured | This change |
| A1, a reference | One small dense open model is pinned by revision and file hash, token ids and logits for fixed prompts are recorded from a pinned reference implementation, and a test compares against them | Not started |
| A2, a forward pass on a CPU | Tokenizer, embeddings, RMSNorm, attention over a real key-value cache, MLP and sampling in C. Logits match A1 within a stated tolerance, and greedy output matches token for token | Not started |
| A3, serving | An OpenAI-compatible `/v1` with streaming, and usage counted from the tokens actually produced | Not started |
| A4, one GPU | The A2 comparison passes on CUDA | Not started |
| A5, experts from a drive | A mixture-of-experts checkpoint streams from NVMe with direct I/O and prefetch, and tokens per second are published only as measured, with the hardware named | Not started |
| A6, reuse and constraints | Prefix reuse over real key-value tensors, and constrained decoding that masks real logits | Not started |

## What is in the repository

Nothing below has been checked against an external reference yet. "Tested" means the repository's own tests exercise it, on inputs they build themselves.

| Component | Where | State |
|---|---|---|
| SafeTensors and GGUF v2/v3 header parsers | `runtime/ingest.mjs` | Tested on synthetic files |
| Checkpoint rewrite with every tensor at a 4096-byte offset | `runtime/ingest.mjs` | Tested on synthetic files |
| Block-wise INT4 and INT8 quantizer | `runtime/quantize.mjs` | Tested for round-trip error on random tensors |
| Round-robin striping of a file across directories | `runtime/stripe.mjs` | Tested for layout. Read throughput has not been measured |
| JSON Schema to pushdown automaton compiler | `runtime/grammar.mjs`, `engine/src/grammar.c` | The JavaScript half is tested. Neither masks any logits, since none exist yet |
| SafeTensors parser in C | `engine/src/safetensors.c` | Untested. Known defect: the `__metadata__` block is read as a tensor |
| RMSNorm, SwiGLU, and int2/3/4/8 matrix-vector products, with scalar, AVX2 and NEON paths | `engine/src/kernels.c` | Untested beyond a timing loop on synthetic data |
| Rotary position embedding | `engine/src/kv_cache.c` | Untested. Uses the interleaved-pair convention, not the half-split one that Hugging Face Llama checkpoints expect |
| Paged key-value cache | `engine/src/kv_cache.c` | Untested. Known defect: all layers share one block pool sized for a single layer |
| Top-k softmax routing over experts | `engine/src/router.c` | Untested |
| Expert store and asynchronous reads | `engine/src/expert_store.c`, `engine/src/async_io.c` | Untested. Shards are opened with direct I/O off |
| Prefix tree over token ids | `engine/src/radix_kv.c` | Untested. It holds no key-value tensors |
| SHA-256 | `engine/src/crypto.c` | Untested |
| HTTP server | `engine/src/server.c`, `runtime/anchor-run.mjs` | Tested: health says no model, generation answers 501 |

## Build the engine from source

The engine is C11 with no dependencies beyond the C library.

Linux and macOS:

```bash
cd engine
make
./vitna-anchor info --model path/to/model.safetensors
```

Windows, with CMake and Visual Studio:

```powershell
cmake -B engine/build -S engine
cmake --build engine/build --config Release
```

Windows, with any clang, gcc or `zig cc`. Set `VITNA_CC` to the compiler's path if it is not on `PATH`:

```powershell
.\engine\build.ps1
```

`vitna-anchor serve` starts the HTTP server, `info` lists the tensors in a SafeTensors file, and `bench` times the int4 matrix-vector kernel on synthetic weights. That timing describes one kernel on one machine, not a model.

## The command line

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
| `chat` | Says that no model runs, and exits |

`probe`, `route`, `draft`, `bench`, `tune` and `registry` were removed. Calling one prints why.

## Tests

```bash
npm test
```

`tests/engine-server.test.mjs` runs the built engine when it finds one, and is skipped otherwise. CI builds the engine on Linux, macOS and Windows and requires it.

## Measurements

A performance figure appears in this repository only with the hardware and the command that produced it, and it says whether it was measured or estimated.

## License

Apache License 2.0. See [LICENSE](LICENSE).
