# vitna-anchor

[![CI](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml/badge.svg)](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)

An inference engine in C, with a Node.js command line, being built one gate at a time.

**What runs today.** The engine runs one small dense model, [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), on a CPU in float32. Its token ids, logits and greedy output match a pinned reference implementation, and CI checks that on Linux, macOS and Windows. Built with its CUDA path, it runs the same forward pass on an NVIDIA GPU, also in float32, and the same comparison passed there on one GPU, checked by hand ([below](#how-gate-a4-was-checked)). CI compiles the CUDA path but has no GPU to run it on. It serves the model over an OpenAI-compatible HTTP API, `/v1/chat/completions` and `/v1/completions`, streamed or not, with usage counted from the tokens it reads and produces. A request reuses the key-value cache for the prefix it shares with the one before it. JSON mode masks, at every step, the logits of each token that could not continue a JSON object. Nothing has been published to npm, so there is no install command: build from source.

Earlier versions of this README described an engine that streams experts from NVMe at a stated line rate, drafts tokens speculatively for a speedup, answers from a prefix cache in under a millisecond, guarantees schema-valid JSON, certifies an air gap, and routes to the cheapest cloud provider for a stated saving. None of that was measured, and most of it had not been built. The v0.1.0 release binaries are that earlier simulator.

## The gate ladder

Each gate has a pass condition that a test checks. Nothing is claimed here, in the command line's output or anywhere else in this repository before the gate that proves it has passed.

| Gate | Passes when | Status |
|---|---|---|
| A0, honesty | This README, the command line and both servers state what runs today, and print no figure nobody measured | Passed, [#2](https://github.com/convexityos/vitna-anchor/pull/2) |
| A1, a reference | One small dense open model is pinned by revision and file hash, token ids and logits for fixed prompts are recorded from a pinned reference implementation, and a test compares against them | Passed, [#4](https://github.com/convexityos/vitna-anchor/pull/4). See [`reference/`](reference/README.md) |
| A2, a forward pass on a CPU | Tokenizer, embeddings, RMSNorm, attention over a real key-value cache, MLP and sampling in C. Logits match A1 within a stated tolerance, and greedy output matches token for token | Passed, [#5](https://github.com/convexityos/vitna-anchor/pull/5). CI checks it on every push |
| A3, serving | An OpenAI-compatible `/v1` with streaming, and usage counted from the tokens actually produced | Passed, [#7](https://github.com/convexityos/vitna-anchor/pull/7). CI checks it on every push |
| A4, one GPU | The A2 comparison passes on CUDA | This change: passed on one GPU, an NVIDIA GeForce RTX 3070, checked by hand with the command [below](#how-gate-a4-was-checked). CI compiles the CUDA path and does not run it: GitHub's runners have no GPU |
| A5, experts from a drive | A mixture-of-experts checkpoint streams from NVMe with direct I/O and prefetch, and tokens per second are published only as measured, with the hardware named | Not started |
| A6, reuse and constraints | Prefix reuse over real key-value tensors, and constrained decoding that masks real logits | Passed, [#9](https://github.com/convexityos/vitna-anchor/pull/9), for reuse of the previous request's cache and for JSON object mode. CI checks it on every push. Constraining output to a JSON Schema is not built |

## What is in the repository

"Matches the reference" means `tests/reference.test.mjs` compares it with the fixture in [`reference/`](reference/README.md). "Tested" means the repository's own tests exercise it, on inputs they build themselves: the C unit tests in `engine/tests/` check each part against published test vectors, a formula evaluated in double precision, or an input worked by hand.

| Component | Where | State |
|---|---|---|
| Llama forward pass in float32: embeddings, RMSNorm, grouped-query attention over a key-value cache, half-split rotary embeddings, SwiGLU MLP, tied output layer | `engine/src/model.c`, `engine/src/ops.c` | Matches the reference: every compared logit within the stated tolerance of 1e-2, and 192 of 192 greedy tokens equal. The largest difference on the machine that recorded the reference was 2.2e-4 |
| The same forward pass on an NVIDIA GPU, in float32: five fused kernels a layer, attention split across blocks by position, and each token a replay of CUDA graphs; the weights uploaded once, the key-value cache on the device, logits copied back only when a step asks for them | `engine/src/model_cuda.cu` | Matches the reference on one GPU, checked by hand: every compared logit within 1e-2, the largest difference 2.25e-4, and 192 of 192 greedy tokens equal. CI compiles it and does not run it |
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
| JSON Schema to pushdown automaton compiler | `runtime/grammar.mjs` | Tested. Used by the Node.js command line's `schema` command, not by the engine. Its C counterpart, `grammar.c`, rejected every object with a key (`{"key": 1}` at byte 2) and accepted `{abc}`; A6 deleted it |
| OpenAI-compatible HTTP API: chat and text completions, streamed or not, usage from tokens | `engine/src/api.c`, `engine/src/server.c` | A greedy completion through it equals the reference's greedy output, with prompt and completion tokens counted exactly, on all six reference prompts. Streaming, stop sequences, seeds, refusals, a client hanging up mid-stream and a step that fails are tested, the last with a hook that makes a step fail on the CPU |
| Prefix reuse: a request keeps the key-value cache for the tokens it shares with the one before | `engine/src/api.c`, `engine/src/model.c` | Tested: a repeated prompt reuses all but its last token and still returns the reference's greedy output, and replies equal those of a server that reuses nothing, greedy or sampled |
| JSON mode: every token that could not continue a JSON object has its logit set to minus infinity before each choice | `engine/src/jsonpfx.c`, `engine/src/api.c` | The check is tested against the engine's JSON parser on 3,000 mutated objects, and on cases worked by hand. Replies parse as JSON objects, or are valid starts of one when `max_tokens` cuts them short |
| The Node.js command line's server | `runtime/anchor-run.mjs` | Tested: it serves no model, answers generation with 501, and names the engine's server |
| SafeTensors and GGUF v2/v3 header parsers | `runtime/ingest.mjs` | Tested on synthetic files |
| Checkpoint rewrite with every tensor at a 4096-byte offset | `runtime/ingest.mjs` | Tested on synthetic files |
| Block-wise INT4 and INT8 quantizer | `runtime/quantize.mjs` | Tested for round-trip error on random tensors |
| Round-robin striping of a file across directories | `runtime/stripe.mjs` | Tested for layout. Read throughput has not been measured |

## Build the engine and run the model

The engine is C11 with no dependencies beyond the C library. Its CUDA path, built only when asked for, also needs the CUDA toolkit.

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

With the CUDA path, which runs the model on an NVIDIA GPU when given `--device cuda`. It needs the CUDA toolkit's `nvcc` (A4 was checked with 13.1) and, on Windows, Visual Studio's C++ build tools as nvcc's host compiler. Unless an architecture is named, it builds for the GPU in the machine:

```bash
cd engine
make VITNA_CUDA=1              # CUDA_ARCH=sm_86 names an architecture
```

```powershell
cmake -B engine/build -S engine -DVITNA_CUDA=ON     # -DCMAKE_CUDA_ARCHITECTURES=86 names one
cmake --build engine/build --config Release
```

The CUDA runtime is linked in statically, so an engine built with the CUDA path loads no CUDA library to start, and on the CPU, its default, it makes no CUDA call at all. Asked for `--device cuda`, an engine built without the CUDA path says so and stops, and so does one that finds no usable GPU: the model never runs on the CPU in the GPU's place. `engine/build.ps1` builds the CPU engine only.

Fetch the model, 269 MB, at its pinned revision, with every file checked against its SHA-256. Then run it:

```bash
node scripts/fetch-model.mjs
./engine/vitna-anchor run --model models/smollm2-135m --prompt "The capital of France is" --max-new 20
```

| Command | What it does |
|---|---|
| `run --model <dir> --prompt <text>` | Prints the prompt's continuation as it is generated |
| `generate --model <dir> (--prompt <text> \| --ids <a,b,...>)` | Prints JSON: the prompt's ids, the new ids and their text. `--logits-out <file>` writes each step's logits, and `--timing` prints to stderr how long the prompt and the new tokens took |
| `logits --model <dir> (--prompt <text> \| --ids <a,b,...>) --out <file>` | Writes the logits at every position of the prompt, as float32 |
| `tokenize --model <dir> [--text <text>]` | Prints token ids as JSON |
| `info --model <file.safetensors>` | Lists the tensors in a SafeTensors file |
| `serve --model <dir> [--model-id <id>] [--host <ip>] [--port <port>]` | Serves the model over the OpenAI-compatible API below. Without `--model` its generation endpoints answer 501 |
| `bench` | Times the int4 matrix-vector kernel on synthetic weights. That describes one kernel on one machine, not a model |

`run`, `generate`, `logits` and `serve` run the model on the CPU unless given `--device cuda`, which runs it on the first CUDA device, in an engine built with the CUDA path. Decoding is greedy unless `--temperature <t>` is given, with optional `--top-k`, `--top-p` and `--seed`. The model directory needs `config.json`, `tokenizer.json` and a single `model.safetensors`. A config that asks for something the engine does not implement is refused on load, and the error says what it was: rope scaling, attention or MLP biases, another activation, or a sharded checkpoint.

## Serve the model

```bash
./engine/vitna-anchor serve --model models/smollm2-135m
curl http://127.0.0.1:8765/v1/chat/completions -H "content-type: application/json" \
  -d '{"model": "smollm2-135m", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

Any OpenAI client can use `http://127.0.0.1:8765/v1` as its base URL. The model is served under its directory's name unless `--model-id` gives another, and a request naming a different model gets a 404.

- **Endpoints.** `GET /v1/models`, `GET /v1/health`, `POST /v1/chat/completions` and `POST /v1/completions`, each streamed as server-sent events with `"stream": true`. A completion prompt may be a string or an array of token ids.
- **Usage.** `prompt_tokens` counts the tokens the model read, after the chat format is applied. `completion_tokens` counts the tokens it generated, including the end-of-text or other special token it stopped on. `prompt_tokens_details.cached_tokens` counts the prompt tokens whose keys and values were reused from the request before, rather than computed again. A streamed response carries usage on its final chunk, and also sends OpenAI's separate usage chunk when `stream_options.include_usage` is true.
- **Chat format.** Messages are formatted as ChatML (`<|im_start|>role`), the format the SmolLM2 family's instruct models use. The pinned model is the base model, which was not trained on it, so its chat replies are poor. Generation stops at any special token.
- **Parameters.** `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `top_k`, `seed`, `stop` (up to four), `stream`, `stream_options` and `response_format`. The seed used is returned in an `x-vitna-seed` header, so a sampled reply can be reproduced. Anything that would change the output and is not implemented is refused with a 400 naming it: `n` above 1, tools, a JSON Schema (`response_format` of type `json_schema`), log probabilities, penalties and logit bias. A field the server does not know is ignored and named in an `x-vitna-ignored` header.
- **JSON mode.** With `"response_format": {"type": "json_object"}`, before each token is chosen the logit of every token that could not continue a JSON object is set to minus infinity: special tokens, and any token whose bytes would break the object. Strings must be valid UTF-8, and a run of whitespace is capped at 16 characters so the model cannot pad forever. Generation stops when the object closes. A reply cut short, by `max_tokens` (`finish_reason` `length`) or by a stop sequence, is the valid start of an object. The model is not told to write JSON, so the prompt should ask for it.
- **Prefix reuse.** The server keeps the key-value cache of the last request and reuses it for the longest prefix the next prompt shares with it, recomputing at least the last prompt token. A conversation sent back with one more turn reuses its history. Reuse only skips work: a reply, and its usage apart from `cached_tokens`, is what a server started with `--no-prefix-cache`, which turns reuse off, returns for the same request.
- **A step that fails.** If the model fails to run a step, the request ends with an error, never with the reply so far. A response not yet begun is a 500 of type `server_error`. A stream has already sent its 200, so it ends with an event carrying the same error, in place of the final chunk, with no usage and no `[DONE]` after it. The next request reuses only the tokens whose steps ran. On the GPU a step fails when the device reports an error, which the server prints to stderr; no such error has been caused to test this. The tests make a step fail with a hook instead (see [Tests](#tests)).
- **Limits.** One request at a time. The cache holds one sequence, so two clients taking turns evict each other's prefix. The context is the model's maximum, at most 4096 tokens, or `--ctx`. A request that would not fit is refused, not cut short. There is no authentication, so the server listens on 127.0.0.1 unless `--host` says otherwise, and warns if it does.

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
| `serve [--port <port>]` | Starts an HTTP server that serves no model: its generation endpoints answer 501 and name the engine's server |
| `chat` | Says that the engine serves the model, and exits |

`probe`, `route`, `draft`, `bench`, `tune` and `registry` were removed. Calling one prints why.

## Tests

```bash
npm test
```

`tests/engine-server.test.mjs`, `tests/reference.test.mjs`, `tests/serving.test.mjs`, `tests/reuse-and-json.test.mjs` and `tests/step-failure.test.mjs` run the built engine when they find one, and the last four also need the model files. Without them those tests are skipped, and say why. CI builds the engine and runs the unit tests on Linux, macOS and Windows, fetches the model, and requires all five engine tests to pass.

`tests/step-failure.test.mjs` starts each of its servers with `VITNA_TEST_FAIL_STEP=<position>`, a hook for tests only, which makes the step at that position fail once, before it computes anything. It is how the tests check what a client sees when a step fails, with no GPU error to cause one. A server started with it says so on stderr.

With `VITNA_DEVICE=cuda`, the last four run the model on the GPU, in an engine built with the CUDA path; nothing else about them changes, the fixture and the tolerance included. `tests/device.test.mjs` checks that an engine refuses a device it cannot use. CI's CUDA job runs it against an engine built without the CUDA path and one built with it, on a runner with no GPU.

## How gate A4 was checked

By hand, on one machine, on 2026-09-29. CI does not run it again: GitHub's hosted runners have no GPU. What CI does is compile the CUDA path, with Make and with CMake, in NVIDIA's CUDA 13.1.1 development container, and check the refusals above.

- GPU: NVIDIA GeForce RTX 3070, 8 GB, compute capability 8.6, driver 591.86
- CUDA 13.1.1 (nvcc 13.1.115), with MSVC 19.44 from Visual Studio 2022 Build Tools 17.14 as its host compiler
- Windows 11 (10.0.26200), AMD Ryzen 7 3700X, Node.js 24.19.0

The commands, from a clone:

```powershell
cmake -B engine/build -S engine -DVITNA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build engine/build --config Release
node scripts/fetch-model.mjs
$env:VITNA_REQUIRE_REFERENCE = "1"; $env:VITNA_DEVICE = "cuda"
node --test --test-reporter=spec tests/reference.test.mjs
```

All five tests passed. Every compared logit was within the tolerance of 1e-2, which A1 fixed before the engine existed and A4 left alone: the largest difference was 2.30e-4, and 1.48e-4 for logsumexp. 192 of 192 greedy tokens were equal, and the top logits at each step differed by at most 5.53e-5. On the same machine the CPU path's largest difference is 3.01e-4, with 192 of 192 greedy tokens equal. `tests/serving.test.mjs` and `tests/reuse-and-json.test.mjs` passed with `VITNA_DEVICE=cuda` as well, so serving and prefix reuse work over the cache on the device.

The GPU kernels were then rewritten for speed: CUDA graphs, fused kernels, and attention split across blocks by position. The same commands on the same machine, on the same day, passed again, with a largest difference of 2.25e-4, 1.65e-4 for logsumexp, 192 of 192 greedy tokens equal, and at most 4.58e-5 between the top logits at a step.

On the GPU the arithmetic is float32 on CUDA cores. No tensor cores are used, so TF32 does not apply, and the build does not pass `--use_fast_math`, so division, square root and `expf` keep their accurate forms. nvcc's default fused multiply-add is on; the CPU path's AVX2 matrix-vector product uses FMA too.

## Measurements

A performance figure appears in this repository only with the hardware and the command that produced it, and it says whether it was measured or estimated. No speed figure is published here yet: tokens per second belong to gate A5. `scripts/bench-decode.mjs` times decoding with `generate --timing` and prints the machine and the command with every result, so that a figure, when there is one, carries both.

## License

Apache License 2.0. See [LICENSE](LICENSE).
