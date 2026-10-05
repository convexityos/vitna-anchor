# vitna-anchor

[![CI](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml/badge.svg)](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)

An inference engine in C, with a Node.js command line, being built one gate at a time.

**What runs today.** The engine runs one small dense model, [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), on a CPU in float32. Its token ids, logits and greedy output match a pinned reference implementation, and CI checks that on Linux, macOS and Windows. Built with its CUDA path, it runs the same forward pass on an NVIDIA GPU, also in float32, and the same comparison passed there on one GPU, checked by hand ([below](#how-gate-a4-was-checked)). CI compiles the CUDA path but has no GPU to run it on. It serves the model over an OpenAI-compatible HTTP API, `/v1/chat/completions` and `/v1/completions`, streamed or not, with usage counted from the tokens it reads and produces. A request reuses the key-value cache for the prefix it shares with the one before it. JSON mode masks, at every step, the logits of each token that could not continue a JSON object. It also runs a mixture of experts, [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924), on a CPU in float32, and matches that model's own pinned reference: the same tolerance on logits, the same experts chosen at every layer, and the same greedy output, checked by hand ([below](#how-gate-a5s-forward-pass-was-checked)) because its 13.8 GB of weights are more than CI holds. Told to, it reads that model's experts from the drive as the layers want them, with direct I/O, into a cache in memory of a size it is given, and the logits are then the same, bit for bit; how fast that runs was measured on one machine ([below](#speed-with-the-experts-read-from-a-drive)). Built with its CUDA path, it runs that model on an NVIDIA GPU too, which keeps as many of the experts as its memory holds and has the others copied in as the layers want them, from the mapped checkpoint or from that cache in memory. The same comparison passed there, checked by hand, and how fast it runs was measured on one GPU ([below](#speed-on-a-gpu)). Nothing has been published to npm, so there is no install command: build from source.

Earlier versions of this README described an engine that streams experts from NVMe at a stated line rate, drafts tokens speculatively for a speedup, answers from a prefix cache in under a millisecond, guarantees schema-valid JSON, certifies an air gap, and routes to the cheapest cloud provider for a stated saving. None of that was measured, and most of it had not been built. The v0.1.0 release binaries are that earlier simulator.

## The gate ladder

Each gate has a pass condition that a test checks. Nothing is claimed here, in the command line's output or anywhere else in this repository before the gate that proves it has passed.

| Gate | Passes when | Status |
|---|---|---|
| A0, honesty | This README, the command line and both servers state what runs today, and print no figure nobody measured | Passed, [#2](https://github.com/convexityos/vitna-anchor/pull/2) |
| A1, a reference | One small dense open model is pinned by revision and file hash, token ids and logits for fixed prompts are recorded from a pinned reference implementation, and a test compares against them | Passed, [#4](https://github.com/convexityos/vitna-anchor/pull/4). See [`reference/`](reference/README.md) |
| A2, a forward pass on a CPU | Tokenizer, embeddings, RMSNorm, attention over a real key-value cache, MLP and sampling in C. Logits match A1 within a stated tolerance, and greedy output matches token for token | Passed, [#5](https://github.com/convexityos/vitna-anchor/pull/5). CI checks it on every push |
| A3, serving | An OpenAI-compatible `/v1` with streaming, and usage counted from the tokens actually produced | Passed, [#7](https://github.com/convexityos/vitna-anchor/pull/7). CI checks it on every push |
| A4, one GPU | The A2 comparison passes on CUDA | Passed, [#11](https://github.com/convexityos/vitna-anchor/pull/11), on one GPU, an NVIDIA GeForce RTX 3070, checked by hand with the command [below](#how-gate-a4-was-checked). CI compiles the CUDA path and does not run it: GitHub's runners have no GPU |
| A5, experts from a drive | A mixture-of-experts checkpoint streams from NVMe with direct I/O and prefetch, and tokens per second are published only as measured, with the hardware named | Passed, [#29](https://github.com/convexityos/vitna-anchor/pull/29), [#32](https://github.com/convexityos/vitna-anchor/pull/32), [#34](https://github.com/convexityos/vitna-anchor/pull/34) and [#35](https://github.com/convexityos/vitna-anchor/pull/35), on one machine, checked by hand ([below](#how-gate-a5s-forward-pass-was-checked)): OLMoE-1B-7B's experts read from an NVMe drive with direct I/O and the next layer's prefetched, the logits unchanged bit for bit, and the model matching its pinned reference on the CPU and on an NVIDIA GeForce RTX 3070. Streamed through a cache of 4 GiB in memory it decodes 2.2 tokens a second on the CPU and 8.7 on the GPU; with the whole model in the file cache, 6.1 and 61 ([on the CPU](#speed-with-the-experts-read-from-a-drive), [on the GPU](#speed-on-a-gpu)). CI checks the model's tokenizer and the expert cache's unit tests, and not the model: its weights are 13.8 GB |
| A6, reuse and constraints | Prefix reuse over real key-value tensors, and constrained decoding that masks real logits | Passed, [#9](https://github.com/convexityos/vitna-anchor/pull/9), for reuse of the previous request's cache and for JSON object mode. CI checks it on every push. Constraining output to a JSON Schema is not built |
| A7, quantized weights | GGUF checkpoints with F32, F16, BF16, Q8_0, Q4_K and Q6_K tensors run on the CPU and the GPU, and with OLMoE's official GGUFs at Q8_0 and Q4_K_M the logits match the pinned reference run on the same dequantized weights within 1e-2, with 192 of 192 greedy tokens equal | Not started. This gate and those after it are planned in [`docs/PARITY.md`](docs/PARITY.md) |
| A8, the CPU computes what the GPU lacks | Experts the GPU's cache lacks run on the CPU from memory, at the same time as the GPU runs the others, and logits are byte-identical to the GPU-only path at every cache size | Not started |
| A9, Qwen3-30B-A3B | The model matches a pinned reference within 1e-2, with the reference's experts chosen and 192 of 192 greedy tokens equal, on the CPU and the GPU, from BF16, Q8_0 and Q4_K_M | Not started |
| A10, long context | A 32K-token prompt runs with the part of the key-value cache the GPU cannot hold in page-locked host memory, and logits are byte-identical to the cache-on-GPU run where both fit | Not started |
| A11, the API surface | Anthropic's `/v1/messages`, tool calls and OpenAI's Responses API, each tested against recorded shapes | Not started |
| A12, installing and a chat page | One command on Windows and on Linux picks a model that fits the machine, fetches it with checks, and starts the server, and a chat page talks to it | Not started |
| A13, AMD and several GPUs | The A7 and A9 comparisons pass on an AMD GPU and on a model split across two GPUs | Not started |

## What is in the repository

"Matches the reference" means `tests/reference.test.mjs` compares it with the fixture in [`reference/`](reference/README.md), or for OLMoE `tests/reference-moe.test.mjs` with that model's fixture. "Tested" means the repository's own tests exercise it, on inputs they build themselves: the C unit tests in `engine/tests/` check each part against published test vectors, a formula evaluated in double precision, or an input worked by hand.

| Component | Where | State |
|---|---|---|
| Llama forward pass in float32: embeddings, RMSNorm, grouped-query attention over a key-value cache, half-split rotary embeddings, SwiGLU MLP, tied output layer | `engine/src/model.c`, `engine/src/ops.c` | Matches the reference: every compared logit within the stated tolerance of 1e-2, and 192 of 192 greedy tokens equal. The largest difference on the machine that recorded the reference was 2.2e-4 |
| The same forward pass on an NVIDIA GPU, in float32: five fused kernels a layer, attention split across blocks by position, and each token a replay of CUDA graphs; the weights uploaded once, the key-value cache on the device, logits copied back only when a step asks for them | `engine/src/model_cuda.cu` | Matches the reference on one GPU, checked by hand: every compared logit within 1e-2, the largest difference 2.25e-4, and 192 of 192 greedy tokens equal. CI compiles it and does not run it |
| OLMoE's mixture of experts in float32: an RMSNorm over all of each query and key projection, a router over 64 experts, the 8 with the largest weights each a SwiGLU MLP, weighted by the router's softmax without renormalizing; a checkpoint in shards, read through its index | `engine/src/model.c` | Matches its reference on the CPU, checked by hand: with every token's experts pinned to the reference's, every compared logit within the tolerance of 1e-2, the largest difference 4.96e-5, and 192 of 192 greedy tokens equal; the engine's own router chose the reference's experts in all 6,848 decisions, and unpinned it decodes the same 192 tokens. On the GPU too, checked by hand: the largest difference 4.77e-5, 192 of 192 greedy tokens equal pinned and unpinned, and the router the reference's in every decision. CI does not run it: the weights are 13.8 GB |
| Byte-level BPE tokenizer read from `tokenizer.json`, with NFC normalization and added tokens matched before and after it, as the tokenizers library does | `engine/src/tokenizer.c`, `engine/src/unicode.c` | Matches each model's own `tokenizer.json` on all of its reference strings, 48 for SmolLM2 and 60 for OLMoE, both in CI. Tested on vocabularies worked by hand. NFC equals Python's `unicodedata` (Unicode 15.0, where the tables come from) on every code point and 1.2 million random strings of marks, composites and jamo (`engine/tools/check_nfc.py`). Refuses tokenizer features it does not implement |
| Greedy decoding and seeded temperature, top-k and top-p sampling | `engine/src/sampler.c` | Greedy matches the reference. Sampling is tested for its proportions and its seed |
| Float32 matrix-vector product over F32, BF16 or F16 weights, with scalar, AVX2 and NEON paths | `engine/src/ops.c` | Tested against double precision |
| SafeTensors reader in C | `engine/src/safetensors.c`, `engine/src/json.c` | Tested: the `__metadata__` block is skipped, byte ranges are checked, and malformed files are refused. Reads the model in the A2 comparison, and OLMoE's three shards |
| RMSNorm and SwiGLU | `engine/src/kernels.c` | Tested against double precision. RMSNorm is used by the forward pass |
| int2/3/4/8 matrix-vector products, with scalar, AVX2 and NEON paths | `engine/src/kernels.c` | Tested against their documented packing formats. No model uses them yet |
| Interleaved rotary position embedding | `engine/src/kv_cache.c` | Tested against its formula. Unused: Hugging Face Llama checkpoints need the half-split form in `ops.c` |
| Paged key-value cache | `engine/src/kv_cache.c` | Unused, untested. Known defect: all layers share one block pool sized for a single layer. The forward pass uses a contiguous cache |
| SHA-256 | `engine/src/crypto.c` | Tested against the FIPS 180-2 examples |
| Top-k softmax routing over experts | `engine/src/router.c` | Untested, and unused: it takes the softmax over the top k alone, so its weights are renormalized, which OLMoE's are not. The forward pass routes in `model.c` |
| A mixture of experts' experts read from the drive with direct I/O, into a cache in memory of a set size: reads on four threads, each with its own handles; the least used expert given up first; and, before each layer's experts run, the next layer's guessed and prefetched | `engine/src/expert_stream.c`, `engine/src/model.c` | Tested on files made for the test: parts that start mid-sector, parts that run together, an expert in two files, the end of a file, a read past it, a prefetch, and which expert is given up. With OLMoE, the logits and greedy tokens through a cache of 256 MiB equal the mapped run's, byte for byte. Its speed was measured by hand ([below](#speed-with-the-experts-read-from-a-drive)). It replaces an earlier expert store and asynchronous reader that were never used or tested, and opened their files with direct I/O off |
| A mixture of experts on the GPU: everything but the experts on the device; a cache of experts there, the least used given up first, each copied in on a stream of its own when a layer wants it, from the mapped checkpoint or the expert cache in memory, registered page-locked; the half of the next layer's guess the router ranks highest copied behind; QK-norm, the router and the experts in kernels of their own, a token a layer at a time; several tokens together as rows, each expert run once for all the rows routed to it | `engine/src/model_cuda.cu`, `engine/src/model.c` | Checked by hand on one GPU: matches OLMoE's reference as the CPU path does, above, and with the cache on the device as large as it allows, at 256 MiB, and at 256 MiB fed from a cache in memory of 256 MiB, the logits and greedy tokens are the same, byte for byte. Rows give a token at a time's logits, byte for byte: every position of a prompt, greedy tokens drafted with `--speculate`, and a prompt of 148 tokens through a cache of 256 MiB. Its speed was measured by hand ([below](#speed-on-a-gpu)). CI compiles it but has no GPU to run it on |
| Prefix tree over token ids | `engine/src/radix_kv.c` | Untested. It holds no key-value tensors |
| JSON Schema to pushdown automaton compiler | `runtime/grammar.mjs` | Tested. Used by the Node.js command line's `schema` command, not by the engine. Its C counterpart, `grammar.c`, rejected every object with a key (`{"key": 1}` at byte 2) and accepted `{abc}`; A6 deleted it |
| OpenAI-compatible HTTP API: chat and text completions, streamed or not, usage from tokens | `engine/src/api.c`, `engine/src/server.c` | A greedy completion through it equals the reference's greedy output, with prompt and completion tokens counted exactly, on all six reference prompts. Streaming, stop sequences, seeds, refusals, a client hanging up mid-stream and a step that fails are tested, the last with a hook that makes a step fail on the CPU |
| Prefix reuse: a request keeps the key-value cache for the tokens it shares with the one before | `engine/src/api.c`, `engine/src/model.c` | Tested: a repeated prompt reuses all but its last token and still returns the reference's greedy output, and replies equal those of a server that reuses nothing, greedy or sampled. On the GPU the same test passes, but equality holds only up to float32 rounding ([Serve the model](#serve-the-model)) |
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

OLMoE-1B-7B, 13.8 GB, runs the same way. The engine maps its three files rather than reading them in:

```bash
node scripts/fetch-model.mjs olmoe-1b-7b
./engine/vitna-anchor run --model models/olmoe-1b-7b --prompt "The capital of France is" --max-new 20
```

With `--expert-cache <MiB>`, the engine does not map OLMoE's experts, 12.9 GB of its 13.8: it reads each one from the drive when a layer wants it, with direct I/O, past the operating system's file cache, into a cache in memory of that many MiB, and gives up the least used one when it needs room. Before a layer's experts run, the next layer's router scores the residual stream as it stands and the experts it would choose start being read: a guess, which costs a read when it is wrong and decides nothing. The rest of the model, under a gigabyte, stays mapped. The logits are the same as without it, bit for bit, and `--timing` says how the cache did:

```bash
./engine/vitna-anchor generate --model models/olmoe-1b-7b --prompt "The capital of France is" --max-new 32 --greedy --timing --expert-cache 2048
./engine/vitna-anchor read-experts --model models/olmoe-1b-7b    # how fast this drive feeds the cache
```

With `--device cuda`, in an engine built with the CUDA path, OLMoE runs on the GPU. Everything but the experts goes there on load, under a gigabyte. The experts stay where they were, in the mapped checkpoint or, with `--expert-cache`, the cache in memory, and the GPU keeps as many of them as `--gpu-expert-cache <MiB>` holds: by default what it has free once the rest of the model is there, less 512 MiB. When a layer wants an expert the GPU lacks, it is copied in, and the least used one is given up for room; of the experts the next layer's router guesses, the half it ranks highest are copied behind, as the CPU path reads its guesses early. Half, because a copy of a wrong guess costs as much as a right one saves, and the lower half was wrong too often (measured [below](#speed-on-a-gpu)). The memory they are copied from is registered with the GPU, page-locked, so the copies run at the bus's speed; for the mapped checkpoint that keeps its 13.8 GB in memory while the engine runs, and costs a few seconds on load. A token runs a layer at a time, since which experts a layer needs is known only once its router has run. Several run together as rows: a prompt's tokens, up to 1,024 at a time, and drafted tokens (`--speculate`) and requests at once (`serve --parallel`). Each layer's attention and router run for every row, then each expert they want runs once for all the rows routed to it, so it is copied to the GPU once a layer rather than once a token. Each row's arithmetic is the one its own step does, in the same order, so rows give the logits of a token at a time, bit for bit. The logits are the same, bit for bit, whatever the GPU's cache holds and wherever the copies come from, and `--timing` says how the cache did:

```bash
./engine/vitna-anchor generate --model models/olmoe-1b-7b --prompt "The capital of France is" --max-new 32 --greedy --timing --device cuda
```

| Command | What it does |
|---|---|
| `run --model <dir> --prompt <text>` | Prints the prompt's continuation as it is generated |
| `generate --model <dir> (--prompt <text> \| --ids <a,b,...>)` | Prints JSON: the prompt's ids, the new ids and their text. `--logits-out <file>` writes each step's logits, and `--timing` prints to stderr how long the prompt and the new tokens took. `--speculate <k>` drafts up to k tokens after each one taken, from the place the text's last two or three tokens occur earlier in it, and checks them together, drafting nothing for a while after a pass whose drafts were all refused; the ids and logits are the ones decoding a token at a time gives, byte for byte, which `tests/speculate.test.mjs` checks. On a GPU the drafts are checked in one pass, k at most 7, or 63 for a mixture of experts; on the CPU a step at a time |
| `logits --model <dir> (--prompt <text> \| --ids <a,b,...>) --out <file>` | Writes the logits at every position of the prompt, as float32 |
| `tokenize --model <dir> [--text <text>]` | Prints token ids as JSON |
| `normalize --model <dir> [--text <text>]` | Prints, as a JSON string, the text the tokenizer's normalizer makes of the input: its NFC for OLMoE, the text unchanged for SmolLM2 |
| `info --model <file.safetensors>` | Lists the tensors in a SafeTensors file |
| `serve --model <dir> [--model-id <id>] [--host <ip>] [--port <port>]` | Serves the model over the OpenAI-compatible API below. Without `--model` its generation endpoints answer 501. `--parallel <n>` runs up to n requests at once, 1 by default and at most 64 (see Requests at once below). `--speculate <k>` drafts and checks tokens as `generate --speculate` does (see Speculation below) |
| `bench` | Times the int4 matrix-vector kernel on synthetic weights. That describes one kernel on one machine, not a model |
| `read-experts --model <dir> [--expert-cache <MiB>]` | Reads every expert of a mixture once from the drive, as `--expert-cache` reads them, computing nothing, and says how fast |

`run`, `generate`, `logits` and `serve` run the model on the CPU unless given `--device cuda`, which runs it on the first CUDA device, in an engine built with the CUDA path. Decoding is greedy unless `--temperature <t>` is given, with optional `--top-k`, `--top-p` and `--seed`. The model directory needs `config.json`, `tokenizer.json`, and `model.safetensors` or the shards `model.safetensors.index.json` names, each tensor of which must be in the shard it names. A config that asks for something the engine does not implement is refused on load, and the error says what it was: rope scaling, attention or MLP biases, another activation, clipped query, key and value activations, or expert weights renormalized over those a token uses.

When a step cannot run, `run`, `generate` and `logits` stop with status 1 and say why, as far as the engine knows: a token outside the model's vocabulary, which `--ids` can give; a key-value cache already full, which `--ctx` can make larger, up to the model's maximum; or else the forward pass failing at a position, after the engine's own line saying what failed there, a CUDA error or an expert the drive did not give. After a CUDA error that leaves the GPU unable to run anything more in the process, the line says that too, and which error it was.

For tests of a mixture of experts, `logits` and `generate` take `--router-out <file>`, which writes every layer's router logits at every position run, `--experts-out <file>`, the experts the router chose there, and `--experts-in <file>`, experts to send each position's token through in place of those, weighted as the router weighs them: positions x layers x n, little-endian, float32 for logits and int32 for experts. The router's own choice is still what `--experts-out` writes, so a run with its experts pinned also says where the router would have gone.

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
- **Requests at once.** A server started with `--parallel <n>` runs up to n requests at once, each in a sequence of the key-value cache of its own, so its cache takes n times the memory; more wait their turn, in the order they came. Each round it runs every running request's next token together: on a GPU in passes of up to 8 tokens that read each weight once, on the CPU a step at a time, which is no faster than one request at a time. Every token's logits are the ones a step of that request alone would give, bit for bit, so running together changes no response. With `--no-prefix-cache`, a response is the one a server running one request at a time gives: text, finish reason, usage, stream events, a seeded sample, a failed step. `tests/parallel-serve.test.mjs` checks this. With reuse on, a request may find a longer or shorter prefix than it would one at a time, which changes `cached_tokens` and, on a GPU, can move logits by float32 rounding, as reuse itself can (see Prefix reuse). Requests that sample, or mask for JSON mode, then take their tokens on up to 8 threads at once, each with a sampler of its own; each draws from its own seed's stream, so which thread takes a token changes nothing. A new request's prompt runs whole when nothing else is decoding. While others are, it runs in pieces between their rounds, 512 tokens at a time on a GPU and 8 on the CPU, so they wait for a piece rather than the whole prompt. A prompt split into pieces of more than 32 tokens runs exactly as it does whole: its keys, values and logits are the same, bit for bit, on either device. Each connection is served on a thread of its own, up to 64 at once, so a client that stalls or reads slowly holds up no other.
- **Prefix reuse.** Each sequence keeps the key-value cache of the last request it served. A request goes to the free sequence holding the longest prefix of its prompt, reuses that prefix, and recomputes at least the last prompt token. A conversation sent back with one more turn reuses its history, and with `--parallel` above 1 two clients taking turns keep their own. Reuse only skips work. On the CPU, a reply, and its usage apart from `cached_tokens`, is exactly what a server started with `--no-prefix-cache`, which turns reuse off, returns for the same request. On the GPU the tokens of a prompt run together, while a reused prefix may hold keys and values computed a token at a time, so the two servers' logits can differ by float32 rounding, and a reply can differ only where two tokens' logits are that close.
- **Speculation.** A server started with `--speculate <k>` drafts up to k tokens after each one it takes, where the text's last two or three tokens occur earlier in it, and checks them together; on a GPU in one pass, on the CPU a step at a time. With several requests running, drafts take only the rows a round's passes have room for beyond one a request. Each token is still taken from logits a step would give, bit for bit, through the JSON mask, the stop sequences and the sampler in turn, and the cache is left holding what a token at a time would leave, so no response changes: text, finish reason, usage, stream events, a failed step. `tests/speculate-serve.test.mjs` checks this against a server without it. The server prints one line a request to stderr saying how many drafts it took.
- **A step that fails.** If the model fails to run a step, the request ends with an error, never with the reply so far. A response not yet begun is a 500 of type `server_error`. A stream has already sent its 200, so it ends with an event carrying the same error, in place of the final chunk, with no usage and no `[DONE]` after it. The next request reuses only the tokens whose steps ran. On the GPU a step fails when the device reports an error, which the server prints to stderr. The tests make a step fail with a hook instead (see [Tests](#tests)).
- **A GPU that can run nothing more.** Some errors leave the device unable to run anything more in the process: CUDA calls them sticky (an illegal memory access, a kernel that faulted or ran too long, and others), and only a new process can use the device again. After a step fails on the GPU, the server asks the device for further work, a wait on its stream; if that fails too, the device is lost. Every request running then ends with the error above, and every request waiting for its turn, or sent in the quarter of a second before the server notices, gets a 503 with code `device_lost` that names the CUDA error. The server stops listening, lets the connections still open finish, and exits with status 75 (`EX_TEMPFAIL` in `sysexits.h`), so that whatever started it (systemd, Docker, Kubernetes, a shell loop) can start it again. An error that is not sticky leaves the server serving.
- **Limits.** One request at a time unless `--parallel` says more, and each running request holds a sequence of the cache from its first token to its last. The context is the model's maximum, at most 4096 tokens, or `--ctx`, for each sequence. A request that would not fit is refused, not cut short. There is no authentication, so the server listens on 127.0.0.1 unless `--host` says otherwise, and warns if it does.

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

`tests/engine-server.test.mjs`, `tests/reference.test.mjs`, `tests/serving.test.mjs`, `tests/reuse-and-json.test.mjs`, `tests/step-failure.test.mjs`, `tests/step-failure-cli.test.mjs`, `tests/speculate.test.mjs`, `tests/speculate-serve.test.mjs` and `tests/parallel-serve.test.mjs` run the built engine when they find one, and the last eight also need the model files. Without them those tests are skipped, and say why. CI builds the engine and runs the unit tests on Linux, macOS and Windows, fetches the model, and requires all nine engine tests to pass.

`tests/step-failure.test.mjs` starts each of its servers with `VITNA_TEST_FAIL_STEP=<position>`, a hook for tests only, which makes the step at that position fail once, in whichever request reaches it first, before it computes anything. It is how the tests check what a client sees when a step fails, with no GPU error to cause one. A server started with it says so on stderr. With `VITNA_TEST_LOSE_DEVICE=1` as well, that step loses the device too, as a sticky CUDA error does, which is how the file checks, on either device, that the server then answers what is open and exits with status 75. `run`, `generate` and `logits` honour both too, and say so on stderr: `tests/step-failure-cli.test.mjs` sets them to check that a step failing in the prompt, in the reply or in a pass over drafted tokens is named as the forward pass failing there, after the engine's own line, never as `--ctx`, and that a lost device is named as one. Its other tests need no hook: a token outside the vocabulary, given with `--ids`, and a prompt or a reply longer than `--ctx`. `tests/parallel-serve.test.mjs` starts its servers with `VITNA_TEST_LOGITS=1`, which makes each response that is not streamed name, in an `x-vitna-test-logits` header, a hash of every row of logits its tokens were taken from, and requires the hashes to agree as well as the replies: a change too small to move a token would leave the replies alike. `tests/reuse-and-json.test.mjs` likewise starts one server with `VITNA_TEST_NO_MASK_CACHE=1`, which makes JSON mode find its mask anew at every step instead of keeping the mask it found for each state of the object, and requires the same JSON-mode replies from it as from a server that keeps them.

A test that talks to servers, when it fails, adds to its error what each server it started printed on stderr while the test ran, and whether the server has exited since (`tests/server-stderr.mjs`, itself checked by `tests/server-stderr.test.mjs`). When a `before` hook fails, every test of its file fails with the hook's one error, which carries one report, of all each server printed since it started. That is where the engine says why a step failed, a CUDA error on the GPU for one, which the 500 a client sees does not. It takes Node 22.14 or later, where node:test tells a hook how its test ended; on an older Node a failure is reported as before.

`tests/reference-moe.test.mjs` checks OLMoE's fixture with nothing but the repository, so `npm test` runs that much everywhere. Its comparisons with the engine need the engine and the model: the tokenizer's needs only `tokenizer.json`, which CI fetches on its own (`node scripts/fetch-model.mjs olmoe-1b-7b --only tokenizer.json`) and requires to match, with `VITNA_REQUIRE_MOE_TOKENIZER=1`; the rest need all 13.8 GB, which CI does not fetch, and are required with `VITNA_REQUIRE_MOE=1`. Among those, the experts read from the drive into a cache of 256 MiB must give the mapped run's logits and greedy tokens byte for byte. With `VITNA_DEVICE=cuda` they run the model on the GPU, one engine at a time, since each takes what the device has free for its experts; for the same reason they run there only when `VITNA_REQUIRE_MOE=1` asks for them, with the file run alone, as other files' servers would find no memory beside them. Three more run there only: with the GPU's cache of experts as large as the device allows, at 256 MiB, and at 256 MiB fed from a cache of 256 MiB in memory, the logits and greedy tokens must be the same byte for byte; a cache there too small for twice the experts a token goes through must be refused; and tokens run together as rows must give the logits and tokens of every token run alone, byte for byte, which the hook `VITNA_TEST_NO_ROWS=1` makes the engine do. The expert cache's own unit tests, in `engine/tests/`, need no model and run in CI on all three systems, each with its own direct I/O. `engine/tools/check_nfc.py` compares the engine's NFC with Python's on every code point and on random strings; it needs Python and a model whose normalizer is NFC, and CI does not run it.

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

Checked again on 2026-09-30, on the same machine with the same commands, at the merge of [#25](https://github.com/convexityos/vitna-anchor/pull/25), after [#13](https://github.com/convexityos/vitna-anchor/pull/13) to #25 changed how the forward pass runs on the GPU. All six tests passed: the file gained a sixth with [#15](https://github.com/convexityos/vitna-anchor/pull/15), which compares a prompt whose tokens run together with the CPU path at every position. The largest logit difference was 2.25e-4, and 1.65e-4 for logsumexp. 192 of 192 greedy tokens were equal, and the top logits at each step differed by at most 4.77e-5. The prompt run together, 565 positions, differed from the CPU path by at most 3.09e-4.

The GPU kernels were then rewritten for speed: CUDA graphs, fused kernels, and attention split across blocks by position, each position scored by a lane of its own. The same commands on the same machine, on the same day, passed again, with a largest difference of 2.25e-4, 1.65e-4 for logsumexp, 192 of 192 greedy tokens equal, and at most 5.15e-5 between the top logits at a step.

On the GPU the arithmetic is float32 on CUDA cores. No tensor cores are used, so TF32 does not apply, and the build does not pass `--use_fast_math`, so division, square root and `expf` keep their accurate forms. nvcc's default fused multiply-add is on; the CPU path's AVX2 matrix-vector product uses FMA too.

## How gate A5's forward pass was checked

By hand, on 2026-10-01, on the machine that checked A4 (AMD Ryzen 7 3700X, 64 GB, Windows 11, Node.js 24.19.0), with the engine built by MSVC 19.44 through CMake. CI does not run it: its runners would have to fetch 13.8 GB of weights for every run, and hold them. CI does check OLMoE's tokenizer, which needs only `tokenizer.json`.

```powershell
cmake -B engine/build -S engine
cmake --build engine/build --config Release
node scripts/fetch-model.mjs olmoe-1b-7b
$env:VITNA_REQUIRE_MOE = "1"
node --test --test-reporter=spec tests/reference-moe.test.mjs
```

Thirteen of its fifteen tests passed, with the six prompts run side by side; the other two run only on a GPU, with `VITNA_DEVICE=cuda` (below). The tolerances are the ones gate A1 set for logits and step 1 of A5 set for routing, before the engine could route a token:

- The engine's tokenizer gave the ids of the model's own `tokenizer.json` for all 60 strings, NFC and the added tokens included.
- With each position's experts pinned to the reference's, which a near-tie decided the other way could otherwise make impossible to compare, every compared logit was within 1e-2 of the reference's: the largest difference was 4.96e-5, and 3.03e-5 for logsumexp. The router logits differed by at most 2.25e-5, against a band of 1e-3.
- Greedy decoding with the experts pinned gave 192 of 192 tokens equal, the top logits at each step within 2.67e-5.
- The engine's own router chose the reference's 8 experts in all 6,848 routing decisions, prompts and greedy tokens alike, the two that came within 1e-5 of a tie included. Where it chooses as the pins do, a pinned run is the run it makes unpinned, bit for bit, and unpinned it decoded the same 192 tokens.
- Pinning the runner-up in place of an expert moves the logits after it: by 9.36e-2 at the closest call in the shortest prompt, where `reference/routing_sensitivity.py` measures transformers moving them by 9.357e-2 with the same experts pinned. So the pins are what the tokens go through, and the tests that pin the reference's own experts do not pass because the pins are ignored.
- Read from the drive into a cache of 256 MiB (`--expert-cache`), nearly every expert read as it was wanted or guessed, the experts gave the mapped run's logits and greedy tokens byte for byte, on two of the prompts. How fast that runs is [below](#speed-with-the-experts-read-from-a-drive).

The same file checked the GPU path the same day, on the same machine with the GPU that checked A4 (NVIDIA GeForce RTX 3070, 8 GB, compute capability 8.6, driver 591.86; CUDA 13.1.1), the engine built with the CUDA path:

```powershell
cmake -B engine/build -S engine -DVITNA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build engine/build --config Release
$env:VITNA_REQUIRE_MOE = "1"; $env:VITNA_DEVICE = "cuda"
node --test --test-reporter=spec tests/reference-moe.test.mjs
```

All fifteen tests passed, in 463 s with the engines run one at a time, against the same fixture and tolerances:

- With the experts pinned, the largest logit difference was 4.77e-5, and 2.67e-5 for logsumexp. The router logits differed by at most 2.81e-5, and the router chose the reference's experts in all 3,872 decisions at the prompts' positions.
- Greedy decoding with the experts pinned gave 192 of 192 tokens equal, the top logits at each step within 2.48e-5.
- Unpinned, its routing was the reference's in all 6,848 decisions, and it decoded the same 192 tokens.
- Pinning the runner-up at the closest call moved the logits after it by 9.36e-2, as on the CPU.
- Copied from a cache of 256 MiB in memory, read from the drive, rather than from the mapped checkpoint, and with the GPU's own cache of experts as large as the device allowed (5,544 MiB that day), at 256 MiB, and at 256 MiB fed from 256 MiB in memory, every position's logits and 16 greedy tokens with theirs were the same, byte for byte, on two of the prompts.
- A cache on the GPU of 100 MiB, too small for twice the 8 experts a token goes through, was refused, saying it needs 192 MiB.

Checked again at the merge of [#35](https://github.com/convexityos/vitna-anchor/pull/35), which followed [#33](https://github.com/convexityos/vitna-anchor/pull/33)'s change to how a failed step on the GPU is handled, on a tree identical to main's after it (tree `59cc9ed`), on the same machine, with both engines built as above and `VITNA_REQUIRE_REFERENCE=1` added: the whole suite on the CPU passed 77 of its 81 tests, the other 4 needing a GPU; on the GPU, `tests/reference-moe.test.mjs` alone passed 15 of 15 with the figures above, and the whole suite 71 of 81, the other 10 skipped: the OLMoE file's 8 that need a model on the device, which on a GPU run only when `VITNA_REQUIRE_MOE=1` asks for them, and 2 that need an engine without the CUDA path or a machine without a GPU.

## Speed with the experts read from a drive

Measured on 2026-10-01 on the machine above: AMD Ryzen 7 3700X, 64 GB of memory, Windows 11, the engine built by MSVC 19.44 through CMake, and the model on a Crucial P5 500 GB (CT500P5SSD8), an NVMe drive. One thread computes; four read. The commands:

```powershell
node scripts/bench-experts.mjs --tokens 64 --runs 3
.\engine\build\Release\vitna-anchor.exe read-experts --model models\olmoe-1b-7b
```

`bench-experts.mjs` decodes 64 greedy tokens after "The capital of France is", three times at each size of the expert cache, and reports the median of the 63 steps it times. The cache starts empty in every run, so the figures include the reads that fill it, and "MiB read a position" is what a run read over all 68 of its positions, the prompt's included. "Mapped" is the same run with nothing streamed, every weight read through the operating system's file cache, warm.

| Expert cache | ms a token | Tokens a second | MiB read a position | Already read when asked for | Being read for a prefetch then |
|---|---|---|---|---|---|
| Mapped, 13.8 GB in the file cache | 163.7 | 6.11 | | | |
| 16 GiB, every expert | 160.6 | 6.23 | 140 | 94% | 5% |
| 8 GiB | 172.6 | 5.80 | 175 | 94% | 4% |
| 4 GiB | 450.2 | 2.22 | 657 | 74% | 18% |
| 2 GiB | 786.1 | 1.27 | 1,152 | 49% | 38% |
| 1 GiB | 997.8 | 1.00 | 1,455 | 32% | 52% |
| 512 MiB | 1,152.9 | 0.87 | 1,677 | 19% | 63% |

`read-experts` read all 1,024 experts, 12,292 MiB, at 1,726 to 1,748 MB/s in three passes. Below 8 GiB, decoding runs at about that rate: a token takes about as long as reading its share does, 1,677 MiB of it at 512 MiB. From 8 GiB the cache holds nearly every expert the run uses, and the time is the computing's: one thread, as without the cache. The lookahead named 80.4% of the experts the layers went on to use, in every run. These figures are this machine's, with this drive and one thread computing; another drive, more threads or the GPU would change them.

## Speed on a GPU

Measured on 2026-10-01 on the same machine, with the GPU that checked A4: an NVIDIA GeForce RTX 3070, 8 GB, driver 591.86, on a PCIe 4.0 x16 link. The desktop ran on the same GPU, and other programs kept the CPU about 16% busy. The commands:

```powershell
node scripts/bench-experts.mjs --device cuda --tokens 64 --runs 5
node scripts/bench-experts.mjs --device cuda --tokens 64 --runs 5 --no-mapped --caches 16384,4096 --gpu-caches default
```

The same 64 greedy tokens after "The capital of France is", five times at each size of the GPU's cache of experts, and the median of the 63 steps timed. The cache on the GPU starts empty in every run, and so does the one in memory; loading, which locks the memory the copies come from, is not timed. "On the GPU when wanted" counts the experts a layer wanted that the GPU held already, those its guess had brought included, and "MiB copied a position" is what a run copied over all 68 of its positions.

| Copied from | Cache on the GPU | ms a token | Tokens a second | On the GPU when wanted | MiB copied a position |
|---|---|---|---|---|---|
| Mapped, 13.8 GB in the file cache | 5,544 MiB, 462 of the 1,024 experts: what the device allowed | 16.3 | 61.3 | 88% | 314 |
| Mapped | 4,092 MiB, 341 experts | 22.7 | 44.0 | 82% | 478 |
| Mapped | 2,040 MiB, 170 experts | 39.9 | 25.0 | 67% | 898 |
| Mapped | 1,020 MiB, 85 experts | 50.4 | 19.9 | 58% | 1,167 |
| 16 GiB in memory, read from the drive | 5,544 MiB, 462 experts | 54.1 | 18.5 | 85% | 312 |
| 4 GiB in memory, read from the drive | 5,544 MiB, 462 experts | 115.3 | 8.7 | 83% | 310 |

With the experts in memory the GPU decodes these tokens about ten times as fast as the CPU does ([above](#speed-with-the-experts-read-from-a-drive): 163.7 ms a token, mapped). Its time follows what it copies over the bus: the smaller its cache, the more it copies, and the slower it runs. Fed from the drive, it is slower again, because those runs read from the drive as well. With 16 GiB every expert fits in memory, but each is read once, and 64 tokens do not get past that; with 4 GiB, an expert the GPU lacks is mostly not in memory either, and is read at the drive's 1.7 GB/s. Of the copies made for the lookahead's guess, 93% to 96% were then wanted. That is with only the half of the guess the router ranks highest copied: the lower half was wrong too often to pay for its copies. A variant of the engine built to try it, not in this repository, decoded at 17.1 ms a token copying all eight guessed experts, 16.1 copying four and 18.0 copying none; with a 2 GiB cache, 45.6, 38.5 and 43.5. Locking the mapped checkpoint for the GPU adds a few seconds to loading, every time the engine starts.

A prompt's tokens run together as rows on the GPU, a chunk of up to 1,024 a layer at a time. Measured the same day on the same machine, against an engine built from main before rows, the two taking turns; another session's work kept the CPU busy too, 46% on average over the run, this benchmark's own engines included:

```powershell
node scripts/bench-prefill.mjs --device cuda --model models/olmoe-1b-7b --engine <main's engine> --engine <this engine> --prompt-tokens 8,100,500,2000 --runs 3
```

`bench-prefill.mjs` runs `generate --max-new 2 --greedy --timing` over the first n tokens of this README, with the context of 4,096 positions the engine takes by default, and reports the median of three runs of the prompt's time, from its first token to the next token's logits on the host; loading is not timed. Both engines chose the same two tokens at every length.

| Prompt | A token at a time | As rows | Faster |
|---|---|---|---|
| 8 tokens | 275.6 ms | 214.3 ms | 1.29x |
| 100 tokens | 1,929 ms | 482 ms | 4.00x |
| 500 tokens | 12,544 ms | 728 ms | 17.2x |
| 2,000 tokens | 53,071 ms | 2,994 ms | 17.7x |

A token at a time, each wanted the experts it was routed to copied to the GPU if it lacked them. As rows, a chunk's tokens share the copies: an expert runs once a layer for every row routed to it. What is left is copying each chunk's experts once a layer, and the rows' own arithmetic, which reads each expert's weights once for every 8 rows routed to it, so that every row's sums stay its step's.

## Measurements

A performance figure appears in this repository only with the hardware and the command that produced it, and it says whether it was measured or estimated. The first published are gate A5's, [above](#speed-with-the-experts-read-from-a-drive), on the CPU and [on a GPU](#speed-on-a-gpu). `scripts/bench-decode.mjs` times decoding with `generate --timing` and prints the machine and the command with every result, so that a figure, when there is one, carries both.

## License

Apache License 2.0. See [LICENSE](LICENSE).
