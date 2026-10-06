# vitna-anchor

[![CI](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml/badge.svg)](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)

An inference engine in C, with a Node.js command line, being built one gate at a time.

**What runs today.** The engine runs one small dense model, [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), on a CPU in float32. Its token ids, logits and greedy output match a pinned reference implementation, and CI checks that on Linux, macOS and Windows. Built with its CUDA path, it runs the same forward pass on an NVIDIA GPU, also in float32, and the same comparison passed there on one GPU, checked by hand ([below](#how-gate-a4-was-checked)). CI compiles the CUDA path but has no GPU to run it on. It serves the model over an OpenAI-compatible HTTP API, `/v1/chat/completions` and `/v1/completions`, streamed or not, with usage counted from the tokens it reads and produces. It also serves Anthropic's `/v1/messages` and OpenAI's `/v1/responses`, the APIs Claude Code and Codex speak: a conversation is written in the model's own chat template, Qwen3's byte for byte as transformers renders it, and the model's reasoning and tool calls come back in each API's own shape. Against Qwen3-30B-A3B on a GPU, Claude Code read and edited a file through it and Codex called its tools through it, checked by hand ([below](#how-gate-a11-was-checked)). A request reuses the key-value cache for the prefix it shares with the one before it. JSON mode masks, at every step, the logits of each token that could not continue a JSON object. It also runs a mixture of experts, [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924), on a CPU in float32, and matches that model's own pinned reference: the same tolerance on logits, the same experts chosen at every layer, and the same greedy output, checked by hand ([below](#how-gate-a5s-forward-pass-was-checked)) because its 13.8 GB of weights are more than CI holds. Told to, it reads that model's experts from the drive as the layers want them, with direct I/O, into a cache in memory of a size it is given, and the logits are then the same, bit for bit; how fast that runs was measured on one machine ([below](#speed-with-the-experts-read-from-a-drive)). Built with its CUDA path, it runs that model on an NVIDIA GPU too, which keeps as many of the experts as its memory holds and has the others copied in as the layers want them, from the mapped checkpoint or from that cache in memory. The same comparison passed there, checked by hand, and how fast it runs was measured on one GPU ([below](#speed-on-a-gpu)). Given one of the GGUF files OLMoE's authors publish, quantized to 8 or mostly 4 bits, it runs that instead (`--weights`), on the CPU and on the GPU, and matches a reference recorded on the same quantized weights, checked by hand ([below](#quantized-weights-from-a-gguf-file)); with every expert then fitting on that GPU, it decodes 3.5 times as fast as the BF16 model ([below](#speed-with-quantized-weights)). It runs Qwen3-30B-A3B, the model parity with Strata is measured on, from BF16 and from Qwen's own Q8_0 and Q4_K_M files, on the CPU and on that GPU, which holds under a third of its experts, and matches a reference recorded a layer at a time, checked by hand ([below](#how-gate-a9-was-checked)); on that GPU it decodes 1.6 to 2.8 times as fast as llama.cpp at its best setting for the same files, and reads a prompt 0.68 to 1.03 times as fast ([below](#speed-beside-llamacpp)). It reads a prompt of 32,768 tokens there, keeping the part of the key-value cache the GPU has no room for in the computer's memory with the logits unchanged, bit for bit ([below](#a-long-context)), at 182 tokens a second against llama.cpp's 469 ([below](#speed-with-a-long-context)). With `--precision fast`, an opt-in mode that rounds a prompt's products to float16 on the GPU's tensor cores, it reads that prompt at 521 tokens a second and one of 4K at 1,014, faster than llama.cpp at both, and moves the logits less from the reference than quantizing to Q8_0 does ([below](#faster-prompts-in-less-precision)). It also runs a text embedding model, [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), on a CPU in float32, and serves its embeddings at `/v1/embeddings`. Its token ids equal the model's own tokenizer's on every code point, and its embeddings are within 1e-5 of the model's pinned reference, the largest difference 3.05e-7; CI checks both on Linux, macOS and Windows. Nothing has been published to npm. From v0.2.0 a release carries an installer of one command for Windows and for Linux ([Install](#install)), and the engine builds from source as well.

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
| A7, quantized weights | GGUF checkpoints with F32, F16, BF16, Q8_0, Q4_K and Q6_K tensors run on the CPU and the GPU, and with OLMoE's official GGUFs at Q8_0 and Q4_K_M the logits match the pinned reference run on the same dequantized weights within 1e-2, with 192 of 192 greedy tokens equal | Passed, [#40](https://github.com/convexityos/vitna-anchor/pull/40) and [#41](https://github.com/convexityos/vitna-anchor/pull/41), on one machine, checked by hand ([reference/README.md](reference/README.md#quantized-weights-gate-a7)): from OLMoE's own Q8_0 and Q4_K_M files, on the CPU and on an NVIDIA GeForce RTX 3070, every tensor widened to gguf-py's bits by digest, the logits within the tolerance of a reference recorded on those weights, and 192 of 192 greedy tokens equal. With every expert then on that GPU, Q4_K_M decodes 3.5 times as fast as the BF16 model ([speed](#speed-with-quantized-weights)). CI checks the fixtures as recorded, not the model: its files are 4 to 7 GB. These gates and the ones after them are planned in [`docs/PARITY.md`](docs/PARITY.md) |
| A8, the CPU computes what the GPU lacks | Experts the GPU's cache lacks run on the CPU from memory, at the same time as the GPU runs the others, and logits are byte-identical to the GPU-only path at every cache size | Passed, [#42](https://github.com/convexityos/vitna-anchor/pull/42), on one machine, checked by hand: with `--cpu-experts`, the experts the GPU's cache lacks run on the CPU in the GPU kernels' order, and every expert gives the same bytes on either device (`expert-check`, all of OLMoE's and Qwen3-30B-A3B's, from BF16, Q8_0 and Q4_K_M), so no cache size changes a logit; through 1 GiB of the GPU the logits and greedy tokens were the GPU alone's, byte for byte. 4 to 12% faster from BF16 and Q8_0 at small caches, and no faster at Q4_K_M, where copying an expert costs less than the CPU running it ([speed](#speed-with-the-cpu-beside-the-gpu)) |
| A9, Qwen3-30B-A3B | The model matches a pinned reference within 1e-2, with the reference's experts chosen and 192 of 192 greedy tokens equal, on the CPU and the GPU, from BF16, Q8_0 and Q4_K_M | Passed, [#45](https://github.com/convexityos/vitna-anchor/pull/45), on one machine, checked by hand ([below](#how-gate-a9-was-checked)): from BF16, Q8_0 and Q4_K_M, on the CPU and on an NVIDIA GeForce RTX 3070, the logits within 1.37e-4 of the reference's, its experts chosen in all 11,472 decisions, and 192 of 192 greedy tokens equal. On that GPU it decodes 1.6 to 2.8 times as fast as llama.cpp at its best on the same files, and reads a prompt 0.68 to 1.03 times as fast ([speed](#speed-beside-llamacpp)). CI checks the tokenizer and the fixtures, not the model: it is 61 GB |
| A10, long context | A 32K-token prompt runs with the part of the key-value cache the GPU cannot hold in page-locked host memory, and logits are byte-identical to the cache-on-GPU run where both fit | Passed, [#46](https://github.com/convexityos/vitna-anchor/pull/46), on one machine, checked by hand ([below](#how-gate-a10-was-checked)): from Q4_K_M on an NVIDIA GeForce RTX 3070, a 32,768-token prompt ran with the cache of 11 of the 48 layers in page-locked host memory, its logits the same byte for byte as with none of it on the GPU, and at 300 and 2,000 positions as with all of it there. It reads a prompt at 353 tokens a second at 4K and 182 at 32K, against llama.cpp's 573 and 469 ([speed](#speed-with-a-long-context)) |
| A11, the API surface | Anthropic's `/v1/messages`, tool calls and OpenAI's Responses API, each tested against recorded shapes | Passed, [#49](https://github.com/convexityos/vitna-anchor/pull/49), on one machine, checked by hand ([below](#how-gate-a11-was-checked)) and in CI: Anthropic's Messages and OpenAI's Responses served beside chat completions, each conversation written in Qwen3's template byte for byte as transformers renders it, and the reply's reasoning and tool calls returned in each API's shape, whole and streamed. Every request Claude Code 2.1.283 and Codex 0.160.0 sent is recorded, read in CI and answered through the model by hand; the official Anthropic and OpenAI SDKs parsed every response; and against Qwen3-30B-A3B, Claude Code read and edited a file and Codex called its tools. No response was recorded from Anthropic's or OpenAI's own APIs, which needs their keys: the SDKs' types are what the responses are held to |
| A12, installing and a chat page | One command on Windows and on Linux picks a model that fits the machine, fetches it with checks, and starts the server, and a chat page talks to it | Passed, [#50](https://github.com/convexityos/vitna-anchor/pull/50), [#51](https://github.com/convexityos/vitna-anchor/pull/51) and [#52](https://github.com/convexityos/vitna-anchor/pull/52), in CI and by hand ([below](#how-gate-a12-was-checked)): release v0.2.0's installer, run as a person runs it on GitHub's hosted Windows and Linux runners, fresh machines with no GPU, chose the small model and said why, resumed a part fetched first, checked every file and started a server that answered. On this machine's RTX 3070 it chose Qwen3-30B-A3B at Q4_K_M with a 32K context, fetched its 18.6 GB into an empty folder and checked them, 340 s in all, and the server it started answered on the GPU, through the chat page as well |
| A13, AMD and several GPUs | The A7 and A9 comparisons pass on an AMD GPU and on a model split across two GPUs | Not started |
| A14, embeddings | One small open embedding model is pinned by revision and file hash, and token ids and embeddings for fixed inputs are recorded from a pinned reference implementation. The engine's token ids equal the model's own `tokenizer.json`'s, as the tokenizers library runs it, for every input and every code point; every value of every embedding is within a stated tolerance of the reference's; and the embeddings are served at `/v1/embeddings` with OpenAI's request and response | Passed, [#43](https://github.com/convexityos/vitna-anchor/pull/43). CI checks it on every push, and the Reference workflow checks the tokenizer over every code point. Numbered after A7 to A13, which [#39](https://github.com/convexityos/vitna-anchor/pull/39) plans. See [`reference/`](reference/README.md#an-embedding-model-gate-a14) |

## What is in the repository

"Matches the reference" means `tests/reference.test.mjs` compares it with the fixture in [`reference/`](reference/README.md), or for OLMoE `tests/reference-moe.test.mjs` with that model's fixture. "Tested" means the repository's own tests exercise it, on inputs they build themselves: the C unit tests in `engine/tests/` check each part against published test vectors, a formula evaluated in double precision, or an input worked by hand.

| Component | Where | State |
|---|---|---|
| Llama forward pass in float32: embeddings, RMSNorm, grouped-query attention over a key-value cache, half-split rotary embeddings, SwiGLU MLP, tied output layer | `engine/src/model.c`, `engine/src/ops.c` | Matches the reference: every compared logit within the stated tolerance of 1e-2, and 192 of 192 greedy tokens equal. The largest difference on the machine that recorded the reference was 2.2e-4 |
| The same forward pass on an NVIDIA GPU, in float32: five fused kernels a layer, attention split across blocks by position, and each token a replay of CUDA graphs; the weights uploaded once, the key-value cache on the device, logits copied back only when a step asks for them | `engine/src/model_cuda.cu` | Matches the reference on one GPU, checked by hand: every compared logit within 1e-2, the largest difference 2.25e-4, and 192 of 192 greedy tokens equal. CI compiles it and does not run it |
| OLMoE's mixture of experts in float32: an RMSNorm over all of each query and key projection, a router over 64 experts, the 8 with the largest weights each a SwiGLU MLP, weighted by the router's softmax without renormalizing; a checkpoint in shards, read through its index | `engine/src/model.c` | Matches its reference on the CPU, checked by hand: with every token's experts pinned to the reference's, every compared logit within the tolerance of 1e-2, the largest difference 4.96e-5, and 192 of 192 greedy tokens equal; the engine's own router chose the reference's experts in all 6,848 decisions, and unpinned it decodes the same 192 tokens. On the GPU too, checked by hand: the largest difference 6.29e-5 (4.77e-5 before gate A8 changed how the GPU computes the experts' activation), 192 of 192 greedy tokens equal pinned and unpinned, and the router the reference's in every decision. CI does not run it: the weights are 13.8 GB |
| Byte-level BPE tokenizer read from `tokenizer.json`, with NFC normalization and added tokens matched before and after it, as the tokenizers library does | `engine/src/tokenizer.c`, `engine/src/unicode.c` | Matches each model's own `tokenizer.json` on all of its reference strings, 48 for SmolLM2 and 60 for OLMoE, both in CI. Tested on vocabularies worked by hand. NFC equals Python's `unicodedata` (Unicode 15.0, where the tables come from) on every code point and 1.2 million random strings of marks, composites and jamo (`engine/tools/check_nfc.py`). Refuses tokenizer features it does not implement |
| Greedy decoding and seeded temperature, top-k and top-p sampling | `engine/src/sampler.c` | Greedy matches the reference. Sampling is tested for its proportions and its seed |
| Float32 matrix-vector product over F32, BF16 or F16 weights, with scalar, AVX2 and NEON paths | `engine/src/ops.c` | Tested against double precision |
| SafeTensors reader in C | `engine/src/safetensors.c`, `engine/src/json.c` | Tested: the `__metadata__` block is skipped, byte ranges are checked, and malformed files are refused. Reads the model in the A2 comparison, and OLMoE's three shards |
| GGUF reader in C (versions 2 and 3): the file's tensors listed under the names a Hugging Face checkpoint gives them, each expert of a stack as a tensor of its own, and its metadata checked against `config.json` | `engine/src/gguf.c`, `engine/src/model.c` | Tested on files made for the test: names, shapes, slices of a stack, and refusals of a permuted Llama file, an unknown tensor, an unknown type, a misaligned or out-of-range offset, a duplicate, a truncated file and another format. Reads OLMoE's official GGUF files ([below](#quantized-weights-from-a-gguf-file)) |
| ggml's Q8_0, Q4_K and Q6_K block formats widened to float32, in gguf-py's arithmetic, and matrix-vector products over them on the CPU | `engine/src/quant.c`, `engine/src/ops.c` | Tested on blocks built from the layout by hand; a quantized matrix multiplies as its widened float32 rows do, bit for bit. Every tensor of OLMoE's Q8_0 and Q4_K_M files widens to the bits gguf-py gives, checked by digest |
| The same block formats on an NVIDIA GPU: each kernel instantiated for the dtype of each matrix it reads, so a file that keeps some matrices at more bits than others (Q4_K_M's Q6_K value and down projections) runs as stored, and the expert cache's slots sized for the largest layer's experts | `engine/src/model_cuda.cu` | Checked by hand on one GPU: every tensor of both files widens to gguf-py's bits there too, read each of the three ways the kernels read weights (eight at a time, four, and one), which must agree. The model then matches the same fixtures as on the CPU ([below](#quantized-weights-from-a-gguf-file)). CI compiles it and does not run it |
| A mixture of experts on the GPU with the CPU beside it (`--cpu-experts`): the experts the GPU lacks shared between copies to it and CPU threads, by what each costs as measured, the CPU computing each row in the GPU kernel's order and the activation in arithmetic both devices round alike, and every output added in the GPU's order | `engine/src/warp.c`, `engine/src/exact.c`, `engine/src/pool.c`, `engine/src/model_cuda.cu` | Checked by hand on one GPU: `expert-check` gives the same activations and outputs on both devices, byte for byte, for all 1,024 experts in BF16, Q8_0 and Q4_K_M, and with the CPU beside it the logits are the GPU alone's, byte for byte, at every cache size tried, from the mapped file and from a cache in memory. The unit tests, which CI runs, check the CPU's order and its threads. How fast it runs was measured ([below](#speed-with-the-cpu-beside-the-gpu)) |
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
| HTTP API: OpenAI's chat and text completions and Responses, and Anthropic's Messages, streamed or not, usage from tokens, tool calls and reasoning in each one's shape | `engine/src/api.c`, `engine/src/server.c` | A greedy completion through it equals the reference's greedy output, with prompt and completion tokens counted exactly, on all six reference prompts. Streaming, stop sequences, seeds, refusals, a client hanging up mid-stream and a step that fails are tested, the last with a hook that makes a step fail on the CPU. Each API's responses and stream events, tool calls, reasoning, forced calls, prefill and limits are tested with a hook that gives the reply, and every request Claude Code 2.1.283 and Codex 0.160.0 sent, as recorded, is read; the official Anthropic and OpenAI SDKs parsed every response, and both clients worked against it, checked by hand ([below](#how-gate-a11-was-checked)) |
| The installer's engine half: the machine (`hardware`), the plan for it (`plan`) from a catalogue built in, and the check of what was fetched (`verify`) | `engine/src/install.c`, `engine/src/catalog.c`, `install/catalog.json` | Plans for machines described in JSON tested, from a GPU of 8 GB and 64 GB of memory to one nothing fits, with the reason each model is passed over; the check tested on files missing, short, long and changed, in CI |
| The installer scripts: the engine checked against the release's SHA-256, the model fetched with resume, every file checked, the server started | `install/install.sh`, `install/install.ps1` | Run end to end in CI on fresh Windows and Linux runners, in a folder named with `é` and `Ж`: from an engine built in the same run, and from each release as a person runs it. By hand on Windows, from release v0.2.0, into an empty folder on an RTX 3070 ([below](#how-gate-a12-was-checked)) |
| A chat page, built into the engine and served at `/`: streamed replies, reasoning folded away, Markdown rendered from escaped text | `engine/web/chat.html`, `engine/src/chat_page.c` | Its rendering, its reading of a stream and its handling of each chunk tested in Node, in CI, and the page served byte for byte under its policy. Used by hand against Qwen3-30B-A3B in a browser: a streamed reply with its reasoning, a second turn, and a reply stopped part way |
| Qwen3's chat template: a conversation, its tools and its tool calls written as transformers renders the model's own Jinja template, with tools and calls in JSON as Python writes it | `engine/src/chat.c` | Byte for byte as transformers 5.17.0 renders it on 13 conversations, in CI. Python's float repr and JSON writing tested against Python's own output |
| A reply in Qwen3's format read back as it streams: its reasoning, its text and its tool calls | `engine/src/chat.c` | Tested on replies worked by hand, each fed whole and a byte at a time, with the same parts either way |
| Requests in Anthropic's and OpenAI's shapes converted into one conversation | `engine/src/convert.c` | Tested on requests worked by hand in each shape, and on Claude Code's and Codex's as recorded |
| Prefix reuse: a request keeps the key-value cache for the tokens it shares with the one before | `engine/src/api.c`, `engine/src/model.c` | Tested: a repeated prompt reuses all but its last token and still returns the reference's greedy output, and replies equal those of a server that reuses nothing, greedy or sampled. On the GPU the same test passes, but equality holds only up to float32 rounding ([Serve the model](#serve-the-model)) |
| JSON mode: every token that could not continue a JSON object has its logit set to minus infinity before each choice | `engine/src/jsonpfx.c`, `engine/src/api.c` | The check is tested against the engine's JSON parser on 3,000 mutated objects, and on cases worked by hand. Replies parse as JSON objects, or are valid starts of one when `max_tokens` cuts them short |
| The Node.js command line's server | `runtime/anchor-run.mjs` | Tested: it serves no model, answers generation with 501, and names the engine's server |
| SafeTensors and GGUF v2/v3 header parsers | `runtime/ingest.mjs` | Tested on synthetic files |
| Checkpoint rewrite with every tensor at a 4096-byte offset | `runtime/ingest.mjs` | Tested on synthetic files |
| Block-wise INT4 and INT8 quantizer | `runtime/quantize.mjs` | Tested for round-trip error on random tensors |
| Round-robin striping of a file across directories | `runtime/stripe.mjs` | Tested for layout. Read throughput has not been measured |
| A BERT-architecture text encoder in float32 on the CPU: word, position and token-type embeddings, self-attention over every position of a text, post-norm residuals, a GELU MLP; several texts as the rows of one matrix product, none padded; pooled and normalized as the model's sentence-transformers files say | `engine/src/encoder.c` | Matches the reference: every value of every embedding within 1e-5, the largest difference 3.05e-7 pooled at `[CLS]` and 1.52e-7 by the mean. The same bits on any number of threads and however texts are grouped, tested |
| A BERT tokenizer read from `tokenizer.json`: BertNormalizer, BertPreTokenizer, WordPiece and `[CLS]` and `[SEP]` around, with the Unicode data of the three versions the tokenizers library takes it from (categories 8.0, NFD 9.0, lowercase 17.0) | `engine/src/wordpiece.c`, `engine/src/wordpiece_data.h` | Matches the model's own `tokenizer.json` on all 61 reference strings, in CI, and on all 1,112,064 code points between two letters and 200,000 random strings (`engine/tools/check_wordpiece.py`, run by the Reference workflow). Tested on a vocabulary worked by hand |
| `POST /v1/embeddings`: OpenAI's request and response, float or base64, usage from tokens | `engine/src/api.c` | Each embedding the `embed` command's, bit for bit, and within the reference's tolerance; requests at once and every refusal tested |

## Install

One command installs the engine and a model that fits the machine, from the latest release:

```powershell
irm https://github.com/convexityos/vitna-anchor/releases/latest/download/install.ps1 | iex      # Windows, in PowerShell
```

```bash
curl -fsSL https://github.com/convexityos/vitna-anchor/releases/latest/download/install.sh | sh  # Linux on x86-64
```

The installer fetches the engine built for the machine and checks it against the SHA-256 the release wrote into the script. It then asks the engine which model fits the machine (`vitna-anchor plan`), fetches that model's files from Hugging Face at their pinned revisions, resuming any part already there, has the engine check each against its pinned SHA-256 (`vitna-anchor verify`), and starts the server, opening its chat page. With an NVIDIA GPU of 8 GB and 32 GB of memory it installs Qwen3-30B-A3B at Q4_K_M, 18.6 GB, with a context of 16K, or 32K with 48 GB of memory; otherwise SmolLM2-360M-Instruct, 724 MB, a small model for trying the server on the CPU, not a coding assistant. The plan says which, and why. Run again, it fetches only what is missing; `serve.sh` or `serve.cmd`, beside what it installed, starts the server later. `VITNA_HOME`, `VITNA_MODEL` and `VITNA_PORT` change where it installs, the model, and the port. The catalogue it chooses from is [`install/catalog.json`](install/catalog.json), built into the engine.

CI runs the installer end to end on GitHub's hosted runners, which are fresh virtual machines, on Windows and on Linux ([`install.yml`](.github/workflows/install.yml)). On a pull request that changes it, the engine is built in the same run. Once a release is published, the release workflow starts it on the release, which it installs as a person would. Either way it plans the small model for a machine with no GPU, resumes a part fetched first, checks every file and serves, and the server answers with its chat page and a reply. It installs into a folder named with `é` and `Ж`, as a user's name may be: on Windows the first is in the ANSI code page and the second is not, and the engine opens both because it is a UTF-8 program there (below).

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

This build embeds [`engine/windows/utf8.manifest`](engine/windows/utf8.manifest), which makes the engine a UTF-8 program on Windows 10 version 1903 and later, so it opens a path with any letters in it. The releases are built this way. The build below does not embed it, and its engine reads paths in the ANSI code page, where a letter outside that code page becomes a question mark.

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

With `--cpu-experts <threads>` as well (gate A8), the experts a token wants that the GPU lacks no longer all wait to be copied in. The GPU starts on the experts it holds; of those it lacks, some are copied in and run there, and the CPU runs the rest from memory at the same time, on that many threads. How many go each way follows what an expert has cost each side lately, measured as they run: the CPU takes as many as let the layer finish soonest, none when copying them all is sooner, and the least used ones, so that the experts copied, which then stay on the GPU, are the ones most worth keeping. A side left out for 64 layers is given one expert, so that neither cost goes stale. The CPU computes an expert in the GPU's own arithmetic: each row's sum in the order a warp adds it, 32 lanes of eight-weight chunks and then the warp's tree (`engine/src/warp.c`), and the activation's `exp` in operations both devices round alike (`engine/src/exact.c`), since CUDA's `expf` ends in a hardware approximation no CPU reproduces. The GPU then weights and adds every expert's output in the order it adds its own, so the logits are the GPU alone's, bit for bit, whichever experts ran where. Tokens that run together as rows (a prompt's, drafted ones, requests at once) still copy in what they lack: this is for decoding a token at a time. `expert-check` runs experts on both devices and compares what they give:

```bash
./engine/vitna-anchor generate --model models/olmoe-1b-7b --prompt "The capital of France is" --max-new 32 --greedy --timing --device cuda --gpu-expert-cache 1024 --cpu-experts 8
./engine/vitna-anchor expert-check --model models/olmoe-1b-7b --device cuda --count 1024
```

### Quantized weights from a GGUF file

With `--weights <file.gguf>`, the weights come from a GGUF file rather than the model directory's SafeTensors files, and may be quantized: Q8_0, Q4_K and Q6_K, as llama.cpp writes them, beside F32, F16 and BF16. `--model` still gives `config.json`, which the file's metadata must agree with, and the tokenizer. A quantized weight is widened to float32 as it is used, block by block, to the bits gguf-py's `dequantize` gives, and the arithmetic after that is the same as for a float32 checkpoint. On the CPU and, in an engine built with the CUDA path, with `--device cuda` on the GPU, which widens to the same bits and keeps the experts in their stored format in its cache, so that more of them fit and each copy is smaller.

```bash
node scripts/fetch-model.mjs olmoe-1b-7b-gguf          # OLMoE's own Q8_0 and Q4_K_M files, 11.6 GB
./engine/vitna-anchor run --model models/olmoe-1b-7b --weights models/olmoe-1b-7b-gguf/olmoe-1b-7b-0924-q4_k_m.gguf --prompt "The capital of France is"
./engine/vitna-anchor weights-sha256 --model models/olmoe-1b-7b --weights models/olmoe-1b-7b-gguf/olmoe-1b-7b-0924-q4_k_m.gguf
```

`weights-sha256` prints the SHA-256 of every tensor as the engine widens it, which [`reference/record_gguf.py`](reference/README.md#quantized-weights-gate-a7) records for gguf-py's widening; `tests/reference-gguf.test.mjs` requires them equal.

### Qwen3-30B-A3B

Gate A9's model, [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B): 128 experts in each of 48 layers, 8 a token, a QK-norm over each head alone, the 8 experts' weights renormalized to sum to one, and Qwen's own tokenizer split. It is 61.1 GB in BF16, and Qwen publishes it as GGUF too:

```bash
node scripts/fetch-model.mjs qwen3-30b-a3b        # 61.1 GB in BF16, with its config and tokenizer
node scripts/fetch-model.mjs qwen3-30b-a3b-gguf   # Qwen's own Q8_0 and Q4_K_M files, 51.0 GB
./engine/vitna-anchor run --model models/qwen3-30b-a3b --weights models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf --prompt "The capital of France is" --device cuda
./engine/vitna-anchor run --model models/qwen3-30b-a3b --prompt "The capital of France is" --device cuda --expert-cache 16384
```

On a GPU the experts are copied from memory the device has locked. A GGUF file is locked as it is, but the whole 61.1 GB of BF16 is more than a machine with 64 GB can lock, so from BF16 the experts come through a cache in memory read from the drive (`--expert-cache <MiB>`).

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
| `weights-sha256 --model <dir> [--weights <file.gguf>]` | Prints, as JSON, the SHA-256 of every tensor's values widened to float32, under its Hugging Face name, each layer's experts together under the name of their stack |
| `expert-check --model <dir> --device cuda [--weights <file.gguf>] [--count <n>]` | Runs n experts of a mixture (64 by default, spread over the layers; layers times experts takes every one) on the GPU and on the CPU in the GPU's arithmetic, for residuals it makes up, and prints, as JSON, how many gave the same activations and outputs, byte for byte; exits 1 if any differs |

`run`, `generate`, `logits` and `serve` run the model on the CPU unless given `--device cuda`, which runs it on the first CUDA device, in an engine built with the CUDA path. Decoding is greedy unless `--temperature <t>` is given, with optional `--top-k`, `--top-p` and `--seed`. The model directory needs `config.json`, `tokenizer.json`, and `model.safetensors` or the shards `model.safetensors.index.json` names, each tensor of which must be in the shard it names. The engine runs Llama (`llama`), OLMoE (`olmoe`) and Qwen3's mixture of experts (`qwen3_moe`). A config that asks for something the engine does not implement is refused on load, and the error says what it was: rope scaling, attention or MLP biases, another activation, clipped query, key and value activations, a sliding window, or dense layers among a mixture's.

When a step cannot run, `run`, `generate` and `logits` stop with status 1 and say why, as far as the engine knows: a token outside the model's vocabulary, which `--ids` can give; a key-value cache already full, which `--ctx` can make larger, up to the model's maximum; or else the forward pass failing at a position, after the engine's own line saying what failed there, a CUDA error or an expert the drive did not give. After a CUDA error that leaves the GPU unable to run anything more in the process, the line says that too, and which error it was.

For tests of a mixture of experts, `logits` and `generate` take `--router-out <file>`, which writes every layer's router logits at every position run, `--experts-out <file>`, the experts the router chose there, and `--experts-in <file>`, experts to send each position's token through in place of those, weighted as the router weighs them: positions x layers x n, little-endian, float32 for logits and int32 for experts. The router's own choice is still what `--experts-out` writes, so a run with its experts pinned also says where the router would have gone.

### A long context

`--ctx <n>` sets how many positions the key-value cache holds, up to the model's own maximum: 40,960 for Qwen3-30B-A3B. The cache is float32, 192 KiB a position for that model, so 32,768 positions are 6 GiB, which an 8 GB GPU cannot hold beside the rest of the model. On a GPU, a mixture of experts keeps the cache of as many layers as the device has room for, beside the rest of the model and the least expert cache, and the other layers' in page-locked memory on the host: each such layer's attention runs on a copy on the device, the positions it reads copied in first and the ones it wrote copied back after. The engine says on stderr when it does that, and `--gpu-kv-layers <n>` sets how many layers' cache the GPU holds instead. Wherever the cache is, the logits are the same, bit for bit. A Windows command line holds about 4,000 ids, so `--ids-file <file>` and `--prompt-file <file>` read them, or the prompt's text, from a file:

```bash
./engine/vitna-anchor generate --model models/qwen3-30b-a3b --weights models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf --prompt-file long.txt --ctx 32784 --max-new 64 --device cuda
```

A mixture's token that attends to more than 1,024 positions takes attention in another arithmetic on the GPU, one a prompt's tokens can share the cache's tiles in: the positions go 32 at a time, each tile folded into a running max and sum, and every 256 positions that running state is merged into the token's, in order. A prompt's tokens then read each tile of the cache once for 64 of them, where before they read the whole cache once for every 8; a step computes its pieces of 256 at once and merges them. Both give a token the same bits, so a prompt's logits are still a token at a time's. Up to 1,024 positions nothing changed, so no figure published here moved.

### Faster prompts in less precision

`--precision fast`, with `--device cuda`, reads a mixture of experts' prompt faster by giving up its bits. A chunk of 64 tokens or more takes its projections and its experts through the GPU's tensor cores, and its attention too when the chunk is one sequence's run of positions, as a prompt's is: the inputs are rounded to float16 and the products summed in float32 by the hardware, in its own order. The weights widen as everywhere else, then round to float16 for the product. Decoding a token at a time stays exact, so what changes is the prompt's cache and logits, and so the tokens that follow from them. It needs a GPU with tensor cores (compute capability 7.0 or later). The default, `--precision exact`, is unchanged: float32, bit for bit, and every gate is checked in it. The owner chose this as an opt-in mode on 2026-10-06 ([`docs/PARITY.md`](docs/PARITY.md)), after gate A10 measured llama.cpp reading prompts faster in less precision.

```bash
./engine/vitna-anchor generate --model models/qwen3-30b-a3b --weights models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf --prompt-file long.txt --ctx 32784 --precision fast --device cuda
```

It is measured against gate A9's reference as quantization was, not held to the reference's tolerance ([below](#speed-with---precision-fast)): from Qwen's Q4_K_M file, it moves less from the reference recorded on those weights than quantizing BF16 to Q8_0 moves from the BF16 one.

## Serve the model

```bash
./engine/vitna-anchor serve --model models/smollm2-135m
curl http://127.0.0.1:8765/v1/chat/completions -H "content-type: application/json" \
  -d '{"model": "smollm2-135m", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

Any OpenAI client can use `http://127.0.0.1:8765/v1` as its base URL, and any Anthropic client `http://127.0.0.1:8765`. The model is served under its directory's name unless `--model-id` gives another, and a request naming a different model gets a 404.

A browser can use `http://127.0.0.1:8765/`, where the engine serves a chat page built into it (`engine/web/chat.html`, written into the engine by `engine/tools/embed_page.mjs`). It streams the reply from `/v1/chat/completions`, shows Qwen3's reasoning folded away above the answer, renders the Markdown a model writes (code, lists, tables, links) from escaped text so that nothing it writes becomes markup, stops a reply on Esc, and says under each reply how many tokens it took, how fast they came and how much of the prompt was reused. With Qwen3 it can switch thinking off, and it samples as Qwen3's authors recommend for either. The page loads nothing from anywhere and may talk to its own server alone, which its content security policy enforces.

- **Endpoints.** `GET /v1/models`, `GET /v1/health`, `POST /v1/chat/completions` and `POST /v1/completions`, OpenAI's `POST /v1/responses` and `POST /v1/responses/input_tokens`, and Anthropic's `POST /v1/messages` and `POST /v1/messages/count_tokens`. Each that generates is streamed as server-sent events with `"stream": true`, in its own API's events: chat chunks ending in `[DONE]`, Anthropic's content blocks from `message_start` to `message_stop`, and Responses' numbered events from `response.created` to `response.completed`. The two counters return the tokens a request's prompt would hold, its template included, and run nothing. A completion prompt may be a string or an array of token ids. A HEAD request is answered as GET would be, without the body.
- **Usage.** `prompt_tokens` counts the tokens the model read, after the chat format is applied. `completion_tokens` counts the tokens it generated, including the end-of-text or other special token it stopped on. `prompt_tokens_details.cached_tokens` counts the prompt tokens whose keys and values were reused from the request before, rather than computed again. A streamed response carries usage on its final chunk, and also sends OpenAI's separate usage chunk when `stream_options.include_usage` is true. Anthropic's usage counts those reused tokens apart, as `cache_read_input_tokens` beside `input_tokens`, as its API counts cache reads; Responses' counts `reasoning_tokens`, the tokens up to the one that closed the reasoning.
- **Chat format.** A conversation from any of the three APIs is converted into one, in OpenAI's chat shape, and written in the model's chat template. For Qwen3-30B-A3B that is Qwen3's own template, which the engine writes in C byte for byte as transformers renders the model's `tokenizer_config.json` (`tests/chat-template.test.mjs` checks 13 conversations in CI): the tools in its system message, each call a `<tool_call>` block, each result a `<tool_response>`, and the reasoning between `<think>` and `</think>`. For any other model it is ChatML (`<|im_start|>role`), the format the SmolLM2 family's instruct models use, which has no tools, so a conversation with any is refused; the pinned model is the base model, which was not trained on it, so its chat replies are poor. Text in several parts or blocks is joined with a newline, as vLLM and llama.cpp join it. Generation stops at any special token.
- **Reasoning and tool calls.** With Qwen3's template a reply is read back as it streams. Its reasoning is chat's `reasoning_content`, Anthropic's `thinking` block, and Responses' `reasoning` item with the text as `reasoning_text`; each `<tool_call>` is a call, its arguments written as the template writes them; the rest is the reply's text. A stream holds back only what could still begin a tag. After a call, chat's `finish_reason` is `tool_calls` and Anthropic's `stop_reason` `tool_use`. A call's arguments sent back as a string of JSON are read as the object it holds, as vLLM does, so the prompt holds them as the model was trained to read them. `tool_choice` `none` keeps the tools in the prompt, so it starts the same whatever the choice, and bans the `<tool_call>` token. `required` (Anthropic's `any`) and a named tool write the call's opening into the prompt for the model to finish, without thinking first, as Anthropic's API requires. Thinking is on unless asked off in OpenAI's chat and Responses, as Qwen3's template has it, and off unless asked for in Anthropic's Messages, as Claude's API has it: `chat_template_kwargs.enable_thinking`, `reasoning_effort` and Responses' `reasoning.effort` (`none` or `minimal` turn it off) and Anthropic's `thinking` switch it.
- **Anthropic's Messages.** `system`; `messages` with text, `tool_use`, `tool_result`, `thinking` and `redacted_thinking` blocks, consecutive messages of one role taken as one turn as Anthropic's API takes them, a `system` message among them (Claude Code sends them), and a last message from the assistant continued as a prefill; `tools` the client runs; `tool_choice`, `thinking`, `max_tokens` (required), `temperature` from 0 to 1, `top_p`, `top_k`, `stop_sequences` and `stream`. As Anthropic's API does since Claude Sonnet 4.5, a `max_tokens` larger than the context has room for runs until the context is full and then stops with `stop_reason` `model_context_window_exceeded`, and a prompt the context cannot hold is refused in that API's words, `prompt is too long`. Errors come in Anthropic's shape on its routes. Images, documents and the tools Anthropic runs on its own side, such as web search, are refused by name. A thinking block comes back with its text whatever its `display` asks, so that a client sending it back has the template write the reasoning the model wrote.
- **OpenAI's Responses.** `instructions`; `input` as a string or items: messages, `function_call`, `function_call_output`, and `reasoning`, whose content (or else summary) goes back to the turn it came with; `tools`, functions and a namespace's functions, which are offered by their own names and come back called with their namespace beside them; `tool_choice`, `reasoning`, `text.format` (`json_object` is JSON mode), `max_output_tokens`, `temperature`, `top_p` and `stream`. A tool OpenAI runs on its own side, such as `web_search`, is left out of what the model is offered and named in `x-vitna-ignored`. The server keeps no responses, so `previous_response_id` and `conversation` are refused, as are `background` and a `truncation` other than `disabled`; `store` and `include` change nothing here.
- **Parameters.** `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `top_k`, `seed`, `stop` (up to four), `stream`, `stream_options`, `response_format`, and for a conversation `tools` and `tool_choice`. The seed used is returned in an `x-vitna-seed` header, so a sampled reply can be reproduced. Anything that would change the output and is not implemented is refused with a 400 naming it: `n` above 1, a JSON Schema (`response_format` of type `json_schema`), log probabilities, penalties and logit bias, and tools for a model whose template has none. A field the server does not know is ignored and named in an `x-vitna-ignored` header.
- **JSON mode.** With `"response_format": {"type": "json_object"}`, before each token is chosen the logit of every token that could not continue a JSON object is set to minus infinity: special tokens, and any token whose bytes would break the object. Strings must be valid UTF-8, and a run of whitespace is capped at 16 characters so the model cannot pad forever. Generation stops when the object closes. A reply cut short, by `max_tokens` (`finish_reason` `length`) or by a stop sequence, is the valid start of an object. The model is not told to write JSON, so the prompt should ask for it.
- **Requests at once.** A server started with `--parallel <n>` runs up to n requests at once, each in a sequence of the key-value cache of its own, so its cache takes n times the memory; more wait their turn, in the order they came. Each round it runs every running request's next token together: on a GPU in passes of up to 8 tokens that read each weight once, on the CPU a step at a time, which is no faster than one request at a time. Every token's logits are the ones a step of that request alone would give, bit for bit, so running together changes no response. With `--no-prefix-cache`, a response is the one a server running one request at a time gives: text, finish reason, usage, stream events, a seeded sample, a failed step. `tests/parallel-serve.test.mjs` checks this. With reuse on, a request may find a longer or shorter prefix than it would one at a time, which changes `cached_tokens` and, on a GPU, can move logits by float32 rounding, as reuse itself can (see Prefix reuse). Requests that sample, or mask for JSON mode, then take their tokens on up to 8 threads at once, each with a sampler of its own; each draws from its own seed's stream, so which thread takes a token changes nothing. A new request's prompt runs whole when nothing else is decoding. While others are, it runs in pieces between their rounds, 512 tokens at a time on a GPU and 8 on the CPU, so they wait for a piece rather than the whole prompt. A prompt split into pieces of more than 32 tokens runs exactly as it does whole: its keys, values and logits are the same, bit for bit, on either device. Each connection is served on a thread of its own, up to 64 at once, so a client that stalls or reads slowly holds up no other.
- **Prefix reuse.** Each sequence keeps the key-value cache of the last request it served. A request goes to the free sequence holding the longest prefix of its prompt, reuses that prefix, and recomputes at least the last prompt token. A conversation sent back with one more turn reuses its history, and with `--parallel` above 1 two clients taking turns keep their own. Reuse only skips work. On the CPU, a reply, and its usage apart from `cached_tokens`, is exactly what a server started with `--no-prefix-cache`, which turns reuse off, returns for the same request. On the GPU the tokens of a prompt run together, while a reused prefix may hold keys and values computed a token at a time, so the two servers' logits can differ by float32 rounding, and a reply can differ only where two tokens' logits are that close.
- **Speculation.** A server started with `--speculate <k>` drafts up to k tokens after each one it takes, where the text's last two or three tokens occur earlier in it, and checks them together; on a GPU in one pass, on the CPU a step at a time. With several requests running, drafts take only the rows a round's passes have room for beyond one a request. Each token is still taken from logits a step would give, bit for bit, through the JSON mask, the stop sequences and the sampler in turn, and the cache is left holding what a token at a time would leave, so no response changes: text, finish reason, usage, stream events, a failed step. `tests/speculate-serve.test.mjs` checks this against a server without it. The server prints one line a request to stderr saying how many drafts it took.
- **A step that fails.** If the model fails to run a step, the request ends with an error, never with the reply so far. A response not yet begun is a 500 of type `server_error`. A stream has already sent its 200, so it ends with an event carrying the same error, in place of the final chunk, with no usage and no `[DONE]` after it. The next request reuses only the tokens whose steps ran. On the GPU a step fails when the device reports an error, which the server prints to stderr. The tests make a step fail with a hook instead (see [Tests](#tests)).
- **A GPU that can run nothing more.** Some errors leave the device unable to run anything more in the process: CUDA calls them sticky (an illegal memory access, a kernel that faulted or ran too long, and others), and only a new process can use the device again. After a step fails on the GPU, the server asks the device for further work, a wait on its stream; if that fails too, the device is lost. Every request running then ends with the error above, and every request waiting for its turn, or sent in the quarter of a second before the server notices, gets a 503 with code `device_lost` that names the CUDA error. The server stops listening, lets the connections still open finish, and exits with status 75 (`EX_TEMPFAIL` in `sysexits.h`), so that whatever started it (systemd, Docker, Kubernetes, a shell loop) can start it again. An error that is not sticky leaves the server serving.
- **Limits.** One request at a time unless `--parallel` says more, and each running request holds a sequence of the cache from its first token to its last. The context is the model's maximum, at most 4096 tokens, or `--ctx`, for each sequence. A request that would not fit is refused, not cut short, but for Anthropic's `max_tokens`, above. There is no authentication, so the server listens on 127.0.0.1 unless `--host` says otherwise, and warns if it does.

### Claude Code and Codex

Claude Code speaks Anthropic's Messages and Codex OpenAI's Responses, so both can work against the server. Their prompts are long, about 14,000 tokens for Claude Code's and 9,000 for Codex's on Qwen3-30B-A3B, so give the server a context of 32K; each turn after the first reuses nearly all of the prompt before it.

```bash
./engine/vitna-anchor serve --model models/qwen3-30b-a3b --weights models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf --model-id qwen3-30b-a3b --device cuda --ctx 32768 --precision fast
```

Claude Code, with the server's model named for every tier it asks for, and a placeholder key, since the server has no authentication:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 ANTHROPIC_API_KEY=placeholder ANTHROPIC_MODEL=qwen3-30b-a3b \
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3-30b-a3b ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3-30b-a3b ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3-30b-a3b claude
```

Codex, with a provider in its `config.toml`, and its key from `VITNA_KEY=placeholder` in the environment:

```toml
model = "qwen3-30b-a3b"
model_provider = "vitna"

[model_providers.vitna]
name = "vitna-anchor"
base_url = "http://127.0.0.1:8765/v1"
env_key = "VITNA_KEY"
wire_api = "responses"
```

How a session with each went is [below](#how-gate-a11-was-checked).

## Serve an embedding model

```bash
node scripts/fetch-model.mjs bge-small-en-v1.5
./engine/vitna-anchor serve --model models/bge-small-en-v1.5 --threads 4
curl http://127.0.0.1:8765/v1/embeddings -H "content-type: application/json" \
  -d '{"model": "bge-small-en-v1.5", "input": ["Represent this sentence for searching relevant passages: how do tides work", "Tides are the regular rise and fall of the sea."]}'
```

A model whose `config.json` says `bert`, with sentence-transformers' `modules.json` beside it, is served as an embedding model: `POST /v1/embeddings` in place of generation. A model that generates answers `/v1/embeddings` with a 404 that says so, and an embedding model answers generation the same way.

- **Request.** `input` is a string or a list of up to 2,048. `encoding_format` is `float`, the default, or `base64`, float32 little-endian. `dimensions`, if given, must be the model's own, since a model not trained to be cut shorter cannot be. Token ids as input are refused, since the model's own `[CLS]` and `[SEP]` would have to be among them. Other fields are ignored and named in an `x-vitna-ignored` header.
- **What a text becomes.** The model's own tokens, `[CLS]` and `[SEP]` included, then the pooling its `1_Pooling/config.json` names and the L2 normalization its `modules.json` asks for: for bge-small-en-v1.5, the `[CLS]` position, normalized, 384 values. That model's card asks for a query to start with `Represent this sentence for searching relevant passages: ` and for documents to go as they are. The server adds nothing, so a client sends the instruction itself.
- **Usage.** `prompt_tokens` and `total_tokens` both count the tokens the model read, `[CLS]` and `[SEP]` included.
- **Limits.** A text longer than the model reads, 512 tokens for this one, is refused with a 400 that names which, never cut short, and one request may hold 131,072 tokens in all. Requests run one at a time, each on every thread `--threads` gives, by default every processor. It runs on the CPU only.
- **The same bits.** Every output of the model's matrix products is one dot product summed the same way wherever it falls, so a text's embedding is the same, bit for bit, on any number of threads and whatever it is sent with.

`embed --model <dir>` prints, for each JSON string on a line of stdin, its token ids, the pooled vector's norm and its embedding; `--timing` says how long it took.

Measured with `embed --timing` on 256 passages of 225 tokens on average, 57,508 tokens in all, on an otherwise idle OVHcloud b3-16 instance, 4 vCPUs of an AMD EPYC (Milan), Linux, the engine built with `zig cc -target x86_64-linux-gnu -O3`: 1,086 tokens a second on 1 thread, 2,032 on 2, 2,795 on 3 and 3,673 on 4. On this machine, an AMD Ryzen 7 3700X (8 cores, 16 threads, AVX2) on Windows, 1 thread gives 1,232. Through the server on 4 threads, a 15-token query takes 10 to 18 ms from request to response, after the first.

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

`tests/chat-template.test.mjs` holds the engine's Qwen3 chat template to transformers' rendering, byte for byte, through the engine's `chat-prompt` command, on 13 conversations recorded by `reference/record_chat_template.py`; it needs only the engine. `tests/api-surface.test.mjs` serves the pinned model through all three conversation APIs with two hooks for tests, which the server names on stderr: `VITNA_TEST_TEMPLATE=qwen3` writes conversations in Qwen3's template whatever the model, and `VITNA_TEST_REPLY=1` takes a request's own `vitna_test_reply` as its reply, a token at a time in place of the model's choices, and names an FNV-1a hash of each prompt in an `x-vitna-test-prompt` header. So a model too small to call a tool still drives calls, reasoning, stop sequences and the token limit through each API's responses and streams, and each prompt is compared with the conversation converted by hand and rendered by `chat-prompt`. It also puts every request Claude Code and Codex sent, as recorded in `reference/api-shapes/`, through the APIs' token counters, which convert it and write it in the template without running the model; with `VITNA_REPLAY_SHAPES=1` it answers each of them through the model as well, which takes about six minutes on one CPU thread, so CI does not. CI requires both files.

`tests/reference-embed.test.mjs` holds the engine to gate A14's fixture and `tests/embeddings-serve.test.mjs` its server; both need the embedding model's files (`node scripts/fetch-model.mjs bge-small-en-v1.5`, 133 MB), which CI fetches, and CI requires both. `engine/tools/check_wordpiece.py` compares the engine's BERT tokenizer with the tokenizers library on every code point and on random strings; the Reference workflow runs it on Linux.

With `VITNA_DEVICE=cuda`, the last four run the model on the GPU, in an engine built with the CUDA path; nothing else about them changes, the fixture and the tolerance included. `tests/device.test.mjs` checks that an engine refuses a device it cannot use. CI's CUDA job runs it against an engine built without the CUDA path and one built with it, on a runner with no GPU.

`tests/reference-qwen3.test.mjs` and `tests/reference-qwen3-gguf.test.mjs` do the same for Qwen3-30B-A3B and its two GGUF files: the fixtures are checked with nothing but the repository, CI fetches the tokenizer alone and requires it to match (`VITNA_REQUIRE_QWEN3_TOKENIZER=1`), and the model's comparisons, which need its 61.1 GB or the GGUF files, run by hand with `VITNA_REQUIRE_QWEN3=1` or `VITNA_REQUIRE_QWEN3_GGUF=1` ([below](#how-gate-a9-was-checked)).

`tests/long-context.test.mjs` checks gate A10 on the GPU from the Q4_K_M file: the logits with every layer's key-value cache on the GPU, with none there and with half, and with every token run alone, must be the same byte for byte, at 300 positions and at 2,000, and a prompt of 32,768 tokens must run where the GPU has no room for its whole cache and give the logits that none of it there gives. It runs by hand, with `VITNA_REQUIRE_QWEN3_GGUF=1` and `VITNA_DEVICE=cuda`, the 32,768-token prompt only with `VITNA_REQUIRE_LONG=1` as well ([below](#how-gate-a10-was-checked)). `VITNA_TEST_LONG_FROM=<n>`, a hook for tests that the engine names on stderr, gives every token of a mixture on the GPU that attends to more than n positions a long context's attention, so that `tests/reference-qwen3-gguf.test.mjs` can check that attention against A9's reference on its short prompts. The same file measures `--precision fast` against the reference as quantization was, with `VITNA_TEST_FAST_ROWS=2`, another hook for tests, which gives the fixture's short prompts the fast path; it asserts only bounds far looser than the figures it reports, which a broken path fails. `tests/long-context.test.mjs` requires the fast mode's logits to be the same, byte for byte, wherever the cache is.

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

Checked again on 2026-10-05, on the same machine, after gate A8 changed how the GPU computes the experts' activation. CUDA's `expf` ends in a hardware approximation no CPU reproduces, so the experts' SiLU on the GPU now takes `exp` in operations both devices round alike (`engine/src/exact.c`, within 3 ulp of the true value), which the CPU can then repeat bit for bit (see [below](#speed-with-the-cpu-beside-the-gpu)). The dense path and the CPU path keep the library's `expf`. `tests/reference-moe.test.mjs` passed 17 of 17 on the GPU, its seventeenth A8's. With the experts pinned, the largest logit difference was 6.29e-5, and 3.67e-5 for logsumexp; the router logits differed by at most 3.15e-5, and the router chose the reference's experts in all 3,872 decisions at the prompts' positions. Greedy decoding with the experts pinned gave 192 of 192 tokens equal, the top logits at each step within 2.29e-5, and unpinned the routing was the reference's throughout and it decoded the same 192 tokens.

## How gate A9 was checked

Gate A9's reference is recorded a layer at a time, since the model is twice this machine's memory in float32 ([reference/README.md](reference/README.md#qwen3-30b-a3b-gate-a9)): BF16, and Qwen's own Q8_0 and Q4_K_M files, each against the same 6 prompts, 239 prompt positions, 192 greedy tokens and 20,400 routing decisions. Checked on 2026-10-05 and 06 on the same machine as A4 and A5 (AMD Ryzen 7 3700X, 64 GB, RTX 3070 with 8 GB), the engine built with the CUDA path:

```powershell
$env:VITNA_REQUIRE_QWEN3 = "1"; $env:VITNA_REQUIRE_QWEN3_GGUF = "1"
node --test --test-reporter=spec tests/reference-qwen3.test.mjs tests/reference-qwen3-gguf.test.mjs
$env:VITNA_DEVICE = "cuda"   # and again on the GPU, a file at a time
```

Every comparison passed, on the CPU and on the GPU, from all three:

| Weights, device | Largest logit difference, experts pinned | Router logits | Greedy tokens, pinned and unpinned |
|---|---|---|---|
| BF16, CPU | 7.25e-5 | 9.30e-6 | 192 of 192, and 192 |
| BF16, GPU | 8.11e-5 | 2.72e-5 | 192 of 192, and 192 |
| Q8_0, CPU | 7.44e-5 | 5.26e-5 | 192 of 192, and 192 |
| Q8_0, GPU | 1.37e-4 | 7.64e-5 | 192 of 192, and 192 |
| Q4_K_M, CPU | 5.34e-5 | 2.77e-5 | 192 of 192, and 192 |
| Q4_K_M, GPU | 7.34e-5 | 3.55e-5 | 192 of 192, and 192 |

The tolerances are 1e-2 for logits and 1e-3 for router logits, A1's and A5's. Everywhere the engine's router chose the reference's experts in all 11,472 decisions at the prompts' positions, and unpinned its routing was the reference's throughout, so it decoded the same 192 tokens. The engine's tokenizer gave tokenizer.json's ids for all 86 strings. From each GGUF file, every one of the 579 tensors widened to gguf-py's bits, on the CPU and on the GPU, by digest. On the GPU the BF16 experts came through a cache of 16 GiB in memory, read from the drive.

Gate A8 on this model: through 1 GiB of the GPU with `--cpu-experts 4`, from both GGUF files, every position's logits and 16 greedy tokens with theirs were the GPU alone's, byte for byte; and `expert-check` ran all 6,144 experts on both devices, with the same activations and outputs byte for byte, in BF16, Q8_0 and Q4_K_M.

## How gate A10 was checked

Gate A10 asks three things of Qwen3-30B-A3B: that a prompt of 32K tokens runs with the part of the key-value cache the GPU has no room for in page-locked host memory; that at a length where both fit, the logits are the same, byte for byte, as with the cache on the GPU; and that reading a prompt is measured at 4K and 32K tokens. Checked on 2026-10-06 on the same machine as A9 (AMD Ryzen 7 3700X, 64 GB, RTX 3070 with 8 GB), from Qwen's Q4_K_M file:

```powershell
$env:VITNA_DEVICE = "cuda"; $env:VITNA_REQUIRE_QWEN3_GGUF = "1"; $env:VITNA_REQUIRE_LONG = "1"
node --test --test-reporter=spec tests/long-context.test.mjs
node --test --test-reporter=spec --test-name-pattern "long context" tests/reference-qwen3-gguf.test.mjs
```

All of it passed:

- 300 positions: the logits with no layer's cache on the GPU, and with 24 of the 48 layers' there, were those with all of it there, byte for byte, both in the attention a short context takes and in a long context's.
- 2,000 positions, the last 976 in a long context's attention: 8 greedy tokens and their logits were the same, byte for byte, with no layer's cache on the GPU, with 24, and with every token run alone.
- 32,768 positions, at `--ctx 32784`: the GPU had room for the cache of 37 of the 48 layers, and the other 11's, 1.4 GiB, were in page-locked host memory. 16 greedy tokens and their logits were those with none of it on the GPU, byte for byte.
- A long context's attention against gate A9's reference: `VITNA_TEST_LONG_FROM=0`, a hook for tests, gives every token that attention, so the fixture's short prompts check it. With the routing pinned, the logits were within 7.34e-5 of the reference's and the router logits within 4.79e-5, the router chose the reference's experts in all 11,472 decisions, and the 192 greedy tokens were the reference's. The prompts' tokens run together gave each token's logits alone, byte for byte, which compares the two kernels that compute that attention.
- Without the hook, A9's figures on the GPU were unchanged: logits 7.34e-5, router logits 3.55e-5.

Two things about the driver were found on the way, on this machine's Windows one. The device had 373 MiB less free after the weights were copied in whenever the plan left it about a GiB or less, and none less with 2 GiB left, so the plan now leaves 512 MiB more besides; that is a margin measured on one machine, not a model of the driver. And a request for an expert cache larger than the device's memory succeeded, which the driver's system memory fallback would explain; that was not checked further, but it is why the engine sizes everything by what the device reports free rather than by an allocation failing.

## How gate A11 was checked

Gate A11 asks for Anthropic's `/v1/messages` with its streamed events, tool calls in OpenAI's and Anthropic's shapes written in the model's own tool format, and OpenAI's Responses API, each tested against recorded shapes, and a session with Claude Code and one with Codex checked by hand. Checked on 2026-10-06 on the same machine as A9 and A10 (AMD Ryzen 7 3700X, 64 GB, RTX 3070 with 8 GB, driver 591.86, Windows 11, Node.js 24.19.0, Python 3.12.10), with Qwen3-30B-A3B from Qwen's Q4_K_M file, served with `--device cuda --ctx 32768 --precision fast`.

In CI, on every push:

- **The model's own tool format.** `tests/chat-template.test.mjs` holds the engine's Qwen3 template to transformers' rendering, byte for byte, on 13 conversations: tools with a system prompt and without, calls whose arguments are a string and an object, calls in parallel and over several steps, reasoning kept and dropped as the template keeps and drops it, thinking switched off, numbers as Python writes them, and empty content. `reference/record_chat_template.py` recorded them with transformers 5.17.0 and jinja2 3.1.6, which it pins, as it pins the model's `tokenizer_config.json` by hash.
- **The three APIs.** `tests/api-surface.test.mjs`: one conversation sent through all three is written as the same prompt, which is the conversation converted by hand and rendered by `chat-prompt`, byte for byte; each API's whole response and stream carry the reply's reasoning, text and calls in its own shape, and the stream's events rebuild the whole response; forced calls, `tool_choice` none, stop sequences, the token limit, a prefill, Anthropic's context window, Codex's namespaces and hosted tools, HEAD, and errors each in its API's shape; and a ChatML model answers one conversation the same through all three. The unit tests read 12 replies back, whole and a byte at a time, and convert requests of each shape worked by hand.
- **Recorded shapes.** `reference/api-shapes/` holds every request Claude Code 2.1.283 and Codex 0.160.0 sent in the sessions below, recorded between them and the server by `scripts/record-api-requests.mjs`, which writes no key, and made into fixtures by `scripts/api-shapes.mjs`, which keeps their structure, replaces every string longer than 32 characters by its length so that neither vendor's prompts are copied, and drops the fields that identify a machine or a session. CI puts each through its API's token counter, which converts it and writes it in the template. With `VITNA_REPLAY_SHAPES=1` each is answered through the model as well, with a call to one of its own tools in its API's stream: all of them passed here, in 380 s.

By hand:

- **The responses, against the official SDKs.** No response was recorded from Anthropic's or OpenAI's own APIs, which would need their keys. The responses are held instead to Anthropic's Python SDK 0.120.2 and OpenAI's 2.53.0, whose types parse every response and every stream event: `scripts/check-api-clients.py` asks each of Messages, chat completions and Responses a question that needs a tool, sends back the result of the call the SDK parsed, and checks the answer uses it, whole and streamed through each SDK's own accumulator, with Anthropic's thinking and token counter besides. All 15 checks passed, in 95 s.
- **Claude Code 2.1.283**, in print mode, with a configuration directory of its own, a placeholder key and only tools that read: asked what a small `hello.py` prints, it read the file and answered correctly, in 2 turns and 84 s. With Edit allowed, in a scratch directory, it added the docstring it was asked for, in 3 turns and 109 s; its closing summary misquoted the function's return line, which the edit had left as it was. Its first prompt was 14,149 tokens, its tools included, and each later turn reused all but 44 to 95 of its prompt's tokens from the cache.
- **Codex 0.160.0**, through `codex exec`, with a home of its own, the Responses wire API and a placeholder key: asked to set a goal with its `create_goal` tool, it called the tool, Codex ran it, and the model reported what it returned, in 2 turns and 60 s. Asked what `hello.py` prints, the model called Codex's `exec_command` with `Get-Content hello.py`, and Codex's own sandbox refused to run it, under both `read-only` and `workspace-write`, as Codex does on Windows with no sandbox set up for a fresh home; the model then said so. Codex's own count of that turn read the server's usage: 18,056 input tokens, 9,189 of them cached, and 725 of reasoning.

Running the two clients is how the requests above came to be taken as they are. Claude Code's first request is `HEAD /api/hello`, which the server answered with a body, which Node's HTTP parser refuses. It sends system messages among its messages (Anthropic's `mid-conversation-system` beta), `max_tokens` 32000 whatever its prompt, adaptive thinking with its display omitted, and `context_management` and `output_config`, which change nothing here and are named in `x-vitna-ignored`. Codex offers a namespace of functions, `multi_agent_v1`, and OpenAI's hosted `web_search`, and asks for its reasoning encrypted, which it sends back with the reasoning's text in it, so the template writes that turn's reasoning as the model wrote it.

## How gate A12 was checked

Gate A12 asks for one command on Windows and one on Linux that detects the GPU, memory and disk, picks a model and size that fit, fetches it with resume and SHA-256 checks, and starts the server, and for a chat page that talks to it, checked on a fresh Windows and a fresh Linux machine. GitHub's hosted runners are the fresh machines: each run is a new virtual machine, and none has a GPU. The GPU path was checked by hand, on the machine that checked A9 to A11 (AMD Ryzen 7 3700X, 64 GB, RTX 3070 with 8 GB, driver 591.86, Windows 11, Windows PowerShell 5.1). Both were checked on 2026-10-06, against release [v0.2.0](https://github.com/convexityos/vitna-anchor/releases/tag/v0.2.0), which the release workflow built from `main`: the engine with its CUDA path for Linux and Windows on x86-64, with code for Turing to Blackwell, and on the CPU for macOS on arm64.

On fresh machines, in CI, in the [run of `install.yml`](https://github.com/convexityos/vitna-anchor/actions/runs/37539807058) that the release workflow started once it had published v0.2.0:

- **The command.** On `windows-latest`, in Windows PowerShell 5.1, `irm https://github.com/convexityos/vitna-anchor/releases/download/v0.2.0/install.ps1 | iex`; on `ubuntu-latest`, `curl -fsSL https://github.com/convexityos/vitna-anchor/releases/download/v0.2.0/install.sh | sh`. Each fetched the release's engine and checked it against the SHA-256 the release wrote into the script.
- **The plan.** Each runner has 16 GiB of memory and no GPU, so the plan chose SmolLM2-360M-Instruct and said why: "Qwen3-30B-A3B at Q4_K_M needs an NVIDIA GPU with 7.5 GiB and 30 GiB of memory, and no NVIDIA GPU was found." Both engines are built with the CUDA path; finding no GPU, they ran on the CPU.
- **The files.** Into a folder named `vitna-anchor-éЖ`, the installer resumed the weights from byte 10,485,760, a part fetched before it ran, and the engine checked every file against its pinned size and SHA-256.
- **The server.** The `serve` script the installer wrote started a server that answered its health check, served the chat page, and answered "What is the capital of France?" with "The capital of France is Paris." Each job took 31 s on Windows and 25 s on Linux, from start to finish.

By hand, on this machine:

- **The command.** The one in [Install](#install), `irm https://github.com/convexityos/vitna-anchor/releases/latest/download/install.ps1 | iex`, in Windows PowerShell 5.1, with `VITNA_HOME` an empty folder on another drive and `VITNA_NO_START=1`, so that the server was started afterwards, by the `serve.cmd` it wrote.
- **The plan.** It chose Qwen3-30B-A3B at Q4_K_M with a context of 32K, on the GPU, and said why: "This machine has an NVIDIA GeForce RTX 3070 with 8.0 GiB, 64 GiB of memory and 356 GiB free where the models go."
- **The files.** It fetched 18.6 GB from Hugging Face at the pinned revisions, the GGUF in 4 min 32 s, at 64.8 MiB a second on average over this connection, and the engine then checked the three files in about 64 s: 340 s in all.
- **The server.** `serve.cmd` started the server, which answered its health check 7.7 s later, with the model on the GPU and the key-value cache of 37 of the 48 layers there, the others' in page-locked host memory. Asked a question with reasoning on, it answered in 550 tokens and 28.0 s.
- **The chat page.** Opened at the server's address in a browser, the page named the model and its context from the server, streamed a reply with its reasoning folded away, and ended it with its count: 266 tokens at 18.8 a second, a prompt of 18 tokens, 3 of them reused.

The chat page was checked further in [#50](https://github.com/convexityos/vitna-anchor/pull/50), in a browser against Qwen3-30B-A3B from Q4_K_M on the same GPU, served with `--precision fast` and a 16K context: a streamed reply of 620 tokens with a code block came back at 26.8 a second, a second turn reused the first's 33 prompt tokens, and Stop ended a reply at 147 tokens, the server noting that the client had gone. In CI, in Node, its Markdown rendering, its reading of a stream and its handling of each chunk are tested, and the page is served byte for byte under its policy.

What checking the gate turned up, before v0.2.0 was tagged:

- **A folder named with any letters, on Windows.** A user's name, and so the default folder, often has letters outside ASCII. With `é` the plan's paths came back garbled through PowerShell, and with `Ж` the engine could not open its own files. On Windows the engine is now a UTF-8 program, through a manifest; the installer reads its output as UTF-8; and `serve.cmd` starts from its own folder ([#51](https://github.com/convexityos/vitna-anchor/pull/51)).
- **The release's Windows engine.** `windows-latest` now carries Visual Studio 2026, for which CMake finds no CUDA 13.1 toolset, so the release builds that engine on `windows-2022` ([#52](https://github.com/convexityos/vitna-anchor/pull/52)).
- **Each release, installed.** A release published with a workflow's own token starts no other workflow, so `install.yml`'s trigger on a published release would never have fired; the release workflow starts it itself ([#52](https://github.com/convexityos/vitna-anchor/pull/52)).

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

## Speed with quantized weights

Measured on 2026-10-05 on the same machine and GPU, the engine built as for gate A4, with OLMoE's own GGUF files (`node scripts/fetch-model.mjs olmoe-1b-7b-gguf`) on the Crucial P5. The commands, the first with each `--weights` and without it, the second likewise:

```powershell
node scripts/bench-experts.mjs --device cuda --tokens 64 --runs 5 --gpu-caches default,2048 [--weights models/olmoe-1b-7b-gguf/<file>]
node scripts/bench-experts.mjs --tokens 64 --runs 3 --caches none [--weights models/olmoe-1b-7b-gguf/<file>]
```

The same 64 greedy tokens after "The capital of France is", the median of the 63 steps timed, the experts copied to the GPU from the mapped file and the GPU's cache empty at the start of every run. On the GPU:

| Weights | Cache on the GPU | ms a token | Tokens a second | On the GPU when wanted | MiB copied a position |
|---|---|---|---|---|---|
| BF16, 13.8 GB | 5,460 MiB, 455 of the 1,024 experts | 16.6 | 60.3 | 88% | 315 |
| Q8_0, 7.4 GB | 5,884 MiB, 923 experts | 6.0 | 166.9 | 95% | 70 |
| Q4_K_M, 4.2 GB | 3,984 MiB, all 1,024 | 4.7 | 213.6 | 95% | 40 |
| BF16 | 2,040 MiB, 170 experts | 38.8 | 25.8 | 68% | 885 |
| Q8_0 | 2,046 MiB, 321 experts | 14.3 | 70.0 | 81% | 270 |
| Q4_K_M | 2,046 MiB, 526 experts | 6.2 | 162.5 | 91% | 72 |

A quantized expert is smaller, so the GPU holds more of them and copies fewer bytes when it lacks one: at Q4_K_M every expert fits beside the rest of the model, and what is left is the arithmetic. With a cache of 2 GiB the gap is wider still, since the BF16 model spends most of a token on the bus.

On the CPU, one thread computing and every weight read through the operating system's file cache:

| Weights | ms a token | Tokens a second |
|---|---|---|
| BF16 | 187.5 | 5.33 |
| Q8_0 | 483.3 | 2.07 |
| Q4_K_M | 1,140.1 | 0.88 |

The CPU path is slower on quantized weights than on BF16: it widens each block to float32 with a scalar loop before the multiply-adds, and that loop, not reading the weights, is what a token costs. Vectorizing it is the next step on the CPU.

Gate A8 then widened the blocks eight weights at a time with AVX2, to the same bits (`engine/src/quant.c`). The same command later the same day, at commit `5b0ee04`:

| Weights | ms a token | Tokens a second |
|---|---|---|
| BF16 | 140.1 | 7.14 |
| Q8_0 | 384.8 | 2.60 |
| Q4_K_M | 326.9 | 3.06 |

Q4_K_M is 3.5 times as fast as before, and Q8_0 1.26 times. BF16's path did not change in between, so its move from 187.5 to 140.1 is the machine's: other programs kept the CPU busy to different degrees in the two runs, and a single-threaded figure here moves that much with them.

## Speed with the CPU beside the GPU

Measured on 2026-10-05 on the same machine and GPU, the engine built as for gate A4 at commit `5b0ee04`. The machine's 64 GB is a DDR4-3200 kit running at 2133 MT/s, two channels: about 34 GB/s in theory, for the CPU and the GPU's copies together. Other programs kept the CPU 8% to 34% busy. The command, with each `--weights` and without it:

```powershell
node scripts/bench-experts.mjs --device cuda --tokens 64 --runs 5 --gpu-caches default,2048,1024 --cpu-experts 0,4,8 [--weights models/olmoe-1b-7b-gguf/<file>]
```

The same 64 greedy tokens after "The capital of France is", the median of the 63 steps timed, five runs at each size of the GPU's cache, with the experts it lacks copied in (the GPU alone) and with `--cpu-experts` at 4 and at 8 threads. In brackets, the share of the experts a step wanted that the CPU ran; the logits are the same in every column, byte for byte.

| Weights | Cache on the GPU | GPU alone, ms a token | 4 CPU threads | 8 CPU threads |
|---|---|---|---|---|
| BF16 | 5,460 MiB, what the device allowed | 16.3 | 15.0 (2%) | 15.8 (2%) |
| BF16 | 2,040 MiB | 37.7 | 33.5 (10%) | 36.1 (9%) |
| BF16 | 1,020 MiB | 48.1 | 44.6 (12%) | 43.8 (12%) |
| Q8_0 | 5,884 MiB, what the device allowed | 5.9 | 5.8 (0%) | 6.3 (0%) |
| Q8_0 | 2,046 MiB | 14.2 | 12.9 (1%) | 12.5 (2%) |
| Q8_0 | 1,020 MiB | 21.2 | 21.0 (3%) | 21.1 (6%) |
| Q4_K_M | 3,984 MiB, every expert | 4.7 | 4.7 (0%) | 4.7 (0%) |
| Q4_K_M | 2,046 MiB | 6.0 | 6.2 (0%) | 5.9 (0%) |
| Q4_K_M | 1,023 MiB | 9.9 | 10.1 (0%) | 10.0 (0%) |

The CPU helps where copies are most of a token: BF16 with a cache of 1 to 2 GiB on the GPU decodes 4% to 11% faster, and Q8_0 at 2 GiB 9% to 12%. At Q4_K_M it does nothing, and the scheduler knows it: it gives the CPU an expert only to measure it again. On this machine an expert costs the CPU more than a copy does. The engine measured a GiB of BF16 experts at 65 to 96 ms on the CPU against 42 to 47 ms over the bus, and a GiB of Q4_K_M experts at 157 to 249 ms against 44 or 45, since the CPU widens each block before it multiplies. So the CPU can only take a share of a layer's misses while the bus carries the rest, and both draw on the same memory, of which the copies alone read 23 to 26 GB/s. Strata's premise, that a missing expert costs less as a read from memory than as a copy over the bus, needs memory that reads well beyond what the bus carries, and this machine's reads little more. On one whose memory does, the same scheduler would give the CPU more, since it follows the costs it measures. The rows a prompt runs together copy in what they lack, with or without the CPU.

## Speed beside llama.cpp

Gate A9 measures the engine beside llama.cpp on the same machine, the same GPU and the same files: Qwen's own Q8_0 and Q4_K_M of Qwen3-30B-A3B, 30.5 GB and 18.6 GB, on an RTX 3070 with 8 GB, of which the desktop held 775 MiB. Neither file fits, so both programs keep some experts in the computer's memory: llama.cpp keeps every expert of its first n layers there and computes them on the CPU (`-ncmoe n`), and the engine keeps as many experts on the GPU as fit, whichever are used most, and copies in what a layer lacks. Measured on 2026-10-06, the CPU 3% busy beforehand, with llama.cpp's own Windows CUDA 12.4 build of release b11433 (the CUDA 13.1 installed here has no cuBLAS to build it against) and the engine at this commit:

```powershell
llama-bench -m Qwen3-30B-A3B-<file>.gguf -ngl 99 -ncmoe <n> -p 512 -n 128 -r 3 -fa on -t 8
node scripts/bench-experts.mjs --device cuda --model models/qwen3-30b-a3b --weights <file> --tokens 129 --runs 3 --gpu-caches default --cpu-experts 0,4 [--prompt <text>]
node scripts/bench-prefill.mjs --device cuda --model models/qwen3-30b-a3b --weights <file> --prompt-tokens 512 --runs 3
```

llama.cpp, swept over `-ncmoe` until a setting no longer helped. At the lowest settings its prompt speed fell by half, which is what the Windows driver spilling the device's memory into the computer's when the device is full would cause; that was not checked:

| File | `-ncmoe` | Prompt, 512 tokens | Decoding, 128 tokens |
|---|---|---|---|
| Q4_K_M | 48 (every expert on the CPU) | 372 tokens a second | 19.9 |
| Q4_K_M | 33 | 505 | 27.8 |
| Q4_K_M | 30 | 224 | 29.2 |
| Q4_K_M | 22 | 242 | 24.0 |
| Q8_0 | 48 | 234 | 12.1 |
| Q8_0 | 40 | 273 | 14.0 |
| Q8_0 | 32 | 142 | 15.8 |

Its best on each, set against the engine with the GPU's cache as large as the device allowed (1,862 of the 6,144 experts from Q4_K_M, 1,001 from Q8_0), decoding 128 tokens after three of the reference's prompts (`capital`, `code` and `long`):

| File | | llama.cpp's best | The engine | With `--cpu-experts 4` |
|---|---|---|---|---|
| Q4_K_M | Prompt, 512 tokens | 505 tokens a second | 345 | |
| Q4_K_M | Decoding after "The capital of France is" | 29.2 | 76.7 | 73.6 |
| Q4_K_M | Decoding after the Fibonacci code | 29.2 | 47.2 | 46.3 |
| Q4_K_M | Decoding after the lighthouse story | 29.2 | 81.7 | 77.6 |
| Q8_0 | Prompt, 512 tokens | 273 | 282 | |
| Q8_0 | Decoding after "The capital of France is" | 15.8 | 34.3 | 35.6 |
| Q8_0 | Decoding after the Fibonacci code | 15.8 | 26.3 | 27.3 |
| Q8_0 | Decoding after the lighthouse story | 15.8 | 40.3 | 41.8 |

The engine decodes 1.6 to 2.8 times as fast as llama.cpp's best here, and reads a prompt 0.68 times as fast from Q4_K_M and about as fast from Q8_0. Its decoding depends on the text, which llama.cpp's does not: a token's experts that the GPU holds already cost nothing to fetch, and the share it held ran from 76% (the code, Q8_0) to 98% (the story); llama.cpp computes the same layers on the CPU whatever the token. llama-bench times generation with no prompt, so its figure is one number for all three. A prompt runs as rows on the GPU, which copies in every expert a chunk wants, most of a layer's for 512 tokens, and that copying is where the engine's prompt time goes. The engine's figures are its timing lines; llama-bench's are its own, the mean of 3 runs.

## Speed with a long context

Gate A10 measures reading a prompt at 4K and 32K tokens, on the same machine and GPU as A9, from Qwen's Q4_K_M file, beside llama.cpp release b11433 at its best `-ncmoe` for each. The prompts are the first n tokens of `README.md`, then `reference/README.md`, then `engine/src/model_cuda.cu`, as they were at commit `86251de3c5b7288e6cfeb9aa35a7c9b9e29fdbc7`. Measured on 2026-10-06:

```powershell
node scripts/bench-prefill.mjs --device cuda --model models/qwen3-30b-a3b --weights <Q4_K_M> --text README.md --text reference/README.md --text engine/src/model_cuda.cu --prompt-tokens 4096 --ctx 4112 --runs 3
node scripts/bench-prefill.mjs --device cuda --model models/qwen3-30b-a3b --weights <Q4_K_M> --text README.md --text reference/README.md --text engine/src/model_cuda.cu --prompt-tokens 32768 --ctx 32784 --runs 1
llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -fa on -t 8 -p 4096 -n 0 -r 2 -ncmoe 30,33,36,40,48
llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -fa on -t 8 -p 32768 -n 0 -r 1 -ncmoe 36,40,44,48
llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -fa on -t 8 -p 0 -n 32 -d 32768 -r 1 -ncmoe 33,40
```

| | llama.cpp's best | The engine before this gate | The engine |
|---|---|---|---|
| Reading a prompt of 4,096 tokens | 573 tokens a second (`-ncmoe 33`) | 226 | 353 |
| Reading a prompt of 32,768 tokens | 469 (`-ncmoe 40`) | not run | 182 |
| Decoding after 32,768 positions | 15.9 (`-ncmoe 40`) | | 8.7 |

"Before this gate" is every token in the attention the engine had before, the same build with long contexts starting past 4,096 positions. At 32K it was not run: from the 512, 2,048 and 4,000-token prompts it would have taken about nine minutes, against three now. The engine's decoding figure is 15 tokens after 32,768 positions in the test above, at 115 ms each, with 11 layers' cache in host memory; with none of it on the GPU it was 293 ms. llama.cpp's sweep: 220, 573, 549, 516 and 443 tokens a second at 4,096 for `-ncmoe` 30, 33, 36, 40 and 48; 144, 469, 443 and 421 at 32,768 for 36, 40, 44 and 48; decoding at 32,768 positions, 8.0 and 15.9 for 33 and 40.

llama.cpp reads a prompt 1.6 times as fast at 4K and 2.6 times as fast at 32K, and decodes 1.8 times as fast at 32K. Where the engine's time goes: of an 8,000-token prompt's 24.8 s, a long context's attention took 5.5 s (timed with CUDA events in a build made for that, not committed); the rest, about 2.4 s for every 1,024 tokens whatever their positions, is the experts and the other matrix products, in float32, which also bounds the engine near 400 tokens a second at any length. That attention runs at about a fifth of the GPU's float32 rate; it grows with the square of the length, which would make it about 88 s of the 180 at 32K, though that was not timed. llama.cpp computes in less precision: its key-value cache is float16 by default, half the engine's float32, which is also why its cache fits on this GPU at 32K and the engine's does not. The engine stays in float32 so that it can be held to a float32 reference bit for bit; going faster than this means changing that, which is a decision for this project rather than for this gate.

## Speed with --precision fast

`--precision fast` measured the way gate A10 measured the engine ([above](#speed-with-a-long-context)): Qwen's Q4_K_M file on the RTX 3070, the prompts the first n tokens of `README.md`, then `reference/README.md`, then `engine/src/model_cuda.cu`, as they were at commit `5307ea7806860bc77ad27bb00b7bfdd9bed9550e`, beside llama.cpp b11433 at its best `-ncmoe` from that section. Measured on 2026-10-06:

```powershell
node scripts/bench-prefill.mjs --device cuda --model models/qwen3-30b-a3b --weights <Q4_K_M> --text README.md --text reference/README.md --text engine/src/model_cuda.cu --prompt-tokens 4096 --ctx 4112 --runs 3 --precision <exact|fast>
node scripts/bench-prefill.mjs --device cuda --model models/qwen3-30b-a3b --weights <Q4_K_M> --text README.md --text reference/README.md --text engine/src/model_cuda.cu --prompt-tokens 32768 --ctx 32784 --runs 1 --precision <exact|fast>
```

| Reading a prompt | llama.cpp's best | `--precision exact` | `--precision fast` |
|---|---|---|---|
| 4,096 tokens | 573 tokens a second | 349 | 1,014 |
| 32,768 tokens | 469 | 180 | 521 |

Decoding a token at a time runs the same in either mode. Against gate A9's reference recorded on the same Q4_K_M weights, measured as quantization was (`tests/reference-qwen3-gguf.test.mjs`, with `VITNA_TEST_FAST_ROWS=2`, a hook for tests, so that the fixture's short prompts take the fast path):

| | KL at the last position, mean of 6 | Top token agrees | Greedy tokens equal before the first difference |
|---|---|---|---|
| `--precision exact` | 1.9e-12 | 239 of 239 | 192 of 192 |
| `--precision fast` | 1.03e-3 | 234 of 239 | 160 of 192: one prompt departs at its first token, five never |
| For scale: Q8_0 against BF16 | 1.4e-3 | 232 of 239 | 136 of 192 |

In the fast mode the engine reads a prompt about three times as fast as in the exact one, at 4K and at 32K, and 1.8 and 1.1 times as fast as llama.cpp's best. Timed with CUDA events in a build made for that and not committed, a 32K prompt in the fast mode spent 34 of its 63 s in the attention phase (the projections, the staging of the 12 layers' cache in host memory and the attention itself) and 13 s in the experts, while 491 GiB of experts crossed the bus: with 6 GiB of the 8 GB device holding the cache, there is room for few experts, so every pass of 1,024 tokens copies nearly all of them again. Larger passes would copy less a token, at the cost of more of the device's memory.

## Measurements

A performance figure appears in this repository only with the hardware and the command that produced it, and it says whether it was measured or estimated. The first published are gate A5's, [above](#speed-with-the-experts-read-from-a-drive), on the CPU and [on a GPU](#speed-on-a-gpu), and the embedding model's, [above](#serve-an-embedding-model). `scripts/bench-decode.mjs` times decoding with `generate --timing` and prints the machine and the command with every result, so that a figure, when there is one, carries both.

## License

Apache License 2.0. See [LICENSE](LICENSE).
