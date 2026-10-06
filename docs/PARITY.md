# Parity with Strata

Written 2026-10-05. This is a plan, so nothing in it is a claim about what Anchor does: the README's gate table says that, and a gate below moves there only when its test passes.

## What Strata is

[Strata](https://github.com/Niko1221/Strata) (retrieved 2026-10-05, MIT, created 2026-09-24) runs one model, Qwen3.8-Flash-Next, on a gaming PC: 125B parameters, 48 layers of 512 experts with 10 routed per token, compressed to 2 to 4 bits per weight by ISTA-DASLab and Unsloth. It builds on parts of llama.cpp and ggml. Its README publishes these, measured by its authors on their hardware, none of them on ours:

| Machine | Size | Writing (tokens a second) | Reading a prompt (tokens a second) |
|---|---|---|---|
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB | Q2_0 | 94 | 2,650 |
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB | IQ3_S | 53 | 1,620 |
| RX 9070 XT 16 GB, Ryzen 9 3900X, 47 GB | Q2_0 | 60 | 1,160 |

How it gets there, from its `docs/HOW_IT_WORKS.md` and `docs/DETAILS.md`:

- Every expert sits in RAM, pinned. The GPU keeps the most-used ones and **the CPU computes the rest where they are**, at the same time as the GPU, so a token rarely waits on PCIe. The SSD holds only a 28.8 GB lookup table the model reads a few rows of per token.
- Weights are 2 to 4 bits, through ggml's i-quant kernels, and prompts run on int8 tensor-core kernels.
- The model's own draft layer proposes up to 3 tokens, checked in one pass (1.6 to 1.8 times faster), plus prompt lookup.
- Prompts run in chunks of up to 8,192 tokens with experts streamed to the GPU; the key-value cache moves to RAM past 64K.
- OpenAI, Anthropic Messages and Responses APIs, tool calls and MCP, images, a chat page, a one-click installer for Windows and Linux, AMD through HIP, and several GPUs.

Its quality is measured statistically against an FP16 path (KL divergence, top-1 agreement), and some of its speedups change the bits of a long prompt. Strata reads experts from the SSD only for its largest 4-bit sizes, which it rates at 7 to 8.5 tokens a second, several times slower than its other sizes.

## Where Anchor stands

As of 2026-10-06 gates A7 to A12 have passed, and the README's table says what each showed. When this was written: two models, SmolLM2-135M and OLMoE-1B-7B, in float32 over their published weights, on a CPU or an NVIDIA GPU, matched to a pinned reference. Results are the same bit for bit however the experts are cached or read, under speculation and with requests running together. OLMoE on the RTX 3070 here decodes 61 tokens a second from memory and 8.7 with its experts read off the drive.

## Decisions, 2026-10-05

Taken by the owner:

1. **Target model: Qwen3-30B-A3B first.** It is one step from OLMoE (query and key norms, grouped-query attention, 128 experts, 8 routed), Qwen publishes it as GGUF from 4 to 8 bits, and at 61 GB in BF16 and 18.6 GB at Q4_K_M it exercises the drive and the memory cache on a 64 GB machine. A model of 100B or more follows once quantized weights and CPU expert compute have passed.
2. **Bit for bit stays.** When the CPU computes experts the GPU lacks, it sums in the GPU kernel's order, so logits never depend on which device ran an expert or what any cache holds. This is the property Strata does not have, and it is kept on purpose.
3. **Scope: all of it.** Engine speed, the API surface, installing and a chat page, and AMD and several GPUs, in the order below.

## Decisions, 2026-10-06

Taken by the owner, after gate A10 measured llama.cpp reading prompts 1.6 to 2.6 times as fast on the same GPU, mostly because it computes in less precision:

1. **A faster mode in less precision, opt-in.** The experts and matrix products may run on the GPU's tensor cores in less than float32, behind a flag. The default stays float32 and bit for bit, and every gate is checked in it; the fast mode is measured against the reference the way quantization was (KL divergence, top-1 agreement), not held to it bit for bit.
2. **The stack merges as it goes green**, and the gates after A10 go on without stopping for a go-ahead at each.

## How a speed claim is made

Strata runs only its own model and needs 12 GB of VRAM, more than this machine's RTX 3070 has, and Anchor will not run Strata's model soon. So no figure here is a head-to-head with Strata. A speed claim for Anchor is measured on one named machine and set beside **llama.cpp at a pinned commit running the same GGUF on the same machine**, which is the engine Strata builds on and the fair baseline for a general one. Strata's published figures stay context, never a comparison.

## The gates

Each continues the README's ladder, with a pass condition a test checks. Figures are published only as measured, with the hardware named.

| Gate | Passes when | Why it is here |
|---|---|---|
| A7, quantized weights | The engine reads GGUF checkpoints with F32, F16, BF16, Q8_0, Q4_K and Q6_K tensors, on the CPU and the GPU. With OLMoE's official GGUFs (`allenai/OLMoE-1B-7B-0924-GGUF`, pinned by revision and SHA-256) at Q8_0 and Q4_K_M, logits match the pinned reference run on the same dequantized weights within 1e-2, and 192 of 192 greedy tokens are equal. The CPU and the GPU dequantize every block to the same bits. The cost against the BF16 model is measured (KL divergence and top-1 agreement) and published, not gated | Strata's first factor: a 4-bit expert is a quarter of the bytes to hold, copy and read |
| A8, the CPU computes what the GPU lacks | On a GPU, a token's experts that the GPU's cache lacks run on the CPU from memory, at the same time as the GPU runs the others, with each sum in the GPU kernel's order. Logits are byte-identical to the GPU-only path at every cache size, on OLMoE and on Qwen3-30B-A3B. Tokens a second measured beside the GPU-only path | Strata's second factor: a missing expert costs a read from RAM, not a copy over PCIe |
| A9, Qwen3-30B-A3B | A head of its own for each query and key norm, expert weights renormalized over those used, head size 128 in prompt attention, and Qwen's pre-tokenizer. Logits match a pinned reference within 1e-2, the router chooses the reference's experts, and 192 of 192 greedy tokens are equal, on the CPU and the GPU, from BF16 and from Q8_0 and Q4_K_M. The reference records layer by layer, since the model in float32 is twice this machine's memory. Speed measured beside llama.cpp | The target model |
| A10, long context | A 32K-token prompt runs on Qwen3-30B-A3B with the key-value cache that does not fit on the GPU kept in page-locked host memory. At a length both fit, logits are byte-identical to the cache-on-GPU run. Prompt reading measured at 4K and 32K tokens | Strata reads 32K prompts at over 1,000 tokens a second |
| A11, the API surface | Anthropic's `/v1/messages` (streamed events included), tool calls in OpenAI's and Anthropic's shapes, written in the model's own tool format, and OpenAI's Responses API. Each is tested against recorded request and response shapes, and a session with Claude Code and one with Codex CLI are checked by hand | Coding agents are where a local model earns its keep |
| A12, installing and a chat page | One command on Windows and one on Linux detects the GPU, memory and disk, picks a model and size that fit, fetches it with resume and SHA-256 checks, and starts the server. A chat page talks to it. Checked by hand on a fresh Windows and a fresh Linux machine | Strata's installer is most of why it spread |
| A13, AMD and several GPUs | The A7 and A9 comparisons pass on an AMD GPU through HIP, and on a model split across two NVIDIA GPUs | Needs hardware this machine does not have |

A7 and A8 can be built and checked on OLMoE today, before A9 exists. A9 needs a new recorder, since `reference/record_moe.py` holds the whole model in memory, which for Qwen3-30B-A3B in float32 is 122 GB.

## What this machine can and cannot show

AMD Ryzen 7 3700X (8 cores, AVX2, no AVX-512), 64 GB, an RTX 3070 with 8 GB of which the desktop holds about 2, and two Gen3 NVMe drives. Qwen3-30B-A3B at Q4_K_M fits in its memory; in BF16 it does not, so that path reads experts from the drive. Anything Strata does with AVX-512, a 12 GB card or Gen4 drives cannot be measured here.
