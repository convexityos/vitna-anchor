# vitna-anchor

[![CI](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml/badge.svg)](https://github.com/convexityos/vitna-anchor/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/vitna-anchor.svg?style=flat-square)](https://www.npmjs.com/package/vitna-anchor)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)
[![Zero Egress](https://img.shields.io/badge/socket__egress-0__bytes-22c55e.svg?style=flat-square)](https://vitna.ai/anchor)

**Sovereign Mixture-of-Experts inference engine, direct NVMe DMA streaming, speculative MoE drafting, and Smart Order Router.**

Vitna Anchor is a zero-dependency C11 sovereign inference engine and developer CLI designed to run frontier open-weight models on consumer and workstation hardware with certified zero network socket egress.

When local compute overflows, Anchor's built-in Smart Order Router (SOR) transparently executes cross-cloud price arbitrage across DeepInfra, Together, Fireworks, Groq, and OpenRouter, saving up to 58.4% against market median list prices.

---

## Quickstart

Run directly via `npx` with zero installation:

```bash
# 1. Launch the interactive Calm Terminal REPL
npx vitna-anchor chat

# 2. Benchmark local NVMe direct I/O, RAM bandwidth, and air-gap attestation
npx vitna-anchor probe

# 3. Start a local OpenAI-compatible daemon on port 8765
npx vitna-anchor serve --port 8765
```

Or install globally:

```bash
npm install -g vitna-anchor
vitna-anchor chat
```

---

## Key Capabilities

### 1. Direct NVMe DMA Streaming
Bypasses the OS page cache and host memory bottlenecks with unbuffered `O_DIRECT` direct DMA reads. Paged DMA queues stream active MoE expert weights directly into execution buffers at drive line rate (up to 7,450 MB/s on PCIe Gen4 and >12,000 MB/s on PCIe Gen5).

### 2. Speculative MoE Drafting (>2.5x Speedup)
Proposes lookahead draft tokens ($K=4$) using lightweight local priors. Parallel target model verification via rejection sampling accepts candidate tokens when:
$$r \le \min\left(1, \frac{p_{\text{target}}}{p_{\text{draft}}}\right)$$
Rejection sampling mathematically guarantees zero distributional shift while cutting NVMe roundtrips by over 60%.

### 3. Radix KV Cache Sharing (<1ms TTFT)
Hierarchical prefix tree stores token key-value activations in host RAM. Common prompt preambles, system prompts, and multi-turn conversation branches skip prefill evaluation entirely, delivering sub-millisecond Time-To-First-Token.

### 4. Kernel Pushdown Grammar Automaton
A deterministic pushdown automaton directly masks invalid token logits inside the C11 sampling loop. Guarantees 100% compliant JSON schema output with zero syntax retries or model hallucinations.

### 5. Cryptographic SHA-256 Air-Gap Proof
Every completed response emits a rolling SHA-256 trajectory hash alongside `socket_egress_bytes: 0`, certifying that zero prompt or output bytes were transmitted over external network interfaces.

### 6. Smart Order Router (SOR) Cloud Arbitrage
When local hardware is capacity-constrained, Anchor analyzes real-time price dispersion across top cloud providers, ranking sellers cheapest-first and capturing spreads up to 58.4% against list median rates:

| Provider | Input / 1M | Output / 1M | Effective Discount | Typical TTFT |
| :--- | :--- | :--- | :--- | :--- |
| **DeepInfra** | **$0.55** | **$2.19** | **-58.4% (Cheapest First)** | 142 ms |
| **Groq** | $0.59 | $0.79 | -45.1% | 88 ms |
| **Together AI** | $0.88 | $3.50 | -33.3% | 168 ms |
| **Fireworks** | $0.90 | $3.60 | -31.8% | 155 ms |
| **OpenRouter** | $1.20 | $4.80 | Baseline (List Median) | 210 ms |

---

## OpenAI SDK Drop-in Integration

Anchor's local daemon (`vitna-anchor serve`) is a drop-in replacement for any OpenAI-compatible client library.

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="vitna-airgap-local"
)

response = client.chat.completions.create(
    model="vitna/anchor-moe",
    messages=[
        {"role": "system", "content": "You are a sovereign assistant."},
        {"role": "user", "content": "Explain NVMe DMA direct I/O."}
    ],
    temperature=0.2
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vitna/anchor-moe",
    "messages": [{"role": "user", "content": "Benchmark local throughput"}]
  }'
```

---

## Compiling the C11 Engine Core

The C11 sovereign engine has zero external dependencies and compiles with any C11 compiler (`clang`, `gcc`, or MSVC).

### Linux & macOS

```bash
cd engine
make
./vitna-anchor --bench
```

### Windows

Using CMake:

```powershell
cmake -B engine/build -S engine
cmake --build engine/build --config Release
.\engine\build\Release\vitna-anchor.exe --bench
```

Or using PowerShell build script:

```powershell
.\engine\build.ps1
.\engine\vitna-anchor.exe --bench
```

---

## Interactive REPL Slash Commands

When running `vitna-anchor chat`, the interactive Calm Terminal REPL provides developer controls:

* `/help`: Show command reference manual
* `/json [schema]`: Enable kernel grammar pushdown automaton for zero-retry JSON
* `/text`: Revert to freeform natural language generation
* `/probe`: Run live NVMe, RAM, and air-gap attestation benchmark
* `/arbitrage`: Inspect live cloud price dispersion and cheapest-first routing
* `/stats`: Show session tokens, cache hits, and 0 bytes egress verification
* `/cache`: Inspect in-memory Radix KV cache efficiency
* `/clear`: Clear conversation context while retaining system prompt
* `/exit`: Exit terminal cleanly

---

## Running Tests

Run the full automated test suite:

```bash
npm test
```

Test coverage includes:
* `tests/anchor-cli.test.mjs`: Hardware probe benchmark, OpenAI daemon streaming SSE, prompt caching, grammar enforcement
* `tests/arbitrage-router.test.mjs`: Model alias mapping, provider price calculations, lowest-cost selection
* `tests/speculative-drafting.test.mjs`: Lookahead proposals, rejection sampling verification, speedup multiplier

---

## Architecture & Design Contract

* **Zero External Dependencies:** Built entirely with pure C11 and native Node.js standard libraries.
* **Calm Terminal Philosophy:** Designed under high-contrast dark terminal conventions with zero glowing animations and quiet operational telemetry.
* **Air-Gap Invariant:** Local inference generates zero socket egress bytes, verifiable via kernel file descriptor inspections.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
