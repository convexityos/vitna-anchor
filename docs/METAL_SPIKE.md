# Apple Silicon Metal Backend Spike and Go/No-Go Decision

**Date:** 2026-11-21
**Baseline commit:** `a3a5908` (`origin/main`)
**Decision:** **Conditional Go** (architecture validated against the codebase; execution deferred until dedicated Apple Silicon hardware is provisioned for gate verification).

## 1. Current GPU codebase inventory

Measured directly against `engine/` on `origin/main`:

- **`engine/src/model_cuda.cu`:** `5,537` lines implementing **`42` `__global__` CUDA kernels** and `34` `LAUNCH(...)` template instantiations across `9` stored weight dtypes (`F32`, `BF16`, `F16`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`).
- **`engine/include/model_cuda.h`:** Narrow C boundary (`261` lines) exposing `vitna_cuda_probe`, `vitna_cuda_devices`, `vitna_cuda_create`, `vitna_cuda_step`, `vitna_cuda_steps`, `vitna_cuda_rows`, `vitna_cuda_moe_route`, `vitna_cuda_moe_experts`, `vitna_cuda_moe_rows_*`, `vitna_cuda_widen`, and `vitna_cuda_expert_check`. `model.c` calls only this C header.
- **`engine/include/gpu_compat.h`:** (`131` lines) Remaps CUDA runtime symbols to AMD ROCm HIP (`-DVITNA_HIP=ON`) with `VITNA_HAS_WMMA 0` so non-NVIDIA targets compile cleanly without NVIDIA tensor cores.
- **`engine/include/gpu_plan.h` / `engine/src/gpu_plan.c`:** Pure C11 multi-device memory budgeting and layer-split planning (`vitna_gpu_plan_split`), reusable as-is for any future backend.

## 2. Porting analysis for Apple Silicon Metal

### What translates cleanly

1. **C11 host boundary:** Everything outside `model_cuda.cu` (`model.c`, `quant.c`, `expert_stream.c`, `tokenizer.c`, `api.c`, `crypto.c`, `server.c`) compiles unmodified on macOS `arm64` (`clang`). Only `engine/CMakeLists.txt` / `engine/Makefile` need a `-DVITNA_METAL=ON` target linking `-framework Metal -framework Foundation` and compiling one Objective-C++ bridge (`engine/src/model_metal.mm`) plus `.metal` shader library (`engine/src/kernels.metal`).
2. **Dequantization & exact `float32` arithmetic:** Every GPU kernel in `model_cuda.cu` computes in IEEE-754 `float32` (`fma`, warp reductions, no `--use_fast_math`) and widens `Q8_0`/`Q4_K`/`Q6_K`/`Q2_K`/`Q3_K`/`Q5_K` blocks identically to `quant.c`. Metal Shading Language (MSL 3.x) supports `float`, `fma()`, `simdgroup` width-32 shuffle/sum (`simd_sum`, `simd_shuffle_xor` on Apple Silicon GPUs), and function templates (MSL 2.2+), preserving the **A4/A7/A9 bit-for-bit reference gates**.
3. **Unified memory advantage:** On Apple Silicon (`M1`/`M2`/`M3`/`M4` Pro/Max/Ultra), CPU and GPU share coherent physical memory via `MTLStorageModeShared` (`newBufferWithBytesNoCopy`). When a model fits in unified RAM:
   - The discrete GPU's `expert_cache` + PCIe H2D async copy ring (`cudaMemcpyAsync`, `cudaHostRegister`, `PENDING_MAX` copy stream) is unnecessary: kernels read `mmap`-ed checkpoint expert buffers directly from shared memory.
   - When a model exceeds unified RAM (e.g. Qwen3-30B-A3B BF16 `61.1 GB` on a `16 GB` or `24 GB` MacBook), `expert_stream.c`'s direct-I/O NVMe expert cache (`--expert-cache`, using `fcntl(F_NOCACHE)` on macOS) feeds shared `MTLBuffer` slots without double-buffering across a PCIe bus.

### Key engineering constraints & risks

1. **Shader divergence from `model_cuda.cu`:** Unlike ROCm HIP (`gpu_compat.h`), Metal cannot reuse CUDA C++ `.cu` kernel source via header macros. All `42` kernels (dense decode/prefill/multi-row, attention tiled/long, MoE fused gate+up/down, QK-norm + RoPE, widening/check kernels) must be maintained in `.metal` alongside `model_cuda.cu`, or factored into shared scalar math headers.
2. **IEEE-754 contraction rules:** Passing Metal fast-math (`MTLCompileOptions.fastMathEnabled = YES`, enabled by default in Xcode/Metal) breaks associative rounding needed for bit-for-bit reference parity (`docs/PARITY.md`); fast-math MUST stay `NO` by default (parallel to `--precision exact` vs opt-in `--precision fast` via MPSGraph/simdgroup_matrix).
3. **CI and gate enforcement:** `vitna-anchor` Gate A0/A4 rule prohibits publishing unverified GPU backends. Current GitHub Actions macOS runners lack Apple Silicon Metal GPU test execution unless a self-hosted `arm64` Mac runner is attached, mirroring CUDA's compile-only CI gap (`A4`).

## 3. Go / No-Go recommendation

- **Engineering feasibility:** **GO** (narrow `model_cuda.h` C ABI, pure `float32` MSL compute + `simdgroup` 32-wide match, unified memory simplifies MoE scheduling).
- **Schedule decision:** **HOLD implementation** until dedicated Apple Silicon test hardware (`M2/M3/M4` with >= 24 GB unified memory) is attached to run the full reference parity gates (`SmolLM2-135M`, `OLMoE-1B-7B`, `Qwen3-30B-A3B` F32 & Q4_K_M, `tests/reference*.test.mjs`) locally and record reproducible `bench/history.json` entries beside `llama.cpp` Metal on identical hardware.

### Proposed phased gates when Apple Silicon hardware is provisioned (~1 week initial M0-M1 spike)

| Phase | Deliverable | Gate pass condition |
|---|---|---|
| **M0** | CMake `-DVITNA_METAL=ON`, Objective-C bridge (`model_metal.mm`), F32/F16/BF16 MSL kernels (`SmolLM2-135M`) | `vitna-anchor-tests` + A2/A4 reference logits within `1e-2` and `192/192` greedy tokens equal on M-series GPU |
| **M1** | `Q8_0`, `Q4_K`, `Q6_K` widening + zero-copy unified-memory MoE kernels (`OLMoE-1B-7B`) | `vitna-anchor widen`/`expert-check` pass bit-for-bit; `OLMoE-1B-7B` Q4_K_M matches A7 reference (`192/192` tokens) |
| **M2** | `Qwen3-30B-A3B` Q4_K_M + NVMe `F_NOCACHE` streaming (`--expert-cache`), head-to-head `bench/history.json` vs `llama.cpp` Metal | A9/A10 reference parity passes; measured prefill + decode recorded in `bench/history.json` and `README.md` |
