# Checkpoint Attestation (SHA-256)

Added in [PR #61](https://github.com/convexityos/vitna-anchor/pull/61). `vitna-anchor serve` can cryptographically identify the exact model weights backing an HTTP inference endpoint at startup, advertise the digest on health checks, and stamp every generation and embedding response with the active checkpoint hash.

For regulated, air-gapped, or audit-heavy on-premises inference, this binds every model output to an independently reproducible byte digest of the weights on disk without calling any external service.

## Enabling attestation

Pass `--attest-checkpoint` when starting the HTTP server:

```bash
./engine/vitna-anchor serve \
  --model models/qwen3-30b-a3b \
  --weights models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --attest-checkpoint
```

The flag works for both generation models (`llama`, `olmoe`, `qwen3_moe`) and BERT embedding models (`--model models/bge-small-en-v1.5 --attest-checkpoint`), implemented in `engine/src/main.c`.

Startup is **fail-closed**: if `--attest-checkpoint` is set and the checkpoint path cannot be stated/read, or a model directory contains no recognized weight or configuration files, `vitna-anchor` exits non-zero before binding a listening socket (`checkpoint attestation failed: ...`).

## Canonical digest specification

`vitna_checkpoint_sha256()` (`engine/src/crypto.c`) targets `--weights` when given, or otherwise `--model`:

1. **Single checkpoint file (`.gguf` or `.safetensors`):** computes raw lowercase 64-character hex SHA-256 directly over the file bytes.
2. **Model directory:** scans top-level regular files matching `.gguf`, `.safetensors`, `config.json`, `tokenizer.json`, or `tokenizer_config.json`; rejects directories with zero matching artifacts; sorts matched filenames in POSIX lexicographical (`strcmp`) order; hashes each file individually; and computes the SHA-256 of the UTF-8 manifest:

```text
<filename_1>:<sha256_1>\n
<filename_2>:<sha256_2>\n
...
```

This manifest format matches `vitna-code`'s `compute_checkpoint_digest()` byte for byte (`tests/unit_tests.c`: `test_checkpoint_attestation`).

## Verification surfaces

When `--attest-checkpoint` is active (`vitna_api_set_checkpoint_sha256` in `engine/src/api.c`):

- **`GET /health` and `GET /v1/health`:** Include `"checkpoint_sha256":"<64-char lowercase hex>"` alongside model metadata.
- **Response headers:** Every generation response (`POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/messages`, `POST /v1/responses`) and embedding response (`POST /v1/embeddings`) emits:

```http
x-vitna-checkpoint-sha256: <64-char lowercase hex>
```

When `--attest-checkpoint` is omitted, both the JSON field and header are omitted so existing payloads remain unchanged.

## Operator and client verification recipe

1. Compute the expected SHA-256 independently before deploying (single `.gguf` example):

```bash
sha256sum models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf
```

```powershell
(Get-FileHash models\qwen3-30b-a3b-gguf\Qwen3-30B-A3B-Q4_K_M.gguf -Algorithm SHA256).Hash.ToLower()
```

2. Compare the startup stdout (`Attested checkpoint sha256: ...`), `GET /health` `"checkpoint_sha256"`, and per-request `x-vitna-checkpoint-sha256` header against that expected digest:

```bash
curl -s http://127.0.0.1:8765/health
curl -i http://127.0.0.1:8765/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

3. Upstream clients such as `vitna-code` (`VitnaAnchorProvider`) verify `checkpoint_sha256` against local weights and record the digest inside signed execution receipts.

## Scope and trust boundary

- Checkpoint attestation proves **weight identity**, not full remote hardware attestation (no TPM/SEV-SNP quote or signed runtime binary attestation is emitted by the server).
- The HTTP server listens unauthenticated by default on `127.0.0.1`; treat network exposure and weight filesystem permissions as operator-side controls.
