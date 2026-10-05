"""Record, or check, the reference that gate A7 pins: OLMoE from a GGUF file.

    python reference/record_gguf.py --file olmoe-1b-7b-0924-q8_0.gguf [--out FILE]
    python reference/record_gguf.py --file olmoe-1b-7b-0924-q8_0.gguf --check

Gate A7 asks whether the engine computes, from a quantized checkpoint, what
the model computes with those weights. The weights a quantized file holds are
its integers times its scales, so the reference here is gate A5's own
recording (record_moe.py, unchanged) run on the model with every weight
replaced by the GGUF file's, widened to float32 by gguf-py, which is
llama.cpp's reference implementation of the block formats. Everything else is
A5's: the pinned software, the settings, the prompts, the fields, the
routing. The fixture lands in reference/olmoe-1b-7b-gguf/ as
fixture-<file's quantization>.json.

Two things are added:

- weights.sha256_f32: for every tensor, the SHA-256 of its widened float32
  values, little-endian, row-major in the checkpoint's shape, each layer's
  experts one after another in expert order under the name of their stack.
  The engine widens the same file and must give the same digests, which is
  how "dequantizes every block to the same bits" is checked rather than
  believed.
- against_bf16: what quantizing cost, measured against gate A5's fixture
  of the BF16 model: the KL divergence at each prompt's last position (where
  both fixtures keep all 50,304 logits), how often the top token agrees at
  every prompt position, and how many greedy tokens agree before the first
  that differs. Measured, not gated.

The replacement is checked as A5's loading is: every GGUF tensor maps to a
place in the model, every parameter of the model is written exactly once,
and each written value is read back equal. record_moe.py's own hash is pinned
by A5's fixture, so this file reaches into it rather than changing it: its
check_weights, which runs once the model is loaded, is replaced by the
replacement.

The GGUF file is read from models/olmoe-1b-7b-gguf (fetch it with
`node scripts/fetch-model.mjs olmoe-1b-7b-gguf`) and checked against
reference/olmoe-1b-7b-gguf/model.json first; config.json and the tokenizer
come from models/olmoe-1b-7b, checked against A5's pin. Peak memory is A5's,
about 36 GB.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import record
import record_moe
from record import sha256_file, sha256_text

ROOT = Path(__file__).resolve().parent.parent
NAME = "olmoe-1b-7b-gguf"
REF = ROOT / "reference" / NAME
A5_REF = ROOT / "reference" / "olmoe-1b-7b"

GGUF_PINNED = "0.19.0"

# llama.cpp's tensor names and the Hugging Face names they stand for, as
# engine/src/gguf.c maps them. A stack of experts maps to transformers'
# stacked parameters: gate and up concatenated in gate_up_proj, down alone.
MODEL_PARTS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
LAYER_PARTS = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate_inp": "mlp.gate",
}
EXPERT_PARTS = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}


def check_gguf_version() -> str:
    import importlib.metadata

    found = importlib.metadata.version("gguf")
    if found != GGUF_PINNED:
        sys.exit(f"gguf is {found}; pinned {GGUF_PINNED}. See reference/requirements.txt.")
    return found


def check_gguf_file(path: Path) -> dict:
    pin = json.loads((REF / "model.json").read_text(encoding="utf-8"))
    entry = next((f for f in pin["files"] if f["path"] == path.name), None)
    if not entry:
        sys.exit(f"{path.name} is not in {REF / 'model.json'}")
    if not path.exists():
        sys.exit(f"{path} is missing; run: node scripts/fetch-model.mjs {NAME}")
    if path.stat().st_size != entry["size"]:
        sys.exit(f"{path}: {path.stat().st_size} bytes, pinned {entry['size']}")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        sys.exit(f"{path}: sha256 {digest}, pinned {entry['sha256']}")
    return {"repo": pin["repo"], "revision": pin["revision"], "file": entry["path"], "size": entry["size"], "sha256": entry["sha256"]}


def hf_place(name: str):
    """Where a GGUF tensor goes: ("plain", hf_name) or ("experts", stack_prefix, part)."""
    if name in MODEL_PARTS:
        return ("plain", MODEL_PARTS[name])
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or not parts[1].isdigit() or parts[3] != "weight":
        return None
    layer, part = int(parts[1]), parts[2]
    if part in LAYER_PARTS:
        return ("plain", f"model.layers.{layer}.{LAYER_PARTS[part]}.weight")
    if part in EXPERT_PARTS:
        return ("experts", f"model.layers.{layer}.mlp.experts", EXPERT_PARTS[part])
    return None


def replace_weights(gguf_path: Path, digests: dict, types: dict):
    """A check_weights for record_moe: every parameter replaced by the GGUF file's, widened by gguf-py."""
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    def replace(model, _model_dir):
        params = dict(model.named_parameters())
        reader = GGUFReader(str(gguf_path))
        written = set()
        for t in reader.tensors:
            place = hf_place(t.name)
            if place is None:
                sys.exit(f"{t.name}: not a tensor the engine knows where to put")
            types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
            shape = tuple(int(d) for d in reversed(t.shape))
            w = np.ascontiguousarray(dequantize(t.data, t.tensor_type).astype(np.float32, copy=False).reshape(shape))
            if place[0] == "plain":
                key = place[1]
                digests[key] = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
                p = params.get(key)
                if p is None or tuple(p.shape) != shape:
                    sys.exit(f"{t.name} -> {key}: the model has no parameter of shape {shape} there")
                targets = [(key, p, w)]
            else:
                stack, kind = place[1], place[2]
                digests[f"{stack}.{kind}_proj.weight"] = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
                if kind == "down":
                    p = params[f"{stack}.down_proj"]
                    if tuple(p.shape) != shape:
                        sys.exit(f"{t.name}: shape {shape}, the model's down_proj {tuple(p.shape)}")
                    targets = [((f"{stack}.down_proj",), p, w)]
                else:
                    both = params[f"{stack}.gate_up_proj"]
                    half = both.shape[1] // 2
                    if (both.shape[0], half, both.shape[2]) != shape:
                        sys.exit(f"{t.name}: shape {shape}, the model's gate_up_proj {tuple(both.shape)}")
                    view = both[:, :half] if kind == "gate" else both[:, half:]
                    targets = [((f"{stack}.gate_up_proj", kind), view, w)]
            for mark, p, values in targets:
                if mark in written:
                    sys.exit(f"{t.name}: a second tensor writes {mark}")
                written.add(mark)
                with torch.no_grad():
                    p.copy_(torch.from_numpy(values))
                if not torch.equal(p, torch.from_numpy(values)):
                    sys.exit(f"{t.name}: the value written does not read back")
        expected = set()
        for name in params:
            if name.endswith(".experts.gate_up_proj"):
                expected |= {(name, "gate"), (name, "up")}
            elif name.endswith(".experts.down_proj"):
                expected.add((name,))
            else:
                expected.add(name)
        if written != expected:
            missing = sorted(map(str, expected - written))[:4]
            extra = sorted(map(str, written - expected))[:4]
            sys.exit(f"model parameters the GGUF file does not write: {missing}; written but not parameters: {extra}")
        print(f"weights: all {len(written)} parameters replaced by {gguf_path.name}'s, widened by gguf-py", flush=True)

    return replace


def against_bf16(fixture: dict) -> dict:
    """What quantizing cost, against gate A5's recording of the BF16 model."""
    import numpy as np

    bf16 = json.loads((A5_REF / "fixture.json").read_text(encoding="utf-8"))
    kl, top1_same, top1_total, greedy_same, greedy_total, first_diff = [], 0, 0, 0, 0, []
    for a, q in zip(bf16["prompts"], fixture["prompts"]):
        if a["ids"] != q["ids"]:
            sys.exit(f"prompt {a['id']}: the two fixtures tokenize it differently")
        p = np.frombuffer(base64.b64decode(a["last_logits_f32le_b64"]), dtype="<f4").astype(np.float64)
        r = np.frombuffer(base64.b64decode(q["last_logits_f32le_b64"]), dtype="<f4").astype(np.float64)
        lp = p - (p.max() + math.log(np.exp(p - p.max()).sum()))
        lr = r - (r.max() + math.log(np.exp(r - r.max()).sum()))
        kl.append(float((np.exp(lp) * (lp - lr)).sum()))
        for pa, pq in zip(a["positions"], q["positions"]):
            top1_total += 1
            top1_same += pa["top"][0][0] == pq["top"][0][0]
        diff = next((i for i, (x, y) in enumerate(zip(a["greedy_ids"], q["greedy_ids"])) if x != y), None)
        first_diff.append(diff)
        greedy_total += len(a["greedy_ids"])
        greedy_same += len(a["greedy_ids"]) if diff is None else diff
    return {
        "against": "reference/olmoe-1b-7b/fixture.json",
        "kl_last_position": [round(x, 6) for x in kl],
        "kl_last_position_mean": round(sum(kl) / len(kl), 6),
        "top1_agree": top1_same,
        "top1_positions": top1_total,
        "greedy_agree_before_first_difference": greedy_same,
        "greedy_tokens": greedy_total,
        "greedy_first_difference": first_diff,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the GGUF file's name in reference/olmoe-1b-7b-gguf/model.json")
    ap.add_argument("--gguf-dir", default=str(ROOT / "models" / NAME))
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "olmoe-1b-7b"))
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true", help="record into memory and compare with the committed fixture")
    args = ap.parse_args()

    gguf_path = Path(args.gguf_dir) / args.file
    quant = args.file.rsplit("-", 1)[-1].removesuffix(".gguf")
    out = Path(args.out) if args.out else REF / f"fixture-{quant}.json"
    gguf_version = check_gguf_version()
    weights = check_gguf_file(gguf_path)

    digests: dict = {}
    types: dict = {}
    record_moe.check_weights = replace_weights(gguf_path, digests, types)

    inputs = json.loads((A5_REF / "prompts.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    fixture = record_moe.record_moe(Path(args.model_dir), inputs, False)
    print(f"recorded in {time.monotonic() - started:.0f} s")
    record_moe.describe_routing(fixture)

    # The tokenizer is A5's and its fixture holds the corpus; this one keeps the model's computation.
    del fixture["corpus"]
    del fixture["tokenizer_disagreements"]
    fixture["recorder"] = {
        "script": "reference/record_gguf.py",
        "sha256_lf": sha256_text(Path(__file__)),
        "runs": {"script": "reference/record_moe.py", "sha256_lf": sha256_text(Path(record_moe.__file__))},
        "helpers": {"script": "reference/record.py", "sha256_lf": sha256_text(Path(record.__file__))},
    }
    fixture["software"]["gguf"] = gguf_version
    fixture["weights"] = {
        **weights,
        "widened_by": f"gguf {gguf_version}, gguf.quants.dequantize",
        "tensor_types": dict(sorted(types.items())),
        "sha256_f32": dict(sorted(digests.items())),
    }
    fixture["against_bf16"] = against_bf16(fixture)
    print(f"against BF16: {json.dumps(fixture['against_bf16'])}")

    if args.check:
        old = json.loads(out.read_text(encoding="utf-8"))
        failures = record_moe.compare_moe(old, fixture)
        if old["weights"]["sha256_f32"] != fixture["weights"]["sha256_f32"]:
            print("the widened weights differ from the committed digests")
            failures = 1
        return failures

    text = record_moe.to_json(fixture) + "\n"
    if json.loads(text) != fixture:
        sys.exit("the fixture does not read back as itself")
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
