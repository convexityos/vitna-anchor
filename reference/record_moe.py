"""Record, or check, the reference that gate A5 pins: a mixture of experts.

    python reference/record_moe.py [--model-dir DIR] [--out FILE]
    python reference/record_moe.py --check [--model-dir DIR]

Records OLMoE-1B-7B, pinned in reference/olmoe-1b-7b/model.json, as
record.py records SmolLM2-135M for gate A1: the same pinned software, the
same settings, and the same fields, so reference/compare.mjs holds an engine
to either fixture. The inputs are reference/olmoe-1b-7b/prompts.json and the
fixture is reference/olmoe-1b-7b/fixture.json.

What a mixture of experts adds is a choice. At every layer, a router scores
all 64 experts for each token, and the token goes through the 8 that score
highest. Two experts whose scores nearly tie can trade places when the same
sums are added in another order, and the output then moves by far more than
the rounding did. So the fixture also keeps the routing, at every layer of
every prompt position and of every token greedy decoding feeds back:

- routing.experts: the 8 experts the model chose, in the router's order,
  then the one that came closest;
- routing.logits: the router logits of those 9;
- routing.lse: the logsumexp of all 64 router logits, so each chosen
  expert's weight is exp(logit - lse) (OLMoE does not renormalize the 8).

The experts run as transformers' eager implementation, one expert at a time,
which this file selects as A1 selects eager attention. Loading must map every
tensor in the checkpoint to the model and leave none of the model's own
unset, or recording stops: transformers fills a missing weight with random
numbers and only warns. Then every tensor in the checkpoint is compared with
the model's copy, each expert included, since transformers stacks the experts
as it loads and a wrong stacking would raise nothing.

--check records again into memory and compares with the committed fixture,
as record.py --check does, and also reports every routing decision that
changed.

The helpers, the pinned versions and the logit comparison come from
record.py. Its own fixture pins its hash, so this file imports them rather
than changing it, and this fixture pins the hashes of both files.

The model files are read from models/olmoe-1b-7b (fetch them with
`node scripts/fetch-model.mjs olmoe-1b-7b`, 13.8 GB) and checked against the
pin before anything runs. The model takes 27.7 GB in float32, and loading it
took the recorder's memory to nearly 36 GB.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import platform
import sys
import time
from pathlib import Path

import record
from record import GREEDY_TOPK, LOGIT_ATOL, TOPK, check_versions, cpu_name, f32, row_summary, sha256_file, sha256_text

ROOT = Path(__file__).resolve().parent.parent
NAME = "olmoe-1b-7b"
REF = ROOT / "reference" / NAME

FIXTURE_FORMAT = 1

# The 8 experts a token goes through, and the one that came closest.
ROUTER_TOPK = 9


def check_model_dir(model_dir: Path) -> dict:
    pin = json.loads((REF / "model.json").read_text(encoding="utf-8"))
    for f in pin["files"]:
        path = model_dir / f["path"]
        if not path.exists():
            sys.exit(f"{path} is missing; run: node scripts/fetch-model.mjs {NAME}")
        if path.stat().st_size != f["size"]:
            sys.exit(f"{path}: {path.stat().st_size} bytes, pinned {f['size']}")
        digest = sha256_file(path)
        if digest != f["sha256"]:
            sys.exit(f"{path}: sha256 {digest}, pinned {f['sha256']}")
    return {"repo": pin["repo"], "revision": pin["revision"], "files": pin["files"]}


def check_weights(model, model_dir: Path) -> None:
    """Compare every tensor in the checkpoint with the model's copy of it.

    transformers 5.17 stacks each layer's 64 experts into two tensors as it
    loads, the gate and up projections concatenated in one, and its loading
    report leaves conversion errors out. A wrong stacking would load without
    a word, so each expert's three matrices are found in the stacks and
    compared with the checkpoint's, and every parameter must be covered.
    """
    import re

    import torch
    from safetensors import safe_open

    params = dict(model.named_parameters())
    experts = model.config.num_experts
    expert_key = re.compile(r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    seen = set()
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(model_dir / shard), framework="pt") as f:
            for key in f.keys():
                m = expert_key.fullmatch(key)
                if m:
                    stack, e, kind = m[1], int(m[2]), m[3]
                    if kind == "down":
                        part, got = (f"{stack}.down_proj", e, kind), params[f"{stack}.down_proj"][e]
                    else:
                        both = params[f"{stack}.gate_up_proj"][e]
                        half = both.shape[0] // 2
                        part, got = (f"{stack}.gate_up_proj", e, kind), both[:half] if kind == "gate" else both[half:]
                else:
                    part, got = (key,), params.get(key)
                want = f.get_tensor(key).to(torch.float32)
                if got is None or got.shape != want.shape or not torch.equal(got, want):
                    sys.exit(f"{key}: the model's copy differs from the checkpoint's")
                if part in seen:
                    sys.exit(f"{key}: two checkpoint tensors map to the same place")
                seen.add(part)
    expected = set()
    for name in params:
        if name.endswith(".experts.gate_up_proj"):
            expected |= {(name, e, kind) for e in range(experts) for kind in ("gate", "up")}
        elif name.endswith(".experts.down_proj"):
            expected |= {(name, e, "down") for e in range(experts)}
        else:
            expected.add((name,))
    if seen != expected:
        sys.exit(f"checkpoint tensors not in the model: {sorted(seen - expected)[:4]}; "
                 f"model parameters not in the checkpoint: {sorted(expected - seen)[:4]}")
    print(f"weights: all {len(seen)} checkpoint tensors equal the model's copies", flush=True)


def to_json(value, level: int = 0) -> str:
    """JSON indented as record.py writes it, except that a list of numbers stays on one line.

    One number to a line, as record.py writes, would make this fixture
    about 40% larger.
    """
    pad = " " * (level + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{pad}{json.dumps(k, ensure_ascii=True)}: {to_json(v, level + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
    if isinstance(value, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            return json.dumps(value)
        return "[\n" + ",\n".join(pad + to_json(v, level + 1) for v in value) + "\n" + " " * level + "]"
    return json.dumps(value, ensure_ascii=True)


def record_moe(model_dir: Path, inputs: dict, allow_other: bool) -> dict:
    software = check_versions(allow_other)
    model_pin = check_model_dir(model_dir)

    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        attn_implementation="eager",
        experts_implementation="eager",
        output_loading_info=True,
    )
    unmapped = {k: sorted(map(str, v))[:8] for k, v in loading.items() if v}
    if unmapped:
        sys.exit(f"the checkpoint and the model do not map one to one: {unmapped}")
    if model.config._experts_implementation != "eager":
        sys.exit(f"the experts run as {model.config._experts_implementation!r}, not eager")
    check_weights(model, model_dir)
    model.eval()
    config = model.config
    vocab = config.vocab_size
    layers = config.num_hidden_layers
    chosen_per_token = config.num_experts_per_tok

    # The tokenizers library running the model's own tokenizer.json, as in
    # record.py. For this model transformers runs that same file, and the two
    # are expected to agree everywhere.
    file_tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    # The probe ids are chosen as record.py chooses them, over this vocabulary.
    probe_ids = sorted(int(i) for i in np.random.default_rng(20260929).choice(vocab, record.PROBES, replace=False))

    # Every router's output, layer by layer, for each forward pass: its
    # logits over all experts, and the experts the model then used.
    captured: list[tuple] = []

    def keep(module, args, output):
        router_logits, _weights, chosen = output
        captured.append((router_logits.numpy().copy(), chosen.numpy().copy()))

    hooks = [layer.mlp.gate.register_forward_hook(keep) for layer in model.model.layers]

    def take_routing(rows: int) -> list[dict]:
        """The routing of the last forward pass, one entry per token."""
        if len(captured) != layers:
            sys.exit(f"expected {layers} routers to run, saw {len(captured)}")
        out = []
        for t in range(rows):
            experts, logits, lse = [], [], []
            for router_logits, chosen in captured:
                row = router_logits[t].astype(np.float32)
                picked = [int(i) for i in chosen[t]]
                rest = np.array([-np.inf if i in picked else v for i, v in enumerate(row)], dtype=np.float64)
                runner_up = int(np.argsort(-rest, kind="stable")[0])
                ids = picked + [runner_up]
                experts.append(ids)
                logits.append([f32(row[i]) for i in ids])
                r64 = row.astype(np.float64)
                m = float(r64.max())
                lse.append(m + math.log(float(np.exp(r64 - m).sum())))
            out.append({"experts": experts, "logits": logits, "lse": lse})
        captured.clear()
        return out

    corpus = []
    for text in inputs["corpus"]:
        ids = tok(text)["input_ids"]
        corpus.append(
            {"text": text, "ids": ids, "tokenizer_json_ids": file_tok.encode(text).ids, "decoded": tok.decode(ids)}
        )

    steps = inputs["greedy_steps"]
    prompts = []
    for p in inputs["prompts"]:
        started = time.monotonic()
        ids = tok(p["text"])["input_ids"]
        if file_tok.encode(p["text"]).ids != ids:
            sys.exit(f"prompt {p['id']}: transformers and tokenizer.json give different ids; choose another prompt")

        # The whole prompt in one pass, through the key-value cache greedy
        # decoding continues from.
        captured.clear()
        out = model(torch.tensor([ids]), use_cache=True)
        logits = out.logits[0].numpy()
        routing = take_routing(len(ids))
        positions = [row_summary(logits[t], probe_ids) for t in range(len(ids))]
        last = logits[-1].astype("<f4")

        # Greedy decoding, one token per step. Each chosen token but the last
        # goes back through the model, and that pass's routing is kept:
        # greedy_routing[s] is the routing of the token chosen at step s, run
        # at position len(ids) + s, in the pass whose logits choose step s + 1.
        past = out.past_key_values
        row = logits[-1]
        greedy, greedy_routing = [], []
        for s in range(steps):
            order = np.argsort(-row, kind="stable")
            chosen = int(order[0])
            greedy.append(
                {
                    "id": chosen,
                    "top": [[int(i), f32(row[i])] for i in order[:GREEDY_TOPK]],
                    "margin": f32(row[order[0]] - row[order[1]]),
                }
            )
            if s == steps - 1:
                break
            out = model(torch.tensor([[chosen]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            row = out.logits[0, -1].numpy()
            greedy_routing.extend(take_routing(1))

        # transformers' own greedy search must choose the same tokens. It
        # stops early at the end-of-text token, so compare that far.
        generated = model.generate(
            torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            do_sample=False,
            max_new_tokens=steps,
            pad_token_id=config.pad_token_id,
        )[0, len(ids):].tolist()
        captured.clear()
        mine = [g["id"] for g in greedy]
        if generated != mine[: len(generated)]:
            sys.exit(f"prompt {p['id']}: generate() chose {generated}, the step loop chose {mine}")

        prompts.append(
            {
                "id": p["id"],
                "text": p["text"],
                "ids": ids,
                "positions": positions,
                "last_logits_f32le_b64": base64.b64encode(last.tobytes()).decode("ascii"),
                "greedy": greedy,
                "greedy_ids": mine,
                "greedy_text": tok.decode(mine),
                "routing": routing,
                "greedy_routing": greedy_routing,
            }
        )
        print(f"  {p['id']}: {len(ids)} positions, {steps} greedy steps, {time.monotonic() - started:.0f} s", flush=True)

    for h in hooks:
        h.remove()

    return {
        "format": FIXTURE_FORMAT,
        "recorded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "recorder": {
            "script": "reference/record_moe.py",
            "sha256_lf": sha256_text(Path(__file__)),
            "helpers": {"script": "reference/record.py", "sha256_lf": sha256_text(Path(record.__file__))},
        },
        "inputs": {"file": f"reference/{NAME}/prompts.json", "sha256_lf": sha256_text(REF / "prompts.json")},
        "model": {
            "repo": model_pin["repo"],
            "revision": model_pin["revision"],
            "vocab_size": vocab,
            "layers": layers,
            "experts": config.num_experts,
            "experts_per_token": chosen_per_token,
        },
        "software": software,
        "hardware": {
            "cpu": cpu_name(),
            "machine": platform.machine(),
            "os": platform.platform(),
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
        "settings": {
            "dtype": "float32",
            "attn_implementation": "eager",
            "experts_implementation": "eager",
            "threads": 1,
            "deterministic_algorithms": True,
            "topk": TOPK,
            "greedy_topk": GREEDY_TOPK,
            "greedy_steps": steps,
            "router_topk": ROUTER_TOPK,
            "tie_break": "lowest token id",
        },
        "probe_ids": probe_ids,
        "tokenizer_disagreements": [c["text"] for c in corpus if c["ids"] != c["tokenizer_json_ids"]],
        "corpus": corpus,
        "prompts": prompts,
    }


def router_margins(fixture: dict):
    """Every routing decision as (margin, prompt, where, layer): how far the 8th expert's logit was above the 9th's."""
    for p in fixture["prompts"]:
        for where, entries in (("position", p["routing"]), ("greedy", p["greedy_routing"])):
            for t, r in enumerate(entries):
                for layer, logits in enumerate(r["logits"]):
                    yield min(logits[:-1]) - logits[-1], p["id"], f"{where} {t}", layer


def describe_routing(fixture: dict) -> None:
    margins = sorted(router_margins(fixture))
    print(f"routing: {len(margins)} decisions; the 8th and 9th router logits are")
    for bound in (1e-2, 1e-3, 1e-4, 1e-5):
        print(f"  within {bound:.0e} in {sum(m < bound for m, *_ in margins)}")
    for m, pid, where, layer in margins[:5]:
        print(f"  {m:.3e}  {pid} {where} layer {layer}")


def compare_moe(old: dict, new: dict) -> int:
    """record.compare, then every routing decision that changed. Returns 1 on any failure."""
    failures = record.compare(old, new)
    worst = 0.0
    changed = []
    for a, b in zip(old["prompts"], new["prompts"]):
        for key in ("routing", "greedy_routing"):
            for t, (ra, rb) in enumerate(zip(a[key], b[key])):
                for layer in range(len(ra["experts"])):
                    ea, eb = ra["experts"][layer], rb["experts"][layer]
                    if sorted(ea[:-1]) != sorted(eb[:-1]):
                        la = ra["logits"][layer]
                        changed.append((a["id"], key, t, layer, min(la[:-1]) - la[-1]))
                        continue
                    by_id = dict(zip(eb, rb["logits"][layer]))
                    for e, v in zip(ea, ra["logits"][layer]):
                        if e in by_id:
                            worst = max(worst, abs(v - by_id[e]))
    print(f"  router  {worst:.3e}  (largest router logit difference where the experts agree)")
    for pid, key, t, layer, margin in changed:
        print(f"routing changed: {pid} {key} {t} layer {layer}, where the committed margin was {margin:.3e}")
    if changed or worst > LOGIT_ATOL:
        failures = 1
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(ROOT / "models" / NAME))
    ap.add_argument("--out", default=str(REF / "fixture.json"))
    ap.add_argument("--check", action="store_true", help="record into memory and compare with the committed fixture")
    ap.add_argument("--allow-other-versions", action="store_true", help="run on versions other than the pinned ones (--check only)")
    args = ap.parse_args()

    inputs = json.loads((REF / "prompts.json").read_text(encoding="utf-8"))
    if args.allow_other_versions and not args.check:
        sys.exit("--allow-other-versions is for --check only; a fixture is recorded with the pinned versions")
    started = time.monotonic()
    fixture = record_moe(Path(args.model_dir), inputs, args.allow_other_versions)
    print(f"recorded in {time.monotonic() - started:.0f} s")
    describe_routing(fixture)

    if args.check:
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        return compare_moe(old, fixture)

    text = to_json(fixture) + "\n"
    if json.loads(text) != fixture:
        sys.exit("the fixture does not read back as itself")
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
