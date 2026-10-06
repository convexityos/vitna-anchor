"""Record, or check, the reference that gate A9 pins: Qwen3-30B-A3B, a layer at a time.

    python reference/record_qwen3.py --engine ENGINE [--model-dir DIR] [--out FILE]
    python reference/record_qwen3.py --file Qwen3-30B-A3B-Q8_0.gguf --engine ENGINE [--out FILE]
    python reference/record_qwen3.py --check [--file FILE]

Records Qwen3-30B-A3B, pinned in reference/qwen3-30b-a3b/model.json, as
record_moe.py records OLMoE for gate A5: the same pinned software, the same
settings and the same fields, so reference/compare.mjs and the engine's
tests read this fixture as they read A5's. With --file it records the model
from one of Qwen's own GGUF files, pinned in
reference/qwen3-30b-a3b-gguf/model.json, as record_gguf.py records OLMoE's
for gate A7: every weight is the file's, widened to float32 by gguf-py, and
the fixture keeps each widened tensor's SHA-256 for the engine to match.

What differs is how. The model is 61 GB in bfloat16 and 122 GB in float32,
and the machine that recorded it has 64 GB, so it never exists whole: each
layer is built alone, as transformers' own Qwen3MoeDecoderLayer, given its
weights from the checkpoint, run over every prompt, and dropped before the
next is built, so that about 2.4 GB of weights are held at a time. The
embedding is looked up, each layer runs, and the final norm and output layer
run, in the order Qwen3MoeForCausalLM runs them, with the rotary embedding
and causal mask its model builds. Each prompt runs as one pass over all of
its positions, the greedy tokens included, as a forward pass over the whole
sequence would; A5's fixture, recorded with a cache a token at a time, and
this one are two valid float32 orders of the same computation.

Greedy decoding needs each token before the next can run, which would be a
pass over all 48 layers for each of 192 tokens. Instead the greedy tokens are
proposed, by the engine (--engine) or, for --check, by the committed fixture,
and confirmed: the pass over a prompt and its proposed tokens gives the
reference's logits at each of their positions, and each proposed token must
be the reference's own choice there, the largest logit and the lowest id
between equals, as record_moe.py's step loop chooses. At the first that is
not, the reference's choice replaces it, the engine proposes again from
there, and that prompt runs again, until every token is confirmed. So the
tokens recorded are the reference's whoever proposed them, and the fixture
says how many passes and replacements that took.

Loading is checked as A5's is: every tensor of the checkpoint (or the GGUF
file) is read exactly once and written to exactly one place, each layer's
weights load strictly, so none of the layer's own is left unset, and the
model's experts are transformers' stacks, gate and up concatenated in
gate_up_proj, built from the checkpoint's experts in expert order.

The model files are read from models/qwen3-30b-a3b (fetch them with
`node scripts/fetch-model.mjs qwen3-30b-a3b`, 61.1 GB) and the GGUF files
from models/qwen3-30b-a3b-gguf, each checked against its pin first.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import record
import record_gguf
import record_moe
from record import GREEDY_TOPK, TOPK, check_versions, cpu_name, f32, row_summary, sha256_file, sha256_text

ROOT = Path(__file__).resolve().parent.parent
NAME = "qwen3-30b-a3b"
REF = ROOT / "reference" / NAME
GGUF_NAME = "qwen3-30b-a3b-gguf"
GGUF_REF = ROOT / "reference" / GGUF_NAME

FIXTURE_FORMAT = 1
ROUTER_TOPK = record_moe.ROUTER_TOPK


def check_pinned(directory: Path, pin_path: Path, only: str | None = None) -> dict:
    """Every pinned file (or only the one named) present, of its size and digest."""
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    files = [f for f in pin["files"] if only is None or f["path"] == only]
    if not files:
        sys.exit(f"{only} is not in {pin_path}")
    for f in files:
        path = directory / f["path"]
        if not path.exists():
            sys.exit(f"{path} is missing; run: node scripts/fetch-model.mjs {pin['name']}")
        if path.stat().st_size != f["size"]:
            sys.exit(f"{path}: {path.stat().st_size} bytes, pinned {f['size']}")
        digest = sha256_file(path)
        if digest != f["sha256"]:
            sys.exit(f"{path}: sha256 {digest}, pinned {f['sha256']}")
    return {"repo": pin["repo"], "revision": pin["revision"], "files": files}


class Checkpoint:
    """The checkpoint's tensors by name, widened to float32, each read once."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.where = index["weight_map"]
        self.handles = {shard: safe_open(str(model_dir / shard), framework="pt") for shard in sorted(set(self.where.values()))}
        self.read: set[str] = set()

    def get(self, name: str):
        import torch

        if name not in self.where:
            sys.exit(f"{name}: not in the checkpoint")
        if name in self.read:
            sys.exit(f"{name}: read twice")
        self.read.add(name)
        return self.handles[self.where[name]].get_tensor(name).to(torch.float32)

    def experts(self, layer: int, n: int):
        """transformers' stacks for layer's n experts: gate_up_proj [n, 2I, H] and down_proj [n, H, I]."""
        import torch

        stack = f"model.layers.{layer}.mlp.experts"
        gate_up = down = None
        # Filled in place, an expert at a time: the stacks are 2.4 GB in float32, and copies of them would double that.
        for e in range(n):
            gate, up, dn = (self.get(f"{stack}.{e}.{k}_proj.weight") for k in ("gate", "up", "down"))
            if gate_up is None:
                inter = gate.shape[0]
                gate_up = torch.empty((n, 2 * inter, gate.shape[1]), dtype=torch.float32)
                down = torch.empty((n, *dn.shape), dtype=torch.float32)
            if gate.shape != up.shape or gate.shape[0] != inter or tuple(dn.shape) != tuple(down.shape[1:]):
                sys.exit(f"{stack}.{e}: shapes {tuple(gate.shape)}, {tuple(up.shape)} and {tuple(dn.shape)} do not stack")
            gate_up[e, :inter] = gate
            gate_up[e, inter:] = up
            down[e] = dn
        return gate_up, down

    def unread(self) -> list[str]:
        return sorted(set(self.where) - self.read)


class GGUFWeights:
    """A GGUF file's tensors under their Hugging Face names, widened by gguf-py, each read once and its digest kept."""

    def __init__(self, path: Path, digests: dict, types: dict):
        from gguf import GGUFReader

        self.reader = GGUFReader(str(path))
        self.by_place: dict = {}
        for t in self.reader.tensors:
            place = record_gguf.hf_place(t.name)
            if place is None:
                sys.exit(f"{t.name}: not a tensor the engine knows where to put")
            if place in self.by_place:
                sys.exit(f"{t.name}: two tensors map to {place}")
            self.by_place[place] = t
        self.read: set = set()
        self.digests, self.types = digests, types

    def _widen(self, place):
        import numpy as np
        from gguf.quants import dequantize

        t = self.by_place.get(place)
        if t is None:
            sys.exit(f"{place}: not in the GGUF file")
        if place in self.read:
            sys.exit(f"{place}: read twice")
        self.read.add(place)
        self.types[t.tensor_type.name] = self.types.get(t.tensor_type.name, 0) + 1
        shape = tuple(int(d) for d in reversed(t.shape))
        w = np.ascontiguousarray(dequantize(t.data, t.tensor_type).astype(np.float32, copy=False).reshape(shape))
        key = place[1] if place[0] == "plain" else f"{place[1]}.{place[2]}_proj.weight"
        self.digests[key] = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
        return w

    def get(self, name: str):
        import torch

        return torch.from_numpy(self._widen(("plain", name)))

    def experts(self, layer: int, n: int):
        import torch

        stack = f"model.layers.{layer}.mlp.experts"
        gate, up, down = (torch.from_numpy(self._widen(("experts", stack, k))) for k in ("gate", "up", "down"))
        if gate.shape[0] != n or up.shape != gate.shape or down.shape[0] != n:
            sys.exit(f"layer {layer}: the GGUF file's expert stacks are {tuple(gate.shape)}, {tuple(up.shape)} and {tuple(down.shape)}")
        return torch.cat([gate, up], dim=1), down

    def unread(self) -> list[str]:
        return sorted(str(p) for p in set(self.by_place) - self.read)


def propose(engine: list[str], ids: list[int], count: int) -> list[int]:
    """The engine's greedy continuation of ids, count tokens."""
    if count <= 0:
        return []
    cmd = engine[:1] + ["generate"] + engine[1:] + ["--ids", ",".join(map(str, ids)), "--max-new", str(count), "--greedy", "--ctx", "512"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit(f"the engine could not propose tokens: {r.stderr.decode('utf-8', 'replace').strip()[-400:]}")
    got = json.loads(r.stdout.decode("utf-8"))["ids"]
    if len(got) != count:
        sys.exit(f"the engine proposed {len(got)} tokens, not {count}")
    return [int(i) for i in got]


def stream(config, source, batch: dict, full: bool = True) -> dict:
    """One pass of every layer of config over each prompt in batch, {id: tokens}, the weights read from source.

    Returns, per prompt, its logits and each layer's router logits and chosen
    experts. full requires every tensor source holds to have been read, which
    a model cut short for --self-check does not.
    """
    import torch
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding

    vocab, layers = config.vocab_size, config.num_hidden_layers
    embed = source.get("model.embed_tokens.weight")
    if tuple(embed.shape) != (vocab, config.hidden_size):
        sys.exit(f"the embedding is {tuple(embed.shape)}")
    state = {}
    rotary = Qwen3MoeRotaryEmbedding(config=config)
    for pid, tokens in batch.items():
        h = embed[torch.tensor(tokens)].unsqueeze(0).clone()
        pos = torch.arange(len(tokens)).unsqueeze(0)
        mask = create_causal_mask(config=config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos)
        # Eager attention adds the mask to its scores, and without one attends to every position.
        if mask is None or len(tokens) > 1 and not (float(mask[0, 0, 0, 1]) < -1e30 and float(mask[0, 0, 1, 0]) == 0.0):
            sys.exit("the causal mask is not one eager attention can use")
        state[pid] = {"h": h, "pos": pos, "mask": mask, "rope": rotary(h, position_ids=pos), "logits": [], "chosen": []}
    del embed

    captured: list = []

    def keep(module, args, output):
        router_logits, _scores, chosen = output
        captured.append((router_logits.numpy().copy(), chosen.numpy().copy()))

    for layer_idx in range(layers):
        layer = Qwen3MoeDecoderLayer(config, layer_idx)
        if type(layer.mlp).__name__ != "Qwen3MoeSparseMoeBlock" or layer.mlp.experts.config._experts_implementation != "eager":
            sys.exit(f"layer {layer_idx}: not a mixture of experts run as eager experts")
        prefix = f"model.layers.{layer_idx}."
        gate_up, down = source.experts(layer_idx, config.num_experts)
        sd = {"mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down}
        for name in layer.state_dict():
            if not name.startswith("mlp.experts."):
                sd[name] = source.get(prefix + name)
        layer.load_state_dict(sd, strict=True, assign=True)
        layer.eval()
        hook = layer.mlp.gate.register_forward_hook(keep)
        for st in state.values():
            captured.clear()
            st["h"] = layer(st["h"], attention_mask=st["mask"], position_ids=st["pos"], position_embeddings=st["rope"])
            if len(captured) != 1:
                sys.exit(f"layer {layer_idx}: the router ran {len(captured)} times for one pass")
            st["logits"].append(captured[0][0])
            st["chosen"].append(captured[0][1])
        hook.remove()
        del layer, sd, gate_up, down
        if (layer_idx + 1) % 8 == 0:
            print(f"    layer {layer_idx + 1} of {layers}", flush=True)

    norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.load_state_dict({"weight": source.get("model.norm.weight")}, strict=True)
    head = source.get("lm_head.weight")
    unread = source.unread()
    if full and unread:
        sys.exit(f"tensors the model never read: {unread[:4]}")
    out = {}
    for pid, st in state.items():
        logits = torch.nn.functional.linear(norm(st["h"]), head)[0].numpy()
        out[pid] = {"logits": logits, "router": st["logits"], "chosen": st["chosen"]}
    return out


def self_check(model_dir: Path, layers: int) -> int:
    """transformers' whole model, cut to its first layers, against stream() over the same layers: the same logits, bit for bit."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(model_dir)
    config.num_hidden_layers = layers
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, config=config, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
    )
    model.eval()
    config = model.config
    tok = AutoTokenizer.from_pretrained(model_dir)
    inputs = json.loads((REF / "prompts.json").read_text(encoding="utf-8"))
    failures = 0
    for p in inputs["prompts"][:3]:
        ids = tok(p["text"])["input_ids"]
        whole = model(torch.tensor([ids])).logits[0].numpy()
        streamed = stream(config, Checkpoint(model_dir), {p["id"]: ids}, full=False)[p["id"]]["logits"]
        same = whole.shape == streamed.shape and bool((whole == streamed).all())
        worst = float(abs(whole.astype("float64") - streamed).max())
        print(f"  {p['id']}: {len(ids)} positions, {layers} layers: {'the same bits' if same else f'differ, by up to {worst:.3e}'}")
        failures += not same
    return 1 if failures else 0


def record_qwen3(model_dir: Path, inputs: dict, weights, proposals: dict, engine: list[str] | None, allow_other: bool) -> dict:
    software = check_versions(allow_other)

    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoTokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(model_dir)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    if config.model_type != "qwen3_moe" or config.mlp_only_layers or config.decoder_sparse_step != 1:
        sys.exit("this records a Qwen3-MoE model whose every layer is a mixture of experts")
    vocab, layers = config.vocab_size, config.num_hidden_layers
    tok = AutoTokenizer.from_pretrained(model_dir)
    file_tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    probe_ids = sorted(int(i) for i in np.random.default_rng(20260929).choice(vocab, record.PROBES, replace=False))

    corpus = []
    for text in inputs["corpus"]:
        ids = tok(text)["input_ids"]
        corpus.append({"text": text, "ids": ids, "tokenizer_json_ids": file_tok.encode(text).ids, "decoded": tok.decode(ids)})

    steps = inputs["greedy_steps"]
    prompts = {}
    for p in inputs["prompts"]:
        ids = tok(p["text"])["input_ids"]
        if file_tok.encode(p["text"]).ids != ids:
            sys.exit(f"prompt {p['id']}: transformers and tokenizer.json give different ids; choose another prompt")
        prompts[p["id"]] = {"text": p["text"], "ids": ids}

    def routing(res: dict, t: int) -> dict:
        """The routing of position t, as record_moe.py keeps it: the 8 experts chosen, the runner-up, their logits, the logsumexp."""
        experts, logits, lse = [], [], []
        for router_logits, chosen in zip(res["router"], res["chosen"]):
            row = router_logits[t].astype(np.float32)
            picked = [int(i) for i in chosen[t]]
            rest = np.array([-np.inf if i in picked else v for i, v in enumerate(row)], dtype=np.float64)
            ids = picked + [int(np.argsort(-rest, kind="stable")[0])]
            experts.append(ids)
            logits.append([f32(row[i]) for i in ids])
            r64 = row.astype(np.float64)
            m = float(r64.max())
            lse.append(m + math.log(float(np.exp(r64 - m).sum())))
        return {"experts": experts, "logits": logits, "lse": lse}

    pending = {pid: list(proposals[pid]) for pid in prompts}
    done: dict = {}
    passes, replaced = 0, 0
    while pending:
        passes += 1
        started = time.monotonic()
        print(f"  pass {passes}: {', '.join(pending)}", flush=True)
        results = stream(config, weights(), {pid: prompts[pid]["ids"] + cand[: steps - 1] for pid, cand in pending.items()})
        for pid, res in results.items():
            ids, cand = prompts[pid]["ids"], pending[pid]
            n = len(ids)
            choices = [int(np.argsort(-res["logits"][n - 1 + s], kind="stable")[0]) for s in range(steps)]
            first = next((s for s in range(steps) if choices[s] != cand[s]), None)
            if first is None:
                done[pid] = res
                del pending[pid]
                continue
            replaced += 1
            if engine is None:
                sys.exit(f"prompt {pid}: the reference chose {choices[first]} at greedy step {first}, not {cand[first]}")
            fixed = cand[:first] + [choices[first]]
            pending[pid] = fixed + propose(engine, ids + fixed, steps - len(fixed))
            print(f"    {pid}: step {first} is the reference's {choices[first]}, not {cand[first]}; proposed again from there", flush=True)
        print(f"  pass {passes} took {time.monotonic() - started:.0f} s", flush=True)

    out_prompts = []
    for p in inputs["prompts"]:
        pid, ids = p["id"], prompts[p["id"]]["ids"]
        res, n = done[pid], len(prompts[p["id"]]["ids"])
        logits = res["logits"]
        greedy = []
        for s in range(steps):
            row = logits[n - 1 + s]
            order = np.argsort(-row, kind="stable")
            greedy.append({"id": int(order[0]), "top": [[int(i), f32(row[i])] for i in order[:GREEDY_TOPK]],
                           "margin": f32(row[order[0]] - row[order[1]])})
        mine = [g["id"] for g in greedy]
        out_prompts.append(
            {
                "id": pid,
                "text": p["text"],
                "ids": ids,
                "positions": [row_summary(logits[t], probe_ids) for t in range(n)],
                "last_logits_f32le_b64": base64.b64encode(logits[n - 1].astype("<f4").tobytes()).decode("ascii"),
                "greedy": greedy,
                "greedy_ids": mine,
                "greedy_text": tok.decode(mine),
                "routing": [routing(res, t) for t in range(n)],
                "greedy_routing": [routing(res, n + s) for s in range(steps - 1)],
            }
        )

    return {
        "format": FIXTURE_FORMAT,
        "recorded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "recorder": {
            "script": "reference/record_qwen3.py",
            "sha256_lf": sha256_text(Path(__file__)),
            "helpers": {"script": "reference/record.py", "sha256_lf": sha256_text(Path(record.__file__))},
            "moe_helpers": {"script": "reference/record_moe.py", "sha256_lf": sha256_text(Path(record_moe.__file__))},
            "gguf_helpers": {"script": "reference/record_gguf.py", "sha256_lf": sha256_text(Path(record_gguf.__file__))},
        },
        "inputs": {"file": f"reference/{NAME}/prompts.json", "sha256_lf": sha256_text(REF / "prompts.json")},
        "model": {
            "repo": None,
            "revision": None,
            "vocab_size": vocab,
            "layers": layers,
            "experts": config.num_experts,
            "experts_per_token": config.num_experts_per_tok,
            "renormalized": bool(config.norm_topk_prob),
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
            "run": "a layer at a time, each prompt with its greedy tokens in one pass",
            "greedy": "proposed, then each confirmed as the reference's own choice at its position",
            "passes": passes,
            "proposals_replaced": replaced,
        },
        "probe_ids": probe_ids,
        "tokenizer_disagreements": [c["text"] for c in corpus if c["ids"] != c["tokenizer_json_ids"]],
        "corpus": corpus,
        "prompts": out_prompts,
    }


def against_bf16(fixture: dict) -> dict:
    """What quantizing cost, against this model's BF16 fixture, measured as record_gguf.py measures it for OLMoE."""
    saved = record_gguf.A5_REF
    record_gguf.A5_REF = REF
    try:
        out = record_gguf.against_bf16(fixture)
    finally:
        record_gguf.A5_REF = saved
    out["against"] = f"reference/{NAME}/fixture.json"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(ROOT / "models" / NAME))
    ap.add_argument("--file", help="a GGUF file's name in reference/qwen3-30b-a3b-gguf/model.json: record from it")
    ap.add_argument("--gguf-dir", default=str(ROOT / "models" / GGUF_NAME))
    ap.add_argument("--engine", help="the engine that proposes greedy tokens, and any arguments it runs with, as one string")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true", help="record into memory, the committed fixture's tokens proposed, and compare")
    ap.add_argument("--allow-other-versions", action="store_true", help="run on versions other than the pinned ones (--check only)")
    ap.add_argument("--self-check", type=int, metavar="LAYERS",
                    help="compare transformers' whole model, cut to its first LAYERS layers, with the streamed run, and stop")
    args = ap.parse_args()
    if args.self_check:
        check_versions(False)
        return self_check(Path(args.model_dir), args.self_check)
    if args.allow_other_versions and not args.check:
        sys.exit("--allow-other-versions is for --check only; a fixture is recorded with the pinned versions")

    model_dir = Path(args.model_dir)
    model_pin = check_pinned(model_dir, REF / "model.json")
    inputs = json.loads((REF / "prompts.json").read_text(encoding="utf-8"))
    digests: dict = {}
    types: dict = {}
    gguf_version = None
    if args.file:
        quant = args.file.rsplit("-", 1)[-1].removesuffix(".gguf").lower()
        out = Path(args.out) if args.out else GGUF_REF / f"fixture-{quant}.json"
        gguf_version = record_gguf.check_gguf_version()
        gguf_pin = check_pinned(Path(args.gguf_dir), GGUF_REF / "model.json", only=args.file)
        gguf_path = Path(args.gguf_dir) / args.file

        def weights():
            digests.clear()
            types.clear()
            return GGUFWeights(gguf_path, digests, types)
    else:
        out = Path(args.out) if args.out else REF / "fixture.json"

        def weights():
            return Checkpoint(model_dir)

    engine = None
    if args.check:
        old = json.loads(out.read_text(encoding="utf-8"))
        proposals = {p["id"]: p["greedy_ids"] for p in old["prompts"]}
    else:
        if not args.engine:
            sys.exit("--engine is required to record: it proposes the greedy tokens the reference then confirms")
        engine = args.engine.split() + ["--model", str(model_dir)] + (["--weights", str(gguf_path)] if args.file else [])
        proposals = {}
        for p in inputs["prompts"]:
            from transformers import AutoTokenizer

            ids = AutoTokenizer.from_pretrained(model_dir)(p["text"])["input_ids"]
            started = time.monotonic()
            proposals[p["id"]] = propose(engine, ids, inputs["greedy_steps"])
            print(f"  {p['id']}: {inputs['greedy_steps']} tokens proposed in {time.monotonic() - started:.0f} s", flush=True)

    started = time.monotonic()
    fixture = record_qwen3(model_dir, inputs, weights, proposals, engine, args.allow_other_versions)
    print(f"recorded in {time.monotonic() - started:.0f} s")
    record_moe.describe_routing(fixture)
    fixture["model"]["repo"], fixture["model"]["revision"] = model_pin["repo"], model_pin["revision"]
    if args.file:
        # The tokenizer is the BF16 fixture's, which holds the corpus; this one keeps the model's computation.
        del fixture["corpus"]
        del fixture["tokenizer_disagreements"]
        fixture["software"]["gguf"] = gguf_version
        fixture["weights"] = {
            "repo": gguf_pin["repo"],
            "revision": gguf_pin["revision"],
            "file": gguf_pin["files"][0]["path"],
            "size": gguf_pin["files"][0]["size"],
            "sha256": gguf_pin["files"][0]["sha256"],
            "widened_by": f"gguf {gguf_version}, gguf.quants.dequantize",
            "tensor_types": dict(sorted(types.items())),
            "sha256_f32": dict(sorted(digests.items())),
        }
        fixture["against_bf16"] = against_bf16(fixture)
        print(f"against BF16: {json.dumps(fixture['against_bf16'])}")

    if args.check:
        failures = record_moe.compare_moe(old, fixture)
        if args.file and old["weights"]["sha256_f32"] != fixture["weights"]["sha256_f32"]:
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
