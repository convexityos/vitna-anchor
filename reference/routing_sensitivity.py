"""Measure how much routing near-ties matter, against the OLMoE fixture.

    python reference/routing_sensitivity.py [--model-dir DIR]

These are the measurements behind the routing rule in reference/compare.mjs,
which reference/README.md quotes:

1. Forced. Each prompt's closest routing decision at a prompt position is
   decided the other way, its 8th expert replaced by the runner-up at the
   runner-up's own weight, twice: once with every other decision left to
   the router, which may then decide some of them differently too, and once
   with every other decision held to the fixture's experts, each weighted as
   the router weighs it, as the engine's --experts-in holds them. Printed for
   each: the largest change in the logits the fixture keeps, at any
   position, and whether any position's argmax changed.
2. Reordered. Each prompt and its greedy tokens, all but the last, go through
   the model in one pass, which batches positions the recording fed one at a
   time: first on one thread with eager attention, as recorded, then on eight
   threads with PyTorch's SDPA attention. Printed: the largest changes in the
   logits and the router logits the fixture keeps, and every routing
   decision that changed.

The model is loaded as record_moe.py loads it, after the same checks, and
needs as much memory. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import record
import record_moe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(record_moe.ROOT / "models" / record_moe.NAME))
    args = ap.parse_args()
    model_dir = Path(args.model_dir)

    record.check_versions(False)
    record_moe.check_model_dir(model_dir)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    fixture = json.loads((record_moe.REF / "fixture.json").read_text(encoding="utf-8"))
    probe_ids = fixture["probe_ids"]
    k = fixture["model"]["experts_per_token"]

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
    )
    record_moe.check_weights(model, model_dir)
    model.eval()
    routers = [layer.mlp.gate for layer in model.model.layers]

    def kept(p, logits, start=0):
        """Largest change in the logits the fixture keeps for p's prompt positions, and the argmax changes."""
        worst, moved = 0.0, []
        for t, pos in enumerate(p["positions"]):
            for i, v in pos["top"] + list(zip(probe_ids, pos["probe"])):
                worst = max(worst, abs(float(logits[start + t][i]) - float(np.float32(v))))
            if int(np.argmax(logits[start + t])) != pos["top"][0][0]:
                moved.append(t)
        return worst, moved

    print("forced: each prompt's closest routing decision at a prompt position, decided the other way")
    for p in fixture["prompts"]:
        margin, t, layer = min(
            (min(lg[:k]) - lg[k], t, layer) for t, r in enumerate(p["routing"]) for layer, lg in enumerate(r["logits"])
        )
        ids = p["routing"][t]["experts"][layer]
        drop, add = ids[k - 1], ids[k]

        def swap(module, inputs, output, t=t, drop=drop, add=add):
            router_logits, weights, chosen = output
            weights, chosen = weights.clone(), chosen.clone()
            slot = int((chosen[t] == drop).nonzero()[0, 0])
            chosen[t, slot] = add
            weights[t, slot] = torch.softmax(router_logits[t].float(), dim=-1)[add].to(weights.dtype)
            return router_logits, weights, chosen

        def hold(l, p=p, t=t, layer=layer, add=add):
            def hook(module, inputs, output):
                router_logits, weights, chosen = output
                pinned = torch.tensor([r["experts"][l][:k] for r in p["routing"]], dtype=chosen.dtype)
                if l == layer:
                    pinned[t, k - 1] = add  # in place of drop, the 8th of the fixture's order
                weights = torch.gather(torch.softmax(router_logits.float(), dim=-1), 1, pinned).to(weights.dtype)
                return router_logits, weights, pinned
            return hook

        results = []
        for others, hooks in (
            ("left to the router", lambda: [routers[layer].register_forward_hook(swap)]),
            ("held", lambda: [r.register_forward_hook(hold(l)) for l, r in enumerate(routers)]),
        ):
            hs = hooks()
            try:
                logits = model(torch.tensor([p["ids"]])).logits[0].numpy()
            finally:
                for h in hs:
                    h.remove()
            worst, moved = kept(p, logits)
            results.append(f"the others {others}, kept logits moved by up to {worst:.3e} (argmax changed at {moved or 'no position'})")
        print(f"  {p['id']:8s} position {t:3d} layer {layer:2d}, margin {margin:.2e}: " + "; ".join(results), flush=True)

    captured = []
    hooks = [r.register_forward_hook(lambda m, i, out: captured.append((out[0].numpy().copy(), out[2].numpy().copy())))
             for r in routers]
    for name, threads, attention in (("one thread, eager attention", 1, "eager"), ("eight threads, SDPA attention", 8, "sdpa")):
        torch.set_num_threads(threads)
        model.set_attn_implementation(attention)
        worst_logit = worst_router = 0.0
        moved, changed = [], []
        for p in fixture["prompts"]:
            n = len(p["ids"])
            captured.clear()
            logits = model(torch.tensor([p["ids"] + p["greedy_ids"][:-1]])).logits[0].numpy()
            w, m = kept(p, logits)
            worst_logit = max(worst_logit, w)
            moved += [(p["id"], "position", t) for t in m]
            for s, g in enumerate(p["greedy"]):
                for i, v in g["top"]:
                    worst_logit = max(worst_logit, abs(float(logits[n - 1 + s][i]) - float(np.float32(v))))
                if int(np.argmax(logits[n - 1 + s])) != g["id"]:
                    moved.append((p["id"], "greedy step", s))
            runs = [("position", t, r) for t, r in enumerate(p["routing"])]
            runs += [("greedy token", s, r) for s, r in enumerate(p["greedy_routing"])]
            for row, (where, t, r) in enumerate(runs):
                for layer, (router_logits, chosen) in enumerate(captured):
                    for e, v in zip(r["experts"][layer], r["logits"][layer]):
                        worst_router = max(worst_router, abs(float(router_logits[row][e]) - float(np.float32(v))))
                    if sorted(int(e) for e in chosen[row]) != sorted(r["experts"][layer][:k]):
                        lg = r["logits"][layer]
                        changed.append((p["id"], where, t, layer, f"margin {min(lg[:k]) - lg[k]:.2e}"))
        print(f"reordered, {name}: kept logits moved by up to {worst_logit:.3e}, router logits by up to {worst_router:.3e}")
        print(f"  argmax changed at {moved or 'no position'}; routing changed at {changed or 'no decision'}", flush=True)
    for h in hooks:
        h.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
