"""How far plausible defects move the embeddings gate A14 compares, against its tolerance.

    python reference/embed_sensitivity.py [--model-dir DIR]

Runs the pinned model through transformers as reference/record_embed.py
does, and again with one defect at a time, each a mistake an implementation
of BERT could make and still produce plausible vectors: GELU in its tanh
form, LayerNorm's epsilon at 1e-5 rather than the config's 1e-12, the
token-type embedding left out, attention scores unscaled, positions counted
from 1, and the mean taken over every position but [CLS] and [SEP]. For each
it prints the largest absolute difference from the clean run over every
input in reference/bge-small-en-v1.5/inputs.json, and whether the tolerance
in reference/compare.mjs (EMBED_ATOL) would catch it. It also prints the
largest difference float64 arithmetic makes, the scale of rounding.

Every defect has to move some embedding by more than the tolerance, and
rounding by far less, or the tolerance is in the wrong place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from record_embed import EMBED_ATOL, NAME, REF, ROOT, check_model_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("ANCHOR_EMBED_MODEL_DIR", str(ROOT / "models" / NAME)))
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    check_model_dir(model_dir)

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_dir)
    inputs = [i["text"] for i in json.loads((REF / "inputs.json").read_text(encoding="utf-8"))["inputs"]]
    ids = [tok(t)["input_ids"] for t in inputs]

    def run(dtype=torch.float32, mutate=None, positions_from=0, pool="cls"):
        model = AutoModel.from_pretrained(model_dir, dtype=dtype, attn_implementation="eager").eval()
        if mutate:
            mutate(model)
        outs = []
        for row in ids:
            n = len(row)
            kw = dict(input_ids=torch.tensor([row]), attention_mask=torch.ones(1, n, dtype=torch.long),
                      position_ids=torch.arange(positions_from, positions_from + n).unsqueeze(0))
            h = model(**kw).last_hidden_state[0].to(torch.float64)
            v = h[0] if pool == "cls" else (h[1:-1].mean(0) if pool == "inner" else h.mean(0))
            outs.append(torch.nn.functional.normalize(v, dim=0))
        return outs

    def worst(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a, b))

    clean = run()
    clean_mean = run(pool="mean")

    def tanh_gelu(m):
        for layer in m.encoder.layer:
            layer.intermediate.intermediate_act_fn = torch.nn.GELU(approximate="tanh")

    def ln_eps(m):
        for mod in m.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.eps = 1e-5

    def no_token_type(m):
        m.embeddings.token_type_embeddings.weight.data.zero_()

    def unscaled(m):
        for layer in m.encoder.layer:
            layer.attention.self.scaling = 1.0

    rows = [
        ("float64 arithmetic (rounding)", worst(clean, run(dtype=torch.float64))),
        ("GELU in its tanh form", worst(clean, run(mutate=tanh_gelu))),
        ("LayerNorm epsilon 1e-5", worst(clean, run(mutate=ln_eps))),
        ("token-type embedding left out", worst(clean, run(mutate=no_token_type))),
        ("attention scores unscaled", worst(clean, run(mutate=unscaled))),
        ("positions counted from 1", worst(clean, run(positions_from=1))),
        ("mean pooling without [CLS] and [SEP]", worst(clean_mean, run(pool="inner"))),
    ]
    print(f"largest |defect - clean| over {len(ids)} inputs; tolerance {EMBED_ATOL}")
    bad = 0
    for i, (name, d) in enumerate(rows):
        caught = d > EMBED_ATOL
        if (i == 0 and caught) or (i > 0 and not caught):
            bad += 1
        print(f"  {name:38s} {d:.3e}  {'caught' if caught else 'within the tolerance'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
