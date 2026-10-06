"""Record, or check, the reference that gate A14 pins and the engine's embeddings must match.

    python reference/record_embed.py [--model-dir DIR] [--out FILE]
    python reference/record_embed.py --check [--model-dir DIR]

Runs the pinned embedding model, BAAI/bge-small-en-v1.5, through Hugging Face
transformers on the CPU, in float32, with eager attention, on one thread, and
writes token ids and embeddings for the inputs in
reference/bge-small-en-v1.5/inputs.json to
reference/bge-small-en-v1.5/fixture.json:

- token ids for every input and every corpus string twice: from transformers,
  and from the tokenizers library running the model's own tokenizer.json;
- for every input, from one forward pass: the embedding as sentence-transformers
  computes it with this model's modules.json (the last hidden state at the
  [CLS] position, divided by its L2 norm), that state's L2 norm before the
  division, and the mean of the last hidden state over every position, divided
  by its L2 norm, which is the pooling other BERT-family embedding models use,
  so the engine's mean pooling is held to the same pass without pinning a
  second model.

--check records again into memory and compares with the committed fixture:
ids must match exactly, and the largest embedding difference is reported.

The model files are read from models/bge-small-en-v1.5 (fetch them with
`node scripts/fetch-model.mjs bge-small-en-v1.5`) and checked against the pin
before anything runs. Nothing is downloaded here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
from pathlib import Path

import record
from record import check_versions, cpu_name, f32, sha256_file, sha256_text

ROOT = Path(__file__).resolve().parent.parent
NAME = "bge-small-en-v1.5"
REF = ROOT / "reference" / NAME
FIXTURE_FORMAT = 1

# The tolerance gate A14 is held to, from reference/compare.mjs, which says
# why it is 1e-5 and not the 1e-4 first written here. --check holds the
# reference itself to it on other machines.
EMBED_ATOL = 1e-5

# The most tokens the model reads at once: its 512 positions, [CLS] and [SEP]
# included. An input longer than this is refused here, as the engine refuses it.
MAX_TOKENS = 512


def check_model_dir(model_dir: Path) -> dict:
    pin = json.loads((REF / "model.json").read_text(encoding="utf-8"))
    for f in pin["files"]:
        path = model_dir / f["path"]
        if not path.exists():
            sys.exit(f"{path} is missing; run: node scripts/fetch-model.mjs {NAME}")
        digest = sha256_file(path)
        if digest != f["sha256"]:
            sys.exit(f"{path}: sha256 {digest}, pinned {f['sha256']}")
    return {"repo": pin["repo"], "revision": pin["revision"], "files": pin["files"]}


def record_embeddings(model_dir: Path, inputs: dict, allow_other: bool) -> dict:
    software = check_versions(allow_other)
    model_pin = check_model_dir(model_dir)

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    # sentence-transformers runs this model as modules.json says: the
    # transformer, then pooling as 1_Pooling/config.json says, then Normalize.
    modules = [m["type"] for m in json.loads((model_dir / "modules.json").read_text(encoding="utf-8"))]
    pooling = json.loads((model_dir / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    if modules != [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ] or not pooling["pooling_mode_cls_token"] or pooling["pooling_mode_mean_tokens"]:
        sys.exit("the pinned model's modules are not Transformer, CLS pooling, Normalize; this recorder computes that")

    file_tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    corpus = []
    for text in inputs["corpus"]:
        ids = tok(text)["input_ids"]
        corpus.append({"text": text, "ids": ids, "tokenizer_json_ids": file_tok.encode(text).ids})

    embedded = []
    for item in inputs["inputs"]:
        text = item["text"]
        ids = tok(text)["input_ids"]
        # The model runs on these ids, so both tokenizers must agree on them.
        if file_tok.encode(text).ids != ids:
            sys.exit(f"input {item['id']}: transformers and tokenizer.json give different ids; choose another input")
        if len(ids) > MAX_TOKENS:
            sys.exit(f"input {item['id']}: {len(ids)} tokens, more than the model's {MAX_TOKENS}; shorten it")
        out = model(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            token_type_ids=torch.zeros(1, len(ids), dtype=torch.long),
        )
        hidden = out.last_hidden_state[0]
        cls = hidden[0]
        mean = hidden.mean(dim=0)
        cls_norm = torch.linalg.vector_norm(cls)
        embedded.append(
            {
                "id": item["id"],
                "text": text,
                "ids": ids,
                "cls_norm": f32(cls_norm.item()),
                "embedding": [f32(v) for v in torch.nn.functional.normalize(cls, p=2, dim=0).tolist()],
                "mean_embedding": [f32(v) for v in torch.nn.functional.normalize(mean, p=2, dim=0).tolist()],
            }
        )

    return {
        "format": FIXTURE_FORMAT,
        "recorded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "recorder": {
            "script": "reference/record_embed.py",
            "sha256_lf": sha256_text(Path(__file__)),
            "imports": {"script": "reference/record.py", "sha256_lf": sha256_text(Path(record.__file__))},
        },
        "inputs_file": {"file": f"reference/{NAME}/inputs.json", "sha256_lf": sha256_text(REF / "inputs.json")},
        "model": {
            "repo": model_pin["repo"],
            "revision": model_pin["revision"],
            "hidden_size": model.config.hidden_size,
            "max_tokens": MAX_TOKENS,
            "pooling": "cls",
            "normalize": True,
            "query_instruction": inputs["query_instruction"],
        },
        "software": software,
        "hardware": {
            "cpu": cpu_name(),
            "machine": platform.machine(),
            "os": platform.platform(),
            "torch_cpu_capability": __import__("torch").backends.cpu.get_cpu_capability(),
        },
        "settings": {
            "dtype": "float32",
            "attn_implementation": "eager",
            "threads": 1,
            "deterministic_algorithms": True,
            "token_type_ids": 0,
        },
        # Where transformers' ids differ from the model's own tokenizer.json.
        "tokenizer_disagreements": [c["text"] for c in corpus if c["ids"] != c["tokenizer_json_ids"]],
        "corpus": corpus,
        "inputs": embedded,
    }


def compare(old: dict, new: dict) -> int:
    """Print how a fresh recording differs from the committed one. Returns 1 if ids differ or values moved too far."""
    failures = 0
    for a, b in zip(old["corpus"], new["corpus"]):
        if a["ids"] != b["ids"] or a["tokenizer_json_ids"] != b["tokenizer_json_ids"]:
            print(f"corpus {a['text']!r}: ids changed")
            failures += 1
    worst = {"embedding": 0.0, "mean_embedding": 0.0, "cls_norm_relative": 0.0}
    for a, b in zip(old["inputs"], new["inputs"]):
        if a["ids"] != b["ids"]:
            print(f"input {a['id']}: ids differ")
            failures += 1
            continue
        for key in ("embedding", "mean_embedding"):
            worst[key] = max(worst[key], max(abs(x - y) for x, y in zip(a[key], b[key])))
        worst["cls_norm_relative"] = max(worst["cls_norm_relative"], abs(a["cls_norm"] - b["cls_norm"]) / a["cls_norm"])
    print("largest difference against the committed fixture:")
    for k, v in worst.items():
        print(f"  {k:17s} {v:.3e}")
    print(f"recorded on: {old['hardware']['cpu']} / {old['hardware']['torch_cpu_capability']}")
    print(f"checked on:  {new['hardware']['cpu']} / {new['hardware']['torch_cpu_capability']}")
    if max(worst["embedding"], worst["mean_embedding"]) > EMBED_ATOL:
        print(f"the reference itself moved by more than the tolerance ({EMBED_ATOL}) on this machine")
        failures += 1
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("ANCHOR_EMBED_MODEL_DIR", str(ROOT / "models" / NAME)))
    ap.add_argument("--out", default=str(REF / "fixture.json"))
    ap.add_argument("--check", action="store_true", help="record into memory and compare with the committed fixture")
    ap.add_argument("--allow-other-versions", action="store_true", help="run on versions other than the pinned ones (--check only)")
    args = ap.parse_args()

    inputs = json.loads((REF / "inputs.json").read_text(encoding="utf-8"))
    if args.allow_other_versions and not args.check:
        sys.exit("--allow-other-versions is for --check only; a fixture is recorded with the pinned versions")
    fixture = record_embeddings(Path(args.model_dir), inputs, args.allow_other_versions)

    if args.check:
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        return compare(old, fixture)

    text = json.dumps(fixture, ensure_ascii=True, indent=1) + "\n"
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(text.encode('utf-8'))} bytes): {len(fixture['inputs'])} inputs embedded, "
          f"{len(fixture['corpus'])} corpus strings tokenized")
    if fixture["tokenizer_disagreements"]:
        print(f"transformers and tokenizer.json disagree on {len(fixture['tokenizer_disagreements'])} corpus strings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
