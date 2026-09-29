"""Record, or check, the reference that gate A1 pins and gate A2 must match.

    python reference/record.py [--model-dir DIR] [--out FILE]
    python reference/record.py --check [--model-dir DIR]

Runs the pinned model through Hugging Face transformers on the CPU, in
float32, with eager attention, on one thread, and writes token ids and logits
for the inputs in reference/smollm2-135m/prompts.json to
reference/smollm2-135m/fixture.json:

- the tokenizer's ids for every prompt and every corpus string, and the text
  its decoder gives back;
- for every position of every prompt: the top 16 logits with their token
  ids, the logits of 64 fixed probe ids, the maximum, the logsumexp, the mean
  and the standard deviation of the whole row, and the margin between the
  first and second logits;
- the full row of logits at each prompt's last position;
- greedy decoding from each prompt, one token at a time through the key-value
  cache: the chosen ids, and at each step the top 8 logits and the margin.
  transformers' own generate() must choose the same tokens, or recording stops.

--check records again into memory and compares with the committed fixture:
ids must match exactly, and the largest logit difference is reported.

The model files are read from models/smollm2-135m (fetch them with
`node scripts/fetch-model.mjs`) and checked against the pin before anything
runs. Nothing is downloaded here.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "smollm2-135m"

# The versions the fixture is recorded with. A different version may compute
# different floats, so recording refuses to run on anything else.
PINNED = {
    "torch": "2.14.0+cpu",
    "transformers": "5.17.0",
    "tokenizers": "0.23.2",
    "safetensors": "0.8.0",
    "numpy": "2.5.3",
}

TOPK = 16
GREEDY_TOPK = 8
PROBES = 64
FIXTURE_FORMAT = 1

# The logit tolerance gate A2 is held to, from reference/compare.mjs. --check
# holds the reference itself to it on other machines: if torch on another CPU
# differed from the fixture by more than this, the tolerance would mean nothing.
LOGIT_ATOL = 1e-2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    """SHA-256 of a text file with CRLF read as LF, so a Windows checkout hashes the same."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_versions(allow_other: bool) -> dict:
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers

    found = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
    }
    wrong = {k: v for k, v in found.items() if v != PINNED[k]}
    if wrong and not allow_other:
        sys.exit(f"these versions differ from the pin: {wrong}; pinned {PINNED}. See reference/requirements.txt.")
    found["python"] = platform.python_version()
    return found


def check_model_dir(model_dir: Path) -> dict:
    pin = json.loads((REF / "model.json").read_text(encoding="utf-8"))
    for f in pin["files"]:
        path = model_dir / f["path"]
        if not path.exists():
            sys.exit(f"{path} is missing; run: node scripts/fetch-model.mjs")
        digest = sha256_file(path)
        if digest != f["sha256"]:
            sys.exit(f"{path}: sha256 {digest}, pinned {f['sha256']}")
    return {"repo": pin["repo"], "revision": pin["revision"], "files": pin["files"]}


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception as err:  # noqa: BLE001 - the name is a label, not a requirement
        return f"unknown ({err})"
    return "unknown"


def f32(x) -> float:
    """The float32 value, written with the fewest digits that read back as it."""
    import numpy as np

    return float(str(np.float32(x)))


def row_summary(row, probe_ids) -> dict:
    """What the fixture keeps of one row of logits (a float32 numpy vector)."""
    import numpy as np

    order = np.argsort(-row, kind="stable")
    top = [[int(i), f32(row[i])] for i in order[:TOPK]]
    r64 = row.astype(np.float64)
    m = float(r64.max())
    return {
        "top": top,
        "probe": [f32(row[i]) for i in probe_ids],
        "max": f32(row.max()),
        "lse": m + math.log(float(np.exp(r64 - m).sum())),
        "mean": float(r64.mean()),
        "std": float(r64.std()),
        "margin": f32(row[order[0]] - row[order[1]]),
    }


def record(model_dir: Path, inputs: dict, allow_other: bool) -> dict:
    software = check_versions(allow_other)
    model_pin = check_model_dir(model_dir)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    vocab = model.config.vocab_size

    # 64 fixed vocabulary ids spread over the whole vocabulary, so every row
    # is also compared well away from its top entries.
    probe_ids = sorted(int(i) for i in np.random.default_rng(20260929).choice(vocab, PROBES, replace=False))

    corpus = []
    for text in inputs["corpus"]:
        ids = tok(text)["input_ids"]
        corpus.append({"text": text, "ids": ids, "decoded": tok.decode(ids)})

    steps = inputs["greedy_steps"]
    prompts = []
    for p in inputs["prompts"]:
        ids = tok(p["text"])["input_ids"]
        logits = model(torch.tensor([ids])).logits[0].numpy()
        positions = [row_summary(logits[t], probe_ids) for t in range(len(ids))]
        last = logits[-1].astype("<f4")

        # Greedy decoding through the key-value cache, one token per step.
        out = model(torch.tensor([ids]), use_cache=True)
        past = out.past_key_values
        row = out.logits[0, -1].numpy()
        greedy = []
        for _ in range(steps):
            order = np.argsort(-row, kind="stable")
            chosen = int(order[0])
            greedy.append(
                {
                    "id": chosen,
                    "top": [[int(i), f32(row[i])] for i in order[:GREEDY_TOPK]],
                    "margin": f32(row[order[0]] - row[order[1]]),
                }
            )
            out = model(torch.tensor([[chosen]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            row = out.logits[0, -1].numpy()

        # transformers' own greedy search must choose the same tokens. It
        # stops early at the end-of-text token (id 0), so compare that far.
        generated = model.generate(
            torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            do_sample=False,
            max_new_tokens=steps,
            pad_token_id=0,
        )[0, len(ids):].tolist()
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
            }
        )

    return {
        "format": FIXTURE_FORMAT,
        "recorded_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "recorder": {"script": "reference/record.py", "sha256_lf": sha256_text(Path(__file__))},
        "inputs": {"file": "reference/smollm2-135m/prompts.json", "sha256_lf": sha256_text(REF / "prompts.json")},
        "model": {"repo": model_pin["repo"], "revision": model_pin["revision"], "vocab_size": vocab},
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
            "topk": TOPK,
            "greedy_topk": GREEDY_TOPK,
            "greedy_steps": steps,
            "tie_break": "lowest token id",
        },
        "probe_ids": probe_ids,
        "corpus": corpus,
        "prompts": prompts,
    }


def compare(old: dict, new: dict) -> int:
    """Print how a fresh recording differs from the committed one. Returns 1 if ids differ."""
    import numpy as np

    failures = 0
    for a, b in zip(old["corpus"], new["corpus"]):
        if a["ids"] != b["ids"]:
            print(f"corpus {a['text']!r}: ids {a['ids']} became {b['ids']}")
            failures += 1
    worst = {"top": 0.0, "probe": 0.0, "lse": 0.0, "last": 0.0, "greedy": 0.0}
    for a, b in zip(old["prompts"], new["prompts"]):
        if a["ids"] != b["ids"] or a["greedy_ids"] != b["greedy_ids"]:
            print(f"prompt {a['id']}: ids or greedy ids differ")
            failures += 1
            continue
        for pa, pb in zip(a["positions"], b["positions"]):
            worst["top"] = max(worst["top"], max(abs(x[1] - y[1]) for x, y in zip(pa["top"], pb["top"])))
            worst["probe"] = max(worst["probe"], max(abs(x - y) for x, y in zip(pa["probe"], pb["probe"])))
            worst["lse"] = max(worst["lse"], abs(pa["lse"] - pb["lse"]))
        la = np.frombuffer(base64.b64decode(a["last_logits_f32le_b64"]), "<f4")
        lb = np.frombuffer(base64.b64decode(b["last_logits_f32le_b64"]), "<f4")
        worst["last"] = max(worst["last"], float(np.abs(la - lb).max()))
        for ga, gb in zip(a["greedy"], b["greedy"]):
            worst["greedy"] = max(worst["greedy"], max(abs(x[1] - y[1]) for x, y in zip(ga["top"], gb["top"])))
    print("largest absolute logit difference against the committed fixture:")
    for k, v in worst.items():
        print(f"  {k:7s} {v:.3e}")
    print(f"recorded on: {old['hardware']['cpu']} / {old['hardware']['torch_cpu_capability']}")
    print(f"checked on:  {new['hardware']['cpu']} / {new['hardware']['torch_cpu_capability']}")
    if max(worst.values()) > LOGIT_ATOL:
        print(f"the reference itself moved by more than the tolerance ({LOGIT_ATOL}) on this machine")
        failures += 1
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("ANCHOR_MODEL_DIR", str(ROOT / "models" / "smollm2-135m")))
    ap.add_argument("--out", default=str(REF / "fixture.json"))
    ap.add_argument("--check", action="store_true", help="record into memory and compare with the committed fixture")
    ap.add_argument("--allow-other-versions", action="store_true", help="run on versions other than the pinned ones (--check only)")
    args = ap.parse_args()

    inputs = json.loads((REF / "prompts.json").read_text(encoding="utf-8"))
    if args.allow_other_versions and not args.check:
        sys.exit("--allow-other-versions is for --check only; a fixture is recorded with the pinned versions")
    fixture = record(Path(args.model_dir), inputs, args.allow_other_versions)

    if args.check:
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        return compare(old, fixture)

    text = json.dumps(fixture, ensure_ascii=True, indent=1) + "\n"
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
