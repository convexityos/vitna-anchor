"""Check the engine's WordPiece tokenizer against the tokenizers library.

    python engine/tools/check_wordpiece.py <engine> <model dir> [--random N] [--seed S]

The model directory must hold a BERT tokenizer.json, such as
models/bge-small-en-v1.5. The engine's tokenize command runs on:

- every code point but the surrogates, between two letters ("a" + c + "b"),
  so a character the normalizer drops, splits off or changes shows in the ids;
- N random strings (default 200,000) of 1 to 16 characters, drawn from
  combining marks of every class (Unicode 13.0's, newer than the tables the
  library uses), Latin, Greek, Cyrillic, kana, CJK, Hangul, whitespace and
  controls, the characters the library treats unlike Python, the model's
  special tokens and stray brackets, and code points from anywhere.

Every result must equal the ids the tokenizers library gives, running the
model's own tokenizer.json. Unlike engine/tools/check_nfc.py this checks the
Unicode versions as well as the algorithm, since the library's categories,
NFD and lowercase come from three versions (gen_wordpiece_unicode.py). Prints
the first mismatches and exits 1 if there are any.
"""

import argparse
import json
import os
import random
import subprocess
import sys


def run(engine, model, texts):
    lines = "\n".join(json.dumps(t, ensure_ascii=True) for t in texts) + "\n"
    r = subprocess.run([engine, "tokenize", "--model", model], input=lines.encode("ascii"), capture_output=True)
    if r.returncode != 0:
        sys.exit(f"tokenize exited {r.returncode}: {r.stderr.decode(errors='replace')}")
    out = [json.loads(line) for line in r.stdout.decode("utf-8").split("\n") if line.strip()]
    if len(out) != len(texts):
        sys.exit(f"{len(texts)} texts in, {len(out)} lines out")
    return out


def pool():
    marks = list(range(0x300, 0x370)) + list(range(0x483, 0x48A)) + list(range(0x591, 0x5C8)) + list(range(0x610, 0x61B))
    marks += list(range(0x64B, 0x660)) + list(range(0x900, 0x904)) + list(range(0x93A, 0x950)) + [0x0E31, 0x0E34, 0x0E35, 0x0E47, 0x0E48]
    marks += list(range(0x1AB0, 0x1AC1)) + list(range(0x1DC0, 0x1E00)) + list(range(0x20D0, 0x20F1)) + [0x3099, 0x309A, 0x0898, 0x10EFD, 0x1E08F]
    marks += [0x11938, 0x11935, 0x11930, 0x0F71, 0x0F72, 0x0F74, 0x0F80, 0x1D165, 0x1D166, 0x1D16D, 0x1E94A, 0x16FF0]
    letters = list(range(0x20, 0x250)) + list(range(0x370, 0x530)) + list(range(0x1E00, 0x1F00)) + list(range(0x3040, 0x3100))
    letters += list(range(0x4E00, 0x4E40)) + list(range(0xAC00, 0xAC40)) + list(range(0x1100, 0x1200, 3)) + list(range(0x2B810, 0x2B930, 7))
    odd = [0, 0x7, 0xB, 0xC, 0x1C, 0x85, 0xA0, 0xAD, 0x130, 0x131, 0x1E9E, 0x3A3, 0x2028, 0x2029, 0x200B, 0x200D, 0xFEFF, 0xFFFD, 0x3000,
           0x2126, 0x212B, 0xFB01, 0xE000, 0x0378, 0x1FAE8, 0x1FAE9, 0x2E43, 0x166D, 0x061D, 0x1C89, 0xA7CB, 0x10D50, 0x16EA0, 0x24B6]
    return marks, letters, odd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--random", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    args.engine = os.path.abspath(args.engine)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    singles = ["a" + chr(cp) + "b" for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    rng = random.Random(args.seed)
    marks, letters, odd = pool()
    pieces = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]", "[SEP", "SEP]", "[", "]", "##", " ", "  ", "\t", "\n"]
    mixed = []
    for _ in range(args.random):
        parts = []
        for _ in range(rng.randint(1, 16)):
            r = rng.random()
            if r < 0.25:
                parts.append(chr(rng.choice(marks)))
            elif r < 0.75:
                parts.append(chr(rng.choice(letters)))
            elif r < 0.85:
                parts.append(chr(rng.choice(odd)))
            elif r < 0.92:
                parts.append(rng.choice(pieces))
            else:
                cp = rng.randrange(0x110000)
                parts.append(chr(cp) if not 0xD800 <= cp <= 0xDFFF else "x")
        mixed.append("".join(parts))

    failures = 0
    for name, texts in (("every code point between two letters", singles), ("random strings", mixed)):
        got = run(args.engine, args.model, texts)
        want = [e.ids for e in tok.encode_batch(texts)]
        bad = [(t, g, w) for t, g, w in zip(texts, got, want) if g != w]
        print(f"{name}: {len(texts) - len(bad)} of {len(texts)} equal the tokenizers library's ids")
        for t, g, w in bad[:10]:
            print(f"  {' '.join(f'{ord(c):04X}' for c in t)}: engine {g}, library {w}")
        failures += len(bad)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
