"""Check the engine's NFC against this Python's unicodedata.

    python engine/tools/check_nfc.py <engine> <model dir> [--random N] [--seed S]

The model directory must hold a tokenizer.json whose normalizer is NFC, such
as models/olmoe-1b-7b. The engine's normalize command runs on:

- every code point alone, but the surrogates;
- N random strings (default 200,000) of 1 to 12 characters, drawn mostly
  from the characters normalization touches: combining marks of every class,
  characters with a canonical decomposition and the parts they decompose to,
  primary composites, Hangul jamo and syllables, and a few plain letters.

Every result must equal unicodedata.normalize("NFC", s). The engine's tables
come from engine/tools/gen_unicode.py with this same unicodedata, so this
checks the algorithm, not the Unicode version. Prints the first mismatches
and exits 1 if there are any.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import unicodedata


def interesting():
    """Characters normalization does something with, and some it leaves alone."""
    pool = set()
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if unicodedata.combining(c):
            pool.add(cp)
        d = unicodedata.decomposition(c)
        if d and not d.startswith("<"):
            pool.add(cp)
            pool.update(int(x, 16) for x in d.split())
    pool.update(range(0x1100, 0x1113))  # Hangul leading consonants
    pool.update(range(0x1161, 0x1176))  # vowels
    pool.update(range(0x11A7, 0x11C3))  # trailing consonants, and one before them
    pool.update(range(0xAC00, 0xAC00 + 400, 7))  # syllables, LV and LVT
    pool.update(map(ord, "aeiouAEIOU kx"))
    return sorted(pool)


def run(engine, model, texts):
    lines = "\n".join(json.dumps(t, ensure_ascii=True) for t in texts) + "\n"
    r = subprocess.run([engine, "normalize", "--model", model], input=lines.encode("ascii"), capture_output=True)
    if r.returncode != 0:
        sys.exit(f"normalize exited {r.returncode}: {r.stderr.decode(errors='replace')}")
    # Split on "\n" alone: splitlines() would also split inside a string the
    # engine wrote with U+2028 or U+0085 in it, which JSON does not escape.
    out = [json.loads(line) for line in r.stdout.decode("utf-8").split("\n") if line]
    if len(out) != len(texts):
        sys.exit(f"{len(texts)} texts in, {len(out)} lines out")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--random", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261001)
    args = ap.parse_args()
    args.engine = os.path.abspath(args.engine)

    singles =[chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    rng = random.Random(args.seed)
    pool = interesting()
    mixed = ["".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 12))) for _ in range(args.random)]

    failures = 0
    for name, texts in (("single code points", singles), ("random strings", mixed)):
        got = run(args.engine, args.model, texts)
        bad = [(t, g) for t, g in zip(texts, got) if g != unicodedata.normalize("NFC", t)]
        print(f"{name}: {len(texts) - len(bad)} of {len(texts)} equal unicodedata's NFC (Unicode {unicodedata.unidata_version})")
        for t, g in bad[:10]:
            want = unicodedata.normalize("NFC", t)
            hexes = lambda s: " ".join(f"{ord(c):04X}" for c in s)
            print(f"  {hexes(t)}: engine {hexes(g)}, unicodedata {hexes(want)}")
        failures += len(bad)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
