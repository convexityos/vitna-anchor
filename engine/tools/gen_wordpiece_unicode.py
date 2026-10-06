"""Generate engine/src/wordpiece_data.h: the Unicode data a BERT tokenizer needs.

    python engine/tools/gen_wordpiece_unicode.py > engine/src/wordpiece_data.h

A BERT model's tokenizer.json names BertNormalizer and BertPreTokenizer, and
the tokenizers library, which defines what they do, builds them from three
Rust crates that carry Unicode data of three different ages:

- unicode_categories, Unicode 8.0: which characters are "other" (Cc, Cf, Co
  and Cs; clean_text drops them), nonspacing marks (Mn; strip_accents drops
  them after NFD) and punctuation (P*; the pre-tokenizer splits each off);
- unicode-normalization-alignments, Unicode 9.0: the canonical decompositions
  and combining classes of the NFD that strip_accents runs;
- Rust's standard library, Unicode 17.0: char::to_lowercase, the lowercase
  step, with SpecialCasing's unconditional mappings.

Those versions were found, not assumed: engine/tools/check_wordpiece.py runs
the tokenizers library over every code point and random strings, and a table
from any other version disagrees with it (Unicode 15.0's categories on about
8,600 code points, its NFD on U+11938, its lowercase on the 29 characters
Unicode 16 and 17 added cases for). So this header takes each table from the
data file of its own version, downloaded from unicode.org and checked against
the SHA-256 below, and not from this Python's unicodedata.

Unassigned code points are in none of the classes, as in those crates: the
library keeps them, and a word holding one becomes the unknown token.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "models" / "ucd"

FILES = {
    "8.0.0/UnicodeData.txt": "38b17e1118206489a7e0ab5d29d7932212d38838df7d3ec025ecb58e8798ec20",
    "9.0.0/UnicodeData.txt": "68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f",
    "17.0.0/UnicodeData.txt": "2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c",
    "17.0.0/SpecialCasing.txt": "efc25faf19de21b92c1194c111c932e03d2a5eaf18194e33f1156e96de4c9588",
}


def fetch(name: str) -> str:
    path = CACHE / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://www.unicode.org/Public/{name.split('/')[0]}/ucd/{name.split('/')[1]}"
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FILES[name]:
        sys.exit(f"{path}: sha256 {digest}, pinned {FILES[name]}")
    return data.decode("utf-8")


def rows(text):
    """(code point, fields) for every assigned code point, ranges expanded."""
    first = None
    for line in text.splitlines():
        f = line.split(";")
        cp, name = int(f[0], 16), f[1]
        if name.endswith(", First>"):
            first = cp
            continue
        for c in range(first, cp + 1) if name.endswith(", Last>") else (cp,):
            yield c, f


def ranges(cps):
    out = []
    for cp in sorted(cps):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def emit_rows(name, rows_, width, comment):
    print(f"/* {comment} */")
    print(f"static const uint32_t {name}[][{width}] = {{")
    line = "   "
    for row in rows_:
        item = " {" + ", ".join(f"0x{v:X}" for v in row) + "},"
        if len(line) + len(item) > 100:
            print(line)
            line = "   "
        line += item
    print(line)
    print("};")
    print(f"#define {name}_COUNT {len(rows_)}")
    print()


def main():
    sys.stdout.reconfigure(newline="\n")  # the header is LF on every system, as .gitattributes keeps it
    cats8 = {cp: f[2] for cp, f in rows(fetch("8.0.0/UnicodeData.txt"))}
    other = [cp for cp, c in cats8.items() if c in ("Cc", "Cf", "Co", "Cs")]
    marks = [cp for cp, c in cats8.items() if c == "Mn"]
    punct = [cp for cp, c in cats8.items() if c.startswith("P")]

    ccc = {}
    decomp = []
    for cp, f in rows(fetch("9.0.0/UnicodeData.txt")):
        if int(f[3]):
            ccc[cp] = int(f[3])
        if f[5] and not f[5].startswith("<") and not (0xAC00 <= cp <= 0xD7A3):
            parts = [int(x, 16) for x in f[5].split()]
            if len(parts) > 2:
                sys.exit(f"U+{cp:04X} decomposes into {len(parts)} characters; this table holds two")
            decomp.append((cp, parts[0], parts[1] if len(parts) == 2 else 0))
    ccc_runs = []
    for cp in sorted(ccc):
        if ccc_runs and ccc_runs[-1][1] == cp - 1 and ccc_runs[-1][2] == ccc[cp]:
            ccc_runs[-1][1] = cp
        else:
            ccc_runs.append([cp, cp, ccc[cp]])

    lower = {cp: [int(f[13], 16)] for cp, f in rows(fetch("17.0.0/UnicodeData.txt")) if f[13]}
    for line in fetch("17.0.0/SpecialCasing.txt").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        f = [x.strip() for x in line.split(";")]
        if len(f) > 4 and f[4]:
            continue  # a conditional mapping, which char::to_lowercase never applies
        cp, low = int(f[0], 16), [int(x, 16) for x in f[1].split()]
        if low != [cp]:
            lower[cp] = low
    for cp, low in lower.items():
        if len(low) > 2:
            sys.exit(f"U+{cp:04X} lowercases to {len(low)} characters; this table holds two")

    print("/* Generated by engine/tools/gen_wordpiece_unicode.py. Do not edit. */")
    print("/* Categories from Unicode 8.0.0, NFD from 9.0.0, lowercase from 17.0.0: see the generator for why. */")
    print()
    print("#ifndef VITNA_WORDPIECE_DATA_H")
    print("#define VITNA_WORDPIECE_DATA_H")
    print()
    print("#include <stdint.h>")
    print()
    emit_rows("VITNA_WP_OTHER", ranges(other), 2, "Cc, Cf, Co and Cs in Unicode 8.0.0: {first, last}.")
    emit_rows("VITNA_WP_MARK", ranges(marks), 2, "Mn in Unicode 8.0.0: {first, last}.")
    emit_rows("VITNA_WP_PUNCT", ranges(punct), 2, "Pc, Pd, Ps, Pe, Pi, Pf and Po in Unicode 8.0.0: {first, last}.")
    emit_rows("VITNA_WP_CCC", ccc_runs, 3, "Canonical combining classes other than 0 in Unicode 9.0.0: {first, last, class}.")
    emit_rows("VITNA_WP_DECOMP", decomp, 3, "Canonical decompositions in Unicode 9.0.0, one level, Hangul aside: {code point, first, second or 0}.")
    emit_rows("VITNA_WP_LOWER", [(cp, *low, 0) if len(low) == 1 else (cp, *low) for cp, low in sorted(lower.items())], 3,
              "Lowercase in Unicode 17.0.0, as char::to_lowercase: {code point, first, second or 0}.")
    print("#endif /* VITNA_WORDPIECE_DATA_H */")


if __name__ == "__main__":
    main()
