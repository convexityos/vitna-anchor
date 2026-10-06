"""Gate A11's answer key for the model's own tool format: conversations
rendered through Qwen3-30B-A3B's chat template by transformers.

The engine writes a conversation, with tools, tool calls and tool results,
into a prompt itself, in C; the model was trained on its chat template's
rendering, so the engine must write the same text. This renders each case of
reference/qwen3-30b-a3b/chat-cases.json with the template in the pinned
tokenizer_config.json, through transformers' own apply_chat_template, and
writes reference/qwen3-30b-a3b/chat-template.json: every case with the prompt
it renders to. --check renders again and compares.

    python reference/record_chat_template.py
    python reference/record_chat_template.py --check
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "qwen3-30b-a3b"

# The rendering is transformers' and Jinja2's: a different version may render
# differently, so recording refuses to run on any other.
PINNED = {"transformers": "5.17.0", "jinja2": "3.1.6"}


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_versions() -> dict:
    import jinja2
    import transformers

    found = {"transformers": transformers.__version__, "jinja2": jinja2.__version__}
    wrong = {k: v for k, v in found.items() if v != PINNED[k]}
    if wrong:
        sys.exit(f"these versions differ from the pin: {wrong}; pinned {PINNED}")
    found["python"] = platform.python_version()
    return found


def render(config_path: Path, cases: list) -> list:
    from transformers.utils.chat_template_utils import render_jinja_template

    template = json.loads(config_path.read_text(encoding="utf-8"))["chat_template"]
    out = []
    for case in cases:
        kwargs = {}
        if "enable_thinking" in case:
            kwargs["enable_thinking"] = case["enable_thinking"]
        rendered, _ = render_jinja_template(
            conversations=[case["messages"]],
            tools=case.get("tools"),
            chat_template=template,
            add_generation_prompt=case.get("add_generation_prompt", True),
            **kwargs,
        )
        # The line the engine is given, in Python's own JSON: a JavaScript
        # JSON.stringify would write 100.0 as 100, which the engine, like
        # Python, keeps apart.
        request = {k: case[k] for k in ("messages", "tools", "enable_thinking", "add_generation_prompt") if k in case}
        out.append({**case, "request": json.dumps(request, ensure_ascii=False), "prompt": rendered[0]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3-30b-a3b"))
    ap.add_argument("--check", action="store_true", help="render again and compare with the committed fixture")
    args = ap.parse_args()

    pin = json.loads((REF / "model.json").read_text(encoding="utf-8"))
    entry = next(f for f in pin["files"] if f["path"] == "tokenizer_config.json")
    config_path = Path(args.model) / "tokenizer_config.json"
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        sys.exit(f"{config_path}: sha256 {digest}, pinned {entry['sha256']}")
    software = check_versions()
    cases_path = REF / "chat-cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    fixture = {
        "format": 1,
        "model": {"repo": pin["repo"], "revision": pin["revision"], "file": "tokenizer_config.json", "sha256": entry["sha256"]},
        "inputs": {"file": "reference/qwen3-30b-a3b/chat-cases.json", "sha256_lf": sha256_text(cases_path)},
        "recorder": {"script": "reference/record_chat_template.py", "sha256_lf": sha256_text(Path(__file__))},
        "software": software,
        "renderer": "transformers.utils.chat_template_utils.render_jinja_template",
        "cases": render(config_path, cases),
    }
    out = REF / "chat-template.json"
    if args.check:
        committed = json.loads(out.read_text(encoding="utf-8"))
        bad = [c["id"] for c, d in zip(fixture["cases"], committed["cases"]) if c["prompt"] != d["prompt"]]
        if len(committed["cases"]) != len(fixture["cases"]) or bad:
            print(f"differs: {bad or 'the cases'}")
            return 1
        print(f"all {len(fixture['cases'])} prompts render as committed")
        return 0
    out.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(fixture['cases'])} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
