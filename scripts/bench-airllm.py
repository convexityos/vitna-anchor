"""Time AirLLM on the token ids the engine was given, for the README's "Speed beside AirLLM".

    python scripts/bench-airllm.py --model <dir> --shards <dir> --prompt <text> --ids <a,b,...>
                                   [--max-new <n>] [--compression 4bit|8bit]

Needs AirLLM and a CUDA build of PyTorch, which this repository does not
otherwise use; it was measured with airllm 4.0.0, torch 2.11.0+cu128 and
transformers 4.57.6. --shards is where AirLLM writes the copy of the
checkpoint it splits into a file a layer on first load (as large as the
checkpoint, or smaller with --compression). --ids are the prompt's ids as the
engine's `generate` printed them, so both read the same tokens; the ids
AirLLM's own tokenizer makes of --prompt are printed beside them, and whether
they are equal.

Prints one JSON object: the greedy ids generated and their text, the prompt's
time (from generate() to the first new token), each later token's time, their
mean and median, and the peak memory PyTorch's allocator held on the device.
The device's whole use, the CUDA runtime included, is what nvidia-smi
reports, which the README samples beside it.
"""

import argparse
import json
import statistics
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--shards", required=True, help="where AirLLM writes its per-layer copy (layer_shards_saving_path)")
p.add_argument("--prompt", required=True)
p.add_argument("--ids", required=True, help="the prompt's ids as the engine tokenized them, comma-separated")
p.add_argument("--max-new", type=int, default=16)
p.add_argument("--compression", default=None)
a = p.parse_args()

from airllm import AutoModel  # noqa: E402  (after argparse, so --help needs no AirLLM)

t0 = time.perf_counter()
model = AutoModel.from_pretrained(a.model, layer_shards_saving_path=a.shards, compression=a.compression)
load_s = time.perf_counter() - t0

hf_ids = model.tokenizer(a.prompt)["input_ids"]
ids = [int(x) for x in a.ids.split(",")]


class Clock:
    """A streamer: generate() hands it the prompt first, then each new token."""

    def __init__(self):
        self.stamps = []
        self.seen_prompt = False

    def put(self, value):
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())

    def end(self):
        pass


clock = Clock()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
out = model.generate(
    torch.tensor([ids], device="cuda"),
    max_new_tokens=a.max_new,
    do_sample=False,
    use_cache=True,
    streamer=clock,
    return_dict_in_generate=True,
)
new_ids = out.sequences[0].tolist()[len(ids):]
steps = [(b - c) * 1000 for c, b in zip(clock.stamps, clock.stamps[1:])]
json.dump(
    {
        "load_s": round(load_s, 1),
        "airllm_prompt_ids": hf_ids,
        "prompt_ids_equal": hf_ids == ids,
        "new_ids": new_ids,
        "text": model.tokenizer.decode(new_ids),
        "prompt_ms": round((clock.stamps[0] - start) * 1000, 1),
        "step_ms_mean": round(statistics.mean(steps), 1) if steps else None,
        "step_ms_median": round(statistics.median(steps), 1) if steps else None,
        "step_ms": [round(s, 1) for s in steps],
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "torch": torch.__version__,
        "dtype": str(model.running_dtype),
    },
    sys.stdout,
)
print()
