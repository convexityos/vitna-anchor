# The reference (gate A1)

Gate A2 asks whether the engine computes what the model computes. This directory is the answer key. It pins one model, holds token ids and logits recorded from a pinned reference implementation, and says how close the engine has to come.

## The model

[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), Apache-2.0, pinned in [`smollm2-135m/model.json`](smollm2-135m/model.json) at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, with the size and SHA-256 of every file used. It is a small dense model of the Llama architecture: RMSNorm, rotary position embeddings in the half-split form, grouped-query attention (9 query heads over 3 key-value heads), a SwiGLU MLP, and input embeddings tied to the output layer. The weights are 269 MB in bfloat16. Qwen2.5-0.5B was the alternative, at nearly four times the size, with biases on its attention projections that the engine does not need yet.

The weights are downloaded, never committed:

```bash
node scripts/fetch-model.mjs           # into models/smollm2-135m, each file checked against the pin
node scripts/fetch-model.mjs --check   # check what is there, download nothing
```

## The recording

[`record.py`](record.py) runs the model through Hugging Face transformers on the CPU and writes [`smollm2-135m/fixture.json`](smollm2-135m/fixture.json) (about 2.2 MB) from the inputs in [`smollm2-135m/prompts.json`](smollm2-135m/prompts.json):

- The tokenizer's ids for 6 prompts and 42 corpus strings: whitespace runs, contractions, digits in several scripts, emoji with joiners and tag characters, CJK, right-to-left text, combining accents, special tokens, and control characters the vocabulary cannot encode (the tokenizer drops them).
- For every position of every prompt: the 16 largest logits with their ids, the logits of 64 fixed probe ids spread over the vocabulary, the logsumexp, mean and standard deviation of the whole row, and the margin between the first and second logits.
- All 49,152 logits at each prompt's last position.
- 32 steps of greedy decoding from each prompt, one token at a time through the key-value cache, with the top 8 logits and the margin at each step. transformers' own `generate()` must choose the same tokens, or recording stops.

Token ids are recorded twice. The first set comes from transformers. The second comes from the tokenizers library running the model's own `tokenizer.json`, which is the model's definition of its tokenizer. They are not always the same. transformers 5.17 builds this model's tokenizer from its `GPT2Tokenizer` class, whose pre-tokenizer is `ByteLevel` alone, and drops the `Digits` step the file puts before it. Digits in ASCII tokenize the same either way, because the vocabulary has no multi-digit tokens. Other numerals do not. For `"٣٤٥ ⅫⅣ ½ ²"`, transformers keeps the space before `½` with the numeral, and the file splits it off. That is the one corpus string where the two disagree, and the fixture names it in `tokenizer_disagreements`. The engine follows the file. The model runs on the prompts' ids, and the recorder refuses any prompt where the two tokenizers disagree.

Settings: float32 weights and arithmetic (the bfloat16 weights widen exactly), eager attention, one thread, deterministic algorithms, ties broken towards the lower token id. The software is pinned in [`requirements.txt`](requirements.txt), and `record.py` refuses to record with any other version:

| | Version |
|---|---|
| torch | 2.14.0+cpu |
| transformers | 5.17.0 |
| tokenizers | 0.23.2 |
| safetensors | 0.8.0 |
| numpy | 2.5.3 |

The committed fixture was recorded on 2026-09-29 on a Snapdragon X Elite X1E80100 (Windows 11, ARM64) with Python 3.12.10. Recording took 78 s there.

```bash
pip install -r reference/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python reference/record.py           # write the fixture
python reference/record.py --check   # record into memory and compare with the committed fixture
```

`--check` requires identical token ids, and fails if any logit moved by more than the tolerance below. On the machine that recorded the fixture it reproduces bit for bit. The [Reference workflow](../.github/workflows/reference.yml) runs it on GitHub's x86-64 Linux and Windows runners and prints the largest difference, which is how much torch itself varies between CPUs.

## The tolerance

Set here in A1, before the engine could produce a single logit, and written in [`compare.mjs`](compare.mjs), which the tests use:

| What | Must hold |
|---|---|
| Token ids | Exactly equal to the model's own `tokenizer.json`, as the tokenizers library runs it, for every prompt and corpus string. This was first written as "the reference's ids". The first comparison with the engine found where transformers departs from the file, as described above |
| Logits | Within 1e-2 absolute: the 16 largest and 64 probe logits at every position, and all 49,152 at each last position |
| logsumexp of each row | Within 1e-2 |
| Argmax at each position | The reference's, unless the reference's top two are within 2e-2 of each other, where either is accepted |
| Greedy decoding | The same ids at all 32 steps from all 6 prompts, and the top 8 logits at each step within 1e-2 |

Why 1e-2: the reference and the engine's float path both compute in float32. Summing in a different order, or rounding exp and sin differently, moves a logit by far less than 1e-2. A real defect, such as the wrong rotary convention, a missing norm, or a position off by one, moves logits by 0.1 or more and usually changes the argmax too. The smallest top-1 margin in any greedy step is 0.0085 (prompt `numbers`, step 17), so a pass there means the engine and the reference agree on a close call.

`tests/reference.test.mjs` checks that the fixture matches its pin, inputs and recorder, that the comparison accepts the reference's own logits and rejects wrong ones, and then compares the engine. The engine comparisons are `todo` until A2.
