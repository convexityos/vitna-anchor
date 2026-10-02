# The reference (gates A1 and A5)

Gate A2 asks whether the engine computes what the model computes. This directory is the answer key. It pins one model, holds token ids and logits recorded from a pinned reference implementation, and says how close the engine has to come.

Gate A5 needs a second answer key, for a mixture of experts. It is [below](#a-mixture-of-experts-gate-a5), and follows A1's in everything but what a mixture of experts adds.

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

## A mixture of experts (gate A5)

Gate A5 streams a mixture-of-experts checkpoint from a drive. The engine has to compute such a model correctly before it is worth making it fast, so A5 starts as A1 did: one model pinned, and a recording to hold the engine to. Step 2 runs it in the engine, on the CPU, and holds it to this recording ([below](#the-engine-against-it)).

### The model

[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924), Apache-2.0, pinned in [`olmoe-1b-7b/model.json`](olmoe-1b-7b/model.json) at revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, with the size and SHA-256 of each of its nine files. It has 6.9B parameters in 16 layers. Each layer has 64 experts, SwiGLU MLPs 1,024 wide, and a router that sends each token through 8 of them, so about 1.3B parameters act on any one token and 93% of the weights are in the experts. The rest is close to SmolLM2's Llama architecture, with four differences the engine will need:

- 16 key-value heads for 16 query heads: no grouping.
- RMSNorm over the query and key projections, each across all 2,048 values before they are split into heads, with weights of its own.
- The router: a softmax over the 64 experts' logits, whose 8 largest weights are used as they are, not renormalized to sum to 1.
- An output layer of its own, not tied to the embeddings, with 50,304 rows for the tokenizer's 50,280 ids.

The tokenizer is GPT-NeoX's byte-level BPE, after NFC normalization, with added tokens matched before anything else: runs of 2 to 24 spaces, three placeholders (`|||IP_ADDRESS|||`, `|||EMAIL_ADDRESS|||`, `|||PHONE_NUMBER|||`), `<|padding|>` and `<|endoftext|>`.

The weights are 13.8 GB in bfloat16, in three files. That is more than the 8 GB GPU that passed A4 holds, which is why A5 streams them. They are downloaded, never committed:

```bash
node scripts/fetch-model.mjs olmoe-1b-7b           # into models/olmoe-1b-7b, each file checked against the pin
node scripts/fetch-model.mjs olmoe-1b-7b --check   # check what is there, download nothing
```

### The recording

[`record_moe.py`](record_moe.py) records it as `record.py` records SmolLM2, with the same pinned software, the same settings and the same fields, into [`olmoe-1b-7b/fixture.json`](olmoe-1b-7b/fixture.json) (3.3 MB). The inputs, in [`olmoe-1b-7b/prompts.json`](olmoe-1b-7b/prompts.json), are A1's 6 prompts and 42 corpus strings, and 12 more strings for this tokenizer's added tokens and NFC. For this model transformers runs the model's own `tokenizer.json`, so the two sets of ids agree on every string and `tokenizer_disagreements` is empty. Three things are new:

- **The routing.** For every layer of every prompt position, and of every token greedy decoding feeds back: the 8 experts the router chose, in its order, then the one that came closest; the router logits of those 9; and the logsumexp of all 64, which gives each chosen expert's weight as exp(logit - logsumexp).
- **The experts' implementation.** They run as transformers' eager implementation, one expert at a time, selected explicitly as eager attention is. Left to itself, transformers 5.17 picks a grouped matrix multiply where torch provides one.
- **Two checks on loading.** transformers fills any weight it cannot find with random numbers and only warns, so recording stops unless its loading report is empty. It also stacks each layer's experts into two tensors as it loads, and that report leaves conversion errors out, so all 3,219 tensors in the checkpoint are then compared with the model's copies, each expert's three matrices found in the stacks.

`record_moe.py` imports its helpers and its pinned versions from `record.py` rather than changing it: A1's fixture pins `record.py`'s hash. This fixture pins the hashes of both.

The committed fixture was recorded on 2026-10-01 on an AMD Ryzen 7 3700X (Windows 11, x86-64) with Python 3.12.10. Recording took 386 to 590 s there in three runs, on one thread. The model takes 27.7 GB in float32, and the recorder's memory peaked near 36 GB while loading it. The [Reference workflow](../.github/workflows/reference.yml) does not record this one again, since GitHub's standard runners have far less memory than that. `--check` is run by hand:

```bash
python reference/record_moe.py           # write the fixture
python reference/record_moe.py --check   # record into memory and compare with the committed fixture
```

On the machine that recorded it, `--check` reproduces it bit for bit, routing included.

### Routing near-ties

A token goes through the 8 experts its router scores highest, so two scores that nearly tie can trade places when the engine adds the same sums in another order. The fixture holds 6,848 routing decisions. In 85 of them the 8th and 9th router logits are within 1e-3 of each other, in 9 within 1e-4, and in 2 within 1e-5: 5.45e-6 (prompt `unicode`, greedy token 26, layer 2) and 6.62e-6 (`code`, greedy token 10, layer 5).

Such a decision going the other way is no rounding error downstream. Each prompt's closest call at a prompt position (margins from 8.8e-5 to 6.9e-4) was forced the other way. With every other decision held to the reference's, the logits after it moved by 0.011 to 0.21. With the others left to the router, which in four of the six prompts then decided some of them differently as well, they moved by 0.019 to 0.39. Either way, past the 1e-2 tolerance every time. So the engine's routing is compared on its own, against a band of its own (below), and step 2 compares the engine's logits with its routing pinned to the fixture's (`--experts-in`), so that a near-tie it decides the other way cannot fail the comparison of everything after it.

### The tolerance

A1's table applies to this fixture unchanged, at this model's 50,304 logits a row. [`compare.mjs`](compare.mjs) adds one rule, set here before the engine could route a single token:

| What | Must hold |
|---|---|
| Routing | At every layer of every position, prompt and greedy alike: the router logits of the 9 experts the fixture keeps are within 1e-3 of the reference's, and the token goes through the reference's 8 experts, except that one whose router logit the reference puts within 2e-3 of the runner-up's may give way to the runner-up |

Why 1e-3: the reference was run in two other valid float32 orders, once with the greedy tokens batched into one pass rather than fed one at a time, and once more like that on eight threads with PyTorch's SDPA attention. Its router logits moved by at most 1.05e-5 and its logits by at most 2.96e-5, and not one routing decision changed, the 5.45e-6 near-tie included. So the band is about a hundred times what reordering does, and a tenth of the logits' 1e-2 for router logits that are small: those the fixture keeps have a median magnitude of 0.42, and none exceeds 5.7. A band of 2e-2 would leave 1,416 of the 6,848 decisions free to go either way; 2e-3 leaves 159.

Both measurements, the forced calls both ways and the two orders, come from one command, run on the machine that recorded the fixture:

```bash
python reference/routing_sensitivity.py
```

`tests/reference-moe.test.mjs` checks that this fixture matches its pin, inputs and both recorders, that its routing is complete and consistent, that the comparisons accept the reference's own logits and routing and reject wrong ones, and that the runner-up may stand in at the fixture's closest call, for one expert only, and not at a wide one.

### The engine against it

Step 2 runs OLMoE in the engine, on the CPU, and the same file compares it with this fixture:

- Its tokenizer's ids, for all 60 strings, exactly.
- Its logits, with every position's experts pinned to the fixture's (`--experts-in`), under A1's table; and its own router, which still runs and says what it chose (`--router-out`, `--experts-out`), under the routing rule.
- Greedy decoding with the experts pinned, prompts and greedy tokens alike, token for token.
- That a pin is not ignored: pinning the runner-up in place of the 8th expert at the shortest prompt's closest call moves the logits after it past the tolerance, and those before it not at all.
- Greedy decoding unpinned, token for token, up to the first decision where the engine's router goes another way than the reference's, which must be one the routing rule allows.

How it was checked, and what it found, is in the [top-level README](../README.md#how-gate-a5s-forward-pass-was-checked).
