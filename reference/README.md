# The reference (gates A1, A5, A7 and A14)

Gate A2 asks whether the engine computes what the model computes. This directory is the answer key. It pins one model, holds token ids and logits recorded from a pinned reference implementation, and says how close the engine has to come.

Gate A5 needs a second answer key, for a mixture of experts. It is [below](#a-mixture-of-experts-gate-a5), and follows A1's in everything but what a mixture of experts adds.

Gate A14 holds an embedding model to a third answer key, [at the end](#an-embedding-model-gate-a14).

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

## Quantized weights (gate A7)

Gate A7 asks whether the engine computes, from a GGUF file whose weights are quantized, what the model computes with those weights. A quantized file's weights are small integers times scales, so this answer key is gate A5's recording again, with every weight of the model replaced by the file's, widened to float32 by gguf-py, which is llama.cpp's own reference implementation of the block formats.

### The files

[`olmoe-1b-7b-gguf/model.json`](olmoe-1b-7b-gguf/model.json) pins two of the GGUF files OLMoE's authors publish, `allenai/OLMoE-1B-7B-0924-GGUF` at revision `70df85ed7132bf21b5acbcb33817f58e7d3cb949`, by size and SHA-256:

| File | Size | Its matrices |
|---|---|---|
| `olmoe-1b-7b-0924-q8_0.gguf` | 7.4 GB | 114 in Q8_0, 8 bits |
| `olmoe-1b-7b-0924-q4_k_m.gguf` | 4.2 GB | 97 in Q4_K, 4 bits, and 17 in Q6_K, 6 bits |

In both, the router and the norms, 81 tensors, are float32. The tensors are written unpermuted, and each layer's experts are stacked in the checkpoint's order: Q8_0's widened query, key and expert rows lie within its rounding (0.4% of the largest weight) of the BF16 checkpoint's, where a permutation would move them by the whole weight. `config.json` and the tokenizer still come from A5's pin.

```bash
node scripts/fetch-model.mjs olmoe-1b-7b-gguf    # both files, 11.6 GB, each checked against the pin
```

### The recording

[`record_gguf.py`](record_gguf.py) runs [`record_moe.py`](record_moe.py) unchanged, whose hash A5's fixture pins, after replacing its weight check with the replacement: every GGUF tensor maps to a place in the model, every parameter is written exactly once, and every value written reads back equal. The inputs are A5's, `olmoe-1b-7b/prompts.json`, and so are the settings and the software, with gguf pinned beside them:

| | Version |
|---|---|
| gguf | 0.19.0 |

Each fixture, `olmoe-1b-7b-gguf/fixture-q8_0.json` and `fixture-q4_k_m.json` (3.3 MB each), holds what A5's does except the tokenizer corpus, which is A5's, and adds two things:

- `weights.sha256_f32`: the SHA-256 of every tensor widened to float32, little-endian and row-major in the checkpoint's shape, each layer's experts together under the name of their stack. The engine's `weights-sha256` command prints the same digests for its own widening, and the test requires them equal, all 195. So "the engine widens to the reference's bits" is checked, not argued.
- `against_bf16`: what quantizing cost, measured against A5's fixture of the BF16 model (below). It is measured, not gated.

```bash
python reference/record_gguf.py --file olmoe-1b-7b-0924-q8_0.gguf            # write the fixture
python reference/record_gguf.py --file olmoe-1b-7b-0924-q8_0.gguf --check    # record into memory and compare
```

Both fixtures were recorded on 2026-10-05 on an AMD Ryzen 7 3700X (Windows 11, x86-64) with Python 3.12.10: in 393 s and 560 s.

### The tolerance

A5's, unchanged: A1's table and the routing rule, applied to the logits and router logits the reference computes from the widened weights. The widening itself is held to more than the tolerance: bit for bit, by digest.

### What quantizing cost

Against A5's fixture of the BF16 model, on the same prompts. The KL divergence is from the BF16 model's distribution to the quantized one's, at each prompt's last position, where both fixtures keep all 50,304 logits. The top token is compared at all 242 prompt positions. Greedy decoding is compared token by token from each prompt, up to the first token that differs.

| File | KL at the last position, mean of 6 | Top token agrees | Greedy tokens equal before the first difference |
|---|---|---|---|
| Q8_0 | 6.8e-4 | 240 of 242 | 177 of 192: one prompt departs at step 17, five never |
| Q4_K_M | 1.9e-2 | 214 of 242 | 78 of 192: one prompt departs at its first token, one never |

`python reference/record_gguf.py --file <file>` prints these figures, and each fixture keeps them under `against_bf16`.

### The engine against it

[`tests/reference-gguf.test.mjs`](../tests/reference-gguf.test.mjs) checks that each fixture matches its pins, its inputs and its recorders, and then runs the engine with `--weights <file.gguf>` on the CPU and compares it with each, as A5's tests compare the BF16 model:

- Every tensor widened to the reference's bits, by digest.
- Its logits with every position's experts pinned to the fixture's, under A1's table, and its own router under the routing rule.
- Greedy decoding with the experts pinned, token for token.
- Greedy decoding unpinned, token for token, up to the first decision where its router goes another way than the reference's, which must be one the rule allows.
- The experts read from the drive into a cache of 128 MiB, giving the mapped run's logits and tokens byte for byte.

With `VITNA_DEVICE=cuda` the same tests run on the GPU, where the digests are of the GPU's own widening, read every way its kernels read weights. Two more run only there, as A5's do: an expert cache of any size on the device, fed from the mapped file or a cache in memory, gives the same logits byte for byte; and tokens run together as rows, a prompt's and drafted ones, give a token at a time's logits byte for byte.

## An embedding model (gate A14)

Gate A14 has the engine produce text embeddings, which a search index stores and compares, and serve them at `/v1/embeddings`. It starts as A1 did: one model pinned, and a recording to hold the engine to.

### The model

[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), MIT, pinned in [`bge-small-en-v1.5/model.json`](bge-small-en-v1.5/model.json) at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, with the size and SHA-256 of each of its ten files. It is BERT: 12 layers 384 wide with 12 heads, absolute position embeddings for 512 positions, LayerNorm after each residual add (epsilon 1e-12), and an MLP 1,536 wide with GELU in its erf form. Its tokenizer is WordPiece over 30,522 entries, lowercasing and stripping accents. Its sentence-transformers files say how a text becomes one vector: the last hidden state at the `[CLS]` position, divided by its L2 norm. The weights are 133 MB in float32, small enough to embed a search index on a server's CPU, which is why it was chosen over larger models that rank higher. Its card asks for a query to start with `Represent this sentence for searching relevant passages: `, and for documents to go as they are.

```bash
node scripts/fetch-model.mjs bge-small-en-v1.5           # into models/bge-small-en-v1.5, each file checked against the pin
```

### The recording

[`record_embed.py`](record_embed.py) imports `record.py`'s helpers and version pin (`requirements.txt`), runs the model through transformers on the CPU in float32 with eager attention on one thread, and writes [`bge-small-en-v1.5/fixture.json`](bge-small-en-v1.5/fixture.json) (343 KB) from [`bge-small-en-v1.5/inputs.json`](bge-small-en-v1.5/inputs.json):

- The tokenizer's ids for 23 inputs and 38 corpus strings, from transformers and from the tokenizers library running `tokenizer.json`, which agree on every one: queries with the instruction and passages, a 509-token passage, an empty string, accents and case, CJK, kana and Hangul, Greek and Cyrillic in capitals, emoji with joiners, control and invisible characters, private-use and unassigned code points, words at and past the 100-character limit, and the special tokens written in the text.
- For every input, from one forward pass: the embedding as the model's modules compute it, the norm of the `[CLS]` state before it is divided by it, and the mean of the last hidden state over every position, normalized, which is the pooling other BERT-family models use, so the engine's mean pooling is checked against the same pass without pinning a second model.

The committed fixture was recorded on 2026-10-06 on an AMD Ryzen 7 3700X (Windows 11, x86-64) with Python 3.12.10, in about 20 s. `--check` reproduces it there bit for bit, and the [Reference workflow](../.github/workflows/reference.yml) runs it on GitHub's x86-64 Linux and Windows runners.

```bash
python reference/record_embed.py           # write the fixture
python reference/record_embed.py --check   # record into memory and compare with the committed fixture
python reference/embed_sensitivity.py      # what plausible defects do, against the tolerance
```

### The tokenizer's Unicode

Comparing the engine's tokenizer with the tokenizers library over every code point found that the library's BERT steps do not take their character data from one Unicode version. They come from three Rust crates: the categories (which characters are other, nonspacing marks, or punctuation) from `unicode_categories`, Unicode 8.0; NFD from `unicode-normalization-alignments`, Unicode 9.0; and lowercase from Rust's standard library, Unicode 17.0 for the build that `tokenizers` 0.23.2 ships. Python 3.12's tables, Unicode 15.0, disagree with the library on 624 of the code points between two letters: marks, format characters and punctuation added since 8.0 that the library neither drops nor splits off, U+11938, which its NFD does not decompose, and the characters given lowercase forms in Unicode 16 and 17. Combining classes from Unicode 10.0 or later also reorder marks the library leaves in place. The engine follows the library, since the model was trained on what the library produces, and [`gen_wordpiece_unicode.py`](../engine/tools/gen_wordpiece_unicode.py) builds each table from the data file of its own version, downloaded from unicode.org and checked by SHA-256. Two more of the library's choices show the same way: it keeps unassigned code points, which then become the unknown token, and it begins CJK Extension E at U+2B920, where the Python BERT tokenizer begins it at U+2B820. With these, the engine and the library give the same ids for all 1,112,064 code points between two letters and 200,000 random strings ([`check_wordpiece.py`](../engine/tools/check_wordpiece.py)).

### The tolerance

Written in [`compare.mjs`](compare.mjs), which the tests use:

| What | Must hold |
|---|---|
| Token ids | Exactly equal to the model's own `tokenizer.json`'s, as the tokenizers library runs it, for every input and corpus string |
| Embeddings | Within 1e-5 absolute, every value of every input's embedding, both as the model pools it and by the mean, each normalized |
| The `[CLS]` state's norm | Within 1e-5 of the reference's, relatively |

Why 1e-5: arithmetic in float64 moves the reference's embeddings by at most 2.5e-7, which is the scale of float32 rounding, and six defects an implementation could make and still produce plausible vectors move some value by far more:

| Defect | Largest difference |
|---|---|
| GELU in its tanh form | 7.0e-4 |
| LayerNorm epsilon 1e-5 rather than 1e-12 | 1.1e-4 |
| The token-type embedding left out | 8.6e-2 |
| Attention scores not scaled | 0.22 |
| Positions counted from 1 | 4.8e-2 |
| Mean pooling without `[CLS]` and `[SEP]` | 0.21 |

The line was first written at 1e-4, in `record_embed.py` before the engine could embed a text. It was moved to 1e-5 after the engine's first comparison, a largest difference of 3.1e-7 that passes either way, because the measurement above showed that 1e-4 caught the wrong epsilon by only 13%. At 1e-5 that defect is eleven times the line, and rounding a fortieth of it.

### The engine against it

`tests/reference-embed.test.mjs` checks that the fixture matches its pin, inputs and recorders, and that the comparison accepts the reference's own embeddings and rejects wrong ones; then, with the model and a built engine, that the engine's tokenizer gives every id exactly, that its embeddings are within the tolerance (largest difference 3.05e-7 pooled at `[CLS]` and 1.52e-7 by the mean, on the machine that recorded the fixture), that it gives the same bits on 1 and 3 threads and with the texts sent alone or in another order, and that a text longer than 512 tokens is refused. `tests/embeddings-serve.test.mjs` checks the same embeddings through `/v1/embeddings`.
