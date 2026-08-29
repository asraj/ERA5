# Session 9 — Loss functions and output heads

**Stephen Raj Arokiasamy**

**Notebook:** [`S9_loss_harness.ipynb`](S9_loss_harness.ipynb) — runs top to bottom on a free Colab T4.
**Model:** `HuggingFaceTB/SmolLM2-135M` — 134,515,008 params, V = 49,152, D = 576, 30 layers, GQA
(9 query heads, 3 KV heads, head_dim 64), tied embeddings.
**Data:** the Session-4 cleaned OpenWebText shard — [`data/s9_owt_clean.jsonl.gz`](data/) —
5,009 documents, 2,998,828 whitespace words, run through the S4 pipeline stages
(normalise → format discipline → quality → exact + MinHash dedup → language-ID → PII scrub).

---

## What the harness fixes

The starting point:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss   = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                       tokens[:, 1:].reshape(-1))
```

The shift is right. Everything else it does not do is a silent bug: padding is trained on,
document joins are trained on, and the denominator is `B×T` rather than the number of positions
that actually counted. None of those raise. They just produce a loss that is measuring something
other than what you think.

---

## Part 1 — the seven numbers

Some of these are fixed by the configuration and can be stated exactly. The rest depend on the
pretrained weights and come from the notebook's final summary cell — **run it and paste**.

### 1 · Shapes, and what each dimension is

| Tensor | Shape | What the dimensions are |
|---|---|---|
| `tokens` | `[B, T]` | B = sequences in the batch, T = positions in each |
| `attention_mask` | `[B, T]` | 1 = real token, 0 = padding |
| `hidden` | `[B, T, D]` | D = 576, width of the residual stream — one vector per position |
| `logits` | `[B, T, V]` | V = 49,152, one raw score per vocabulary entry, per position |
| `shift_logits` | `[B, T-1, V]` | drop the **last** position: nothing follows it to predict |
| `shift_targets` | `[B, T-1]` | drop the **first** token: nothing precedes it to predict from |
| `flat_logits` | `[B(T-1), V]` | every prediction in the batch, stacked |
| `flat_targets` | `[B(T-1)]` | one correct token id per prediction |

**The logits tensor is 85.3× larger than the hidden states that produced it** (V/D = 49,152/576).
That ratio is the whole of item 7.

### 2 · The shift, verified on strings

The notebook prints inputs beside targets as decoded strings, then prints a deliberately
**unshifted** control. The control is the point: with `offset=0` the two columns are identical,
which is the model being handed the answer. In a wall of integers that is invisible; in strings it
is obvious at a glance.

### 3 · Padding masked

| | loss | contributing tokens |
|---|---|---|
| padding counted | *[run]* | *[run]* |
| padding masked | *[run]* | *[run]* |

**The trap this notebook documents:** SmolLM2 ships without a pad token, so the usual fix is
`tok.pad_token = tok.eos_token`. That makes `pad_id == eos_id`, and masking with
`ids == pad_token_id` then silently deletes **every genuine end-of-document token** from the loss.
The mask must come from the tokenizer's `attention_mask`, not from an id comparison.

### 4 · Two documents packed, boundary masked

Done twice — the literal case the assignment asks for, and the same thing at batch scale so the
aggregate is readable:

- **Two documents, one sequence, one join.** The notebook prints the six token pairs around the
  join as strings, so you can see the last token of A being asked to predict the first token of B.
  Loss and contributing count are reported with that single position masked and unmasked.
- **A batch of packed sequences**, many joins, with the mean loss *on* the boundary positions
  printed beside the mean loss everywhere else.

**Why the loss changes.** The boundary positions carry a much higher loss than ordinary ones,
because the target is the opening token of an unrelated document and is genuinely unpredictable
from the context. Masking them removes those large terms from the numerator *and* one count each
from the denominator, so the reported mean falls.

The point is not that the number got smaller — you can always make a loss smaller by deleting hard
examples. It is that the unmasked number was measuring a relationship that does not exist, and
gradient was being spent teaching the model that unrelated things follow each other.

### 5 · Untrained perplexity vs vocabulary size

| | value |
|---|---|
| V | 49,152 |
| ln V — the target loss | **10.8027** |
| untrained loss (measured) | *[run]* |
| untrained perplexity | *[run]*, ratio to V *[run]* |

A randomly initialised model has no reason to prefer any token, so it is as unsure as a uniform
draw from the vocabulary. The notebook **asserts** `0.5 < perplexity/V < 2.0` and fails loudly
otherwise, because if this check does not pass the target alignment is wrong and nothing measured
after it means anything.

*Offline verification produced ratio 1.116 — see below.*

### 6 · Tied vs untied head

| Configuration | Parameters |
|---|---|
| Head matrix `V × D` | 49,152 × 576 = **28,311,552** |
| Tied (as shipped) | **134,515,008** |
| Untied | **162,826,560** |
| Cost of untying | **+28,311,552 (+21.0%)** |

The notebook also proves the tying is real rather than nominal, by checking that
`lm_head.weight` and `embed_tokens.weight` share the same storage (`data_ptr()`).

### 7 · Peak memory, ordinary vs chunked cross-entropy

| | loss | peak MiB | logits MiB (analytic) |
|---|---|---|---|
| ordinary CE | *[run]* | *[run]* | *[run]* |
| chunked CE | *[run]* | *[run]* | *[run]* |
| ratio | — | *[run]* | *[run]* |

The chunked implementation is written from scratch in the notebook: flatten to `[N, D]`, walk it in
blocks of `chunk`, project each block to logits, cross-entropy with `reduction="sum"`, accumulate,
divide once at the end.

Two details that matter. Summing and dividing once is not the same as averaging per-chunk means —
the latter silently weights a short final chunk as heavily as a full one. And the saving is real
only because the loss is *unchanged*: the notebook asserts both the loss and the gradients match
the unchunked version, and offline they matched to 0.00e+00 exactly.

Analytic footprint at fp32, showing the ratio is just `predictions / chunk`:

| Batch × context | logits fp32 | chunk 512 | ratio |
|---|---|---|---|
| 4 × 511 | 383.2 MiB | 96.0 MiB | 4× |
| 8 × 1024 | 1,536 MiB | 96.0 MiB | 16× |
| 4 × 8192 | 6,144 MiB | 96.0 MiB | 64× |

Peak allocation is a CUDA counter; on CPU those columns read `nan` and only the analytic figures
are meaningful.

---

## Part 2 — a second head predicting `t+2`

Shared trunk, two heads. Head 1 reuses the pretrained tied head; head 2 is a new untied
`[V, D]` matrix, initialised at `config.initializer_range` so step 0 is a fair `ln V` start rather
than an artificially large one. The losses add.

| | first steps | last steps |
|---|---|---|
| head 1 (t+1) | *[run]* | *[run]* |
| head 2 (t+2) | *[run]* | *[run]* |
| **sum** | *[run]* | *[run]* |
| gap (h2 − h1) | *[run]* | *[run]* |

### What happens, and why

**Head 2 starts far higher and falls much faster.** It begins near `ln V = 10.80` because it is a
fresh random matrix. For the first stretch it is not learning language at all — it is learning to
read a residual stream that already contains useful information. Head 1 has no such catching up to do.

**It then flattens out above head 1 and stays there.** That gap is not a defect and it does not
close with more training, because it is a property of the data rather than of the model.
`P(token t+2 | context up to t)` has strictly higher entropy than `P(token t+1 | context up to t)`:
predicting two ahead means marginalising over the token in between, which the model never gets to
see. Conditioning on less information cannot lower entropy, so head 2's floor sits above head 1's
whatever the architecture. The notebook asserts this ordering.

**Why do it at all** — two motivations that are worth keeping apart:

- *In training*, it densifies the signal. Every position receives two gradients instead of one, and
  the hidden state is pushed to carry information useful beyond the immediate next word. A
  representation that only supports `t+1` has learned something shallower.
- *At inference*, head 2 becomes a draft — speculative decoding where the draft model **is** the
  model, with nothing extra resident in VRAM.

**The honest cost.** Each extra dense head is another `V × D` = 28.3M parameters, +21% on a 135M
model, for predictions that get rejected most of the time at inference. Four heads at V5's scale
would be 2.1B parameters of head alone. That arithmetic is the argument for a factored head, and it
is the same argument Session 7 made about the input side.

---

## Verification

```bash
python3 verify_local.py      # executes the notebook's own code cells and re-derives every claim
```

`verify_local.py` parses `S9_loss_harness.ipynb`, strips the Colab `!pip` magics, and executes the
**actual notebook cells** in one namespace — so a notebook that has been edited since it last ran
cannot pass. It then re-derives each claim independently. A notebook that merely *contains* asserts
proves nothing until something runs it.

This sandbox has no route to `huggingface.co`, so verification runs with `S9_OFFLINE=1`, which
swaps the hub download for a locally constructed `LlamaConfig` of the same architecture (2 layers
instead of 30, for CPU) plus the Session-2 tokenizer. Every shape, mask, count and identity being
checked is independent of depth.

Result — 14/14 cells executed, 16 checks, 15 pass and 1 correctly skipped:

```
PASS  vocab/width match SmolLM2-135M              V=49152 D=576
PASS  shift drops exactly one position
PASS  flattened logits and targets agree
PASS  padding mask lowers the contributing count  508 -> 341
PASS  boundary mask removes exactly one label per join
PASS  boundary mask selects the document-start targets
PASS  single-join case masks exactly one position
SKIP  boundary-positions-are-harder               (random weights: every position is at ln V)
PASS  untrained perplexity lands on V             ppl=54,848  V=49,152  ratio=1.116
PASS  untrained loss equals ln(V)                 10.9123 vs 10.8027
PASS  untying costs exactly V x D                 +28,311,552
PASS  tied head shares storage with the embeddings
PASS  chunked CE gives the same loss              delta=0.00e+00
PASS  chunked CE gives the same gradients         max|d|=0.00e+00
PASS  chunking reduces the analytic logits footprint  383.2 -> 96.0 MiB
PASS  head 2 stays above head 1
```

**The skip is deliberate and worth reading.** "Boundary positions are harder than ordinary ones" is
a claim about a *trained* model. Under random weights every position sits at `ln V` and the
comparison is noise, so asserting it offline would have been a check that passes for the wrong
reason. It is asserted only when real weights are loaded. An earlier version of the verifier did
assert it unconditionally, and it failed — correctly.

---

## Files

| File | What it is |
|---|---|
| `S9_loss_harness.ipynb` | the submission — 28 cells, runs top to bottom on Colab |
| `build_notebook.py` | generates the notebook; edit here, not the JSON |
| `verify_local.py` | executes the notebook's cells offline and re-derives every claim |
| `data/export_shard.py` | re-runs the S4 cleaning stages over the local OpenWebText copy |
| `data/s9_owt_clean.jsonl.gz` | 5,009 cleaned documents, 7.4 MB, sha256 in the manifest |
| `data/s9_owt_clean.manifest.json` | sha256, counts, and which cleaning stages ran |

## Limitations, stated

- The shard is a 3M-word slice of OpenWebText, not the full corpus. Enough for a few hundred
  training steps; not enough to say anything about convergence.
- Part 2 trains for a few hundred steps. It shows the *shape* of the two curves and the persistence
  of the gap. It is not evidence about MTP's value at scale, and the acceptance-rate argument that
  decides whether MTP pays is not measured here at all.
- Peak-memory numbers are CUDA allocator figures for the loss computation only, not whole-model
  training footprints.
- Stage 8 of the S4 pipeline (decontamination) is a no-op for this shard, because no eval set
  travels with it. The manifest says so rather than implying eight stages ran.
