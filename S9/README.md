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

All numbers below are from the executed Colab run
([notebook](https://colab.research.google.com/drive/1JWMevPGt0SZUl6PhQWHw07dVDTqiTb8d)),
T4 GPU, torch 2.11.0+cu128, transformers 5.15.1, seed 9.

**Read [What this run does and does not establish](#what-this-run-does-and-does-not-establish)
before quoting items 5, 7 or Part 2.** That run loaded the model in bfloat16, which distorts
three of the nine numbers. Items 1, 2, 3, 4 and 6 are unaffected and final.

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

Batch of four documents truncated to 600/150/900/80 characters, padded to T=128.
Real token counts per sequence: **124, 34, 128, 20**.

| | loss | contributing tokens |
|---|---|---|
| padding counted | **6.3235** | **508** |
| padding masked | **3.2104** | **302** |
| change | **−3.1131** | **−206** (40.6% of positions were padding) |

The loss nearly halves. That is the size of the lie: with padding counted, four out of every ten
positions being scored were the model predicting `<eos>` after `<eos>`, which it does with near
certainty. The reported 6.32 was not a worse model — it was a different, easier question, and the
denominator `B×T = 508` was measuring padding as though it were text.

**The trap this notebook documents:** SmolLM2 ships without a pad token, so the usual fix is
`tok.pad_token = tok.eos_token`. That makes `pad_id == eos_id`, and masking with
`ids == pad_token_id` then silently deletes **every genuine end-of-document token** from the loss.
The mask must come from the tokenizer's `attention_mask`, not from an id comparison.

### 4 · Two documents packed, boundary masked

Done twice — the literal case the assignment asks for, and the same thing at batch scale so the
aggregate is readable:

**4a — two documents, one sequence, one join.** Doc A is 94 tokens, doc B is 101, packed to 195.
The join printed as strings, exactly as the harness shows it:

```
input ' D'    -> target 'ems'
input 'ems'   -> target ' n'
input ' n'    -> target 'CH'   <-- CROSS-DOCUMENT
input 'CH'    -> target 'IC'
```

| | loss | contributing |
|---|---|---|
| boundary trained on | **3.0793** | **194** |
| boundary masked | **2.9997** | **193** |
| **loss at that one join position** | **18.4477** | |
| mean loss everywhere else | **2.9997** | |

**18.4477 against 2.9997.** One position out of 194 was carrying six times the loss of a normal
one, and it was pure noise — the model being asked to predict `CH` (the start of "CHICAGO") from
the end of an unrelated article. In perplexity terms that single position was at ≈10² million
against ≈20 for ordinary text.

**4b — batch scale, 4 × 256, eight joins.**

| | loss | contributing |
|---|---|---|
| boundary trained on | **3.1863** | **1,020** |
| boundary masked | **3.1641** | **1,012** |
| mean loss on the 8 boundary positions | **5.9910** | |
| mean loss on the 1,012 other positions | **3.1641** | |

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
| untrained loss (measured) | **11.2230** |
| untrained perplexity | **74,831** — ratio to V **1.522** |
| pretrained loss / perplexity, for contrast | **3.1863** / **24.2** |

A randomly initialised model has no reason to prefer any token, so it is as unsure as a uniform
draw from the vocabulary. The notebook **asserts** `0.5 < perplexity/V < 2.0` and fails loudly
otherwise, because if this check does not pass the target alignment is wrong and nothing measured
after it means anything.

The check passes (the assert requires 0.5 < ratio < 2.0), but 1.522 is further above 1.0 than it
should be, and the reason is a dtype mismatch rather than anything about the shift: the random twin
is built in fp32 while the loaded model was bf16, so the two perplexities in that cell were not
measured under the same conditions. Fixed in the current notebook, which casts the twin to the
loaded model's dtype. Offline verification, where both are fp32, gives ratio **1.116**.

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

Measured, 4 × 511 = 2,044 predictions, chunk 512:

| | loss | peak MiB | logits MiB (analytic, fp32) |
|---|---|---|---|
| ordinary CE | **3.186291** | 1296.0 | 383.2 |
| chunked CE | **3.186291** | 1167.9 | 96.0 |
| ratio | identical to 2.4e-07 | **1.11×** | **4.0×** |

**Gradients matched to 0.000e+00 exactly** and the loss to 2.38e-07 — the saving is free, which is
the entire point.

**But the 1.11× is a bad measurement, not a bad result**, and the write-up should not quote it as
the headline. Two problems, both fixed in the current notebook:

1. `max_memory_allocated()` is an absolute counter. About 1.1 GiB of weights and activations were
   already resident before the loss was called, so a 287 MiB saving shows up as 1296 → 1168 and the
   effect looks negligible. The notebook now subtracts the pre-call baseline and reports the memory
   *attributable to the loss*, which is what chunking actually changes.
2. The analytic column assumed fp32 logits while the run produced bf16 ones, so the true saving was
   about 144 MiB, not 287. The notebook now sizes the analytic figures off
   `model.lm_head.weight.dtype` instead of assuming.

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

300 steps, batch 8 × 256, lr 3e-5, boundaries masked (157,513 contributing tokens):

| | first steps | last steps |
|---|---|---|
| head 1 (t+1) | 3.0706 | 3.0861 |
| head 2 (t+2) | 12.5632 | 11.6698 |
| **sum** | **15.6337** | **14.7559** |
| gap (h2 − h1) | +9.4926 | +8.5837 |

**These two losses are not usable evidence, and the reason is worth more than the numbers.**
Head 2 finished at 11.67 — *above* `ln V = 10.80`. A head above `ln V` is doing worse than guessing
uniformly at random, so after 300 steps it had not learned anything. Head 1 also went slightly
backwards, 3.0706 → 3.0861.

The cause is the bfloat16 load. AdamW was updating bf16 master weights, and at lr 3e-5 the step
size falls below the gap between representable bf16 values, so most updates rounded away to
nothing. The shapes of the two curves are therefore an artefact of numerics, not a property of
`t+1` versus `t+2` prediction.

The current notebook fixes this at the source (fp32 load) and adds two guards so it cannot pass
silently again: an assert that the optimiser's parameters are fp32 before training starts, and a
check after training that prints a loud diagnosis if head 2 finishes above `ln V`.

**Re-run to get clean numbers.** The reasoning below is what the corrected run should show, and it
does not depend on the measurement — it follows from entropy.

### What happens, and why

**Head 2 starts far higher and falls much faster.** It begins near `ln V = 10.80` because it is a
fresh random matrix. (In the bf16 run it started at 12.56 rather than 10.80 — a fresh head whose
logits are scaled by the residual stream is somewhat *worse* than uniform before any training.) For the first stretch it is not learning language at all — it is learning to
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

## What this run does and does not establish

The executed Colab run loaded SmolLM2 in **bfloat16** (transformers v5 defaults to the
checkpoint's dtype) and the heads were then cast to match. That is fine for the forward-only
items and wrong for the rest.

| Item | Status | Why |
|---|---|---|
| 1 shapes | **sound** | structural, dtype-independent |
| 2 shift on strings | **sound** | token ids and strings, no arithmetic |
| 3 padding mask | **sound** | loss computed in fp32 via `.float()`; counts are exact |
| 4 boundary mask | **sound** | same; the 18.45 vs 3.00 contrast is the strongest result here |
| 5 untrained perplexity | **passes, slightly inflated** | random twin fp32 vs model bf16 — not the same conditions |
| 6 tied vs untied | **sound and exact** | parameter counting |
| 7 chunked memory | **correct but badly measured** | absolute counter dominated by 1.1 GiB of resident weights; analytic column assumed fp32 while logits were bf16 |
| Part 2 | **not usable** | AdamW on bf16 master weights at lr 3e-5 — updates rounded away, head 2 never got below `ln V` |

Three fixes are already in the current `S9_loss_harness.ipynb`:

1. **fp32 load.** `from_pretrained(..., dtype=torch.float32)`, so the optimiser has weights it can
   actually update and the analytic logit sizes are honest.
2. **Isolated memory measurement.** `peak()` subtracts the memory resident before the call and
   reports what the loss itself costs; the analytic figures are sized from
   `model.lm_head.weight.dtype` rather than assuming fp32.
3. **Two guards on Part 2.** An assert that the optimiser's parameters are fp32 before the first
   step, and a post-training check that prints a loud diagnosis if head 2 finishes above `ln V`
   instead of letting a plausible-looking number through.

Re-running the current notebook end to end should give a usable Part 2 and a chunked-memory ratio
near the analytic 4×. Items 1–4 and 6 above will not change.

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

Result — 14/14 cells executed, 21 checks, 20 pass and 1 correctly skipped:

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
PASS  head 2 dtype follows the trunk
PASS  TwoHeadModel works on a bfloat16 trunk   head2=torch.bfloat16  l1=11.109 l2=10.917
PASS  TwoHeadModel works on a float16 trunk    head2=torch.float16   l1=10.980 l2=11.097
PASS  TwoHeadModel works on a float32 trunk    head2=torch.float32   l1=10.969 l2=10.936
```

### One bug the offline substitute did not catch

The first Colab run failed in Part 2 with
`expected mat1 and mat2 to have the same dtype: c10::BFloat16 != float`.

transformers v5 loads SmolLM2 in its checkpoint dtype — **bfloat16** — while `nn.Linear` builds
in fp32 unless told otherwise, so head 2's first matmul had a bf16 activation and an fp32 weight.

The verification missed it for an instructive reason: the offline substitute reproduced the
architecture faithfully and the *dtype* not at all, being fp32 throughout. A stand-in only tests
what it actually resembles, and dtype was the one property that mattered here.

Fixed in three places — the model now loads with an explicit `dtype=torch.float32` (which also
keeps item 7's analytic fp32 logit sizes honest), head 2 is built in the trunk's dtype and device
rather than PyTorch's default, and item 5's random-init twin is cast to match the loaded model so
the two perplexities are measured under the same conditions. The regression above rebuilds
`TwoHeadModel` on bf16/fp16/fp32 trunks and runs a real forward pass through each; reverting the
fix makes it fail with the identical error, so it has teeth. See
[`COLAB_PATCH.md`](COLAB_PATCH.md) for the paste-in version.

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
| `COLAB_PATCH.md` | the bf16 dtype fix, as a paste-in patch if you have already uploaded |
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
