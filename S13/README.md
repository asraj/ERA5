# Session 13 — Reversible training: a 20M LLM, three ways

**Stephen Raj Arokiasamy**

**Notebook:** [`S13_reversible.ipynb`](S13_reversible.ipynb)
**Executed on a free Colab T4 (the submission):** [`S13_reversible_op3.ipynb`](S13_reversible_op3.ipynb) — fp16 + loss scaler, torch 2.11, same source as the notebook, cell for cell
**Also executed:** [`op2`](S13_reversible_op2.ipynb), an earlier T4 run; [`op1`](S13_reversible_op1.ipynb), an L4 in bf16. Both predate the fixes in §7, which change only memory figures (§7 says which).
**Verifier:** `python3 verify_local.py` — 20 cells, 43 checks, runs offline in about a minute

The assignment: train a ~20M LLM for 50M tokens at a batch that fits; train it again with
reversibility and say which variant worked; train the reversible model at the largest batch that
fits; report loss, speed, peak memory and anything else found.

---

## The answer, in one table

Free Colab **T4**, fp16. Model: 20.88M parameters (d=256, 10 layers, context 512, tied GPT-2
embeddings). Data: 50.0M tokens of FineWeb-Edu, one pass, **the same token order in every run**.

| run | batch | steps | final val loss | tokens/s | peak memory | time |
|---|---|---|---|---|---|---|
| 1 · baseline | 32 | 3,051 | **4.7864** | 46,608 | 3.14 GiB | 17.9 min |
| 2 · reversible (leapfrog) | 32 | 3,051 | **4.8539** (+0.068) | 40,523 (−13.1%) | **1.75 GiB** | 20.6 min |
| 3 · reversible, max batch | **1,264** | **77** | **7.1387** (+2.28) | 41,329 (1.02×) | 12.71 GiB | 20.1 min |

**The variant that worked was leapfrog** (Gal et al. eq. 2.6). It won the four-way screen in all three
executions. The midpoint(a) rule at Lightning LM's production settings (step 0.25, a=0.5, Euler
bootstrap) came **last** in all three.

At the same batch, reversibility gave this model **1.80× less peak memory**, cost **15.0% more time per
token**, and ended **0.068 nats worse**. Pushing to the largest batch left 77 optimizer steps instead of
3,051 at the same 50M-token budget, and it bought no speed (1.02×). Run 3 is the configuration the
assignment asks for. What it shows is the wrong way to use the memory reversibility frees (§6).

### How reproducible those numbers are

| | T4, op3 | T4, op2 | L4, op1 (bf16) |
|---|---|---|---|
| baseline val loss | 4.7864 | 4.7881 | 4.7856 |
| leapfrog val loss | 4.8539 | 4.8520 | 4.8529 |
| **gap** | **+0.0675** | **+0.0639** | **+0.0674** |
| reversible time cost per token | +15.0% | +14.6% | +10.7% |

The two T4 runs differ by **0.002** in loss. Same seed, same data, same code for training; the
difference comes from non-deterministic GPU kernels. The leapfrog gap is about 30 times that noise,
and it holds across two GPUs and two precisions. For this configuration it is a real effect. It has
only been measured at one seed and one model size.

---

## 1 · What reversibility is, in my own words

A standard block updates the residual stream as `p ← p + f(p)`. To go backwards you would need
`f(p_old)`, which is computed *from the state you are trying to recover*. So an ordinary transformer
cannot be run in reverse. It has to store every layer's input for the backward pass. The lesson
estimates that at about 34 bytes per token per unit of hidden size per layer, and I measured **36.1**
for this block (§4).

A reversible rule evaluates the block at the state *in between* and adds the result to the state
two layers back:

```
midpoint     p[k+1] = p[k-1] + 2h·f(p[k])            -> p[k-1] = p[k+1] − 2h·f(p[k])
leapfrog     p[k+1] = 2p[k] − p[k-1] + h²·f(p[k])    -> p[k-1] = 2p[k] − p[k+1] + h²·f(p[k])
midpoint(a)  p[k+1] = a·p[k-1] + (1−a)·p[k] + h·f(p[k])
```

If you hold the top two states, you can recompute `f` at the upper one and solve for the one below.
Walking down the stack this way, you rebuild every layer's input from the output you already have.
So the forward pass keeps **two states, whatever the depth**, and each block runs forward twice.

I wrote this as one custom `autograd.Function` for all four rules (`RevStack`). All four share the
form `p[k+1] = a·p[k-1] + b·p[k] + c·f(p[k])`. Its backward pass re-runs one block with a graph,
rebuilds the state below it, pulls the gradient through that one block, and moves down. The baseline
and the reversible models use **the identical block**, from the paper's eq. 2.5. With the rule
`p + f(p)` it is exactly a pre-LayerNorm GPT block. So the only differences between runs are the
update rule and the backward pass.

## 2 · Is the memory trick exact? Separating the algorithm from the arithmetic

The notebook compares the reversible backward with ordinary autograd through **the same rule, with
the same weights and the same batch**, in two ways:

| worst parameter, fp32 | T4 | L4 |
|---|---|---|
| **oracle**: the backward pass is fed the true states | **7.6e-07** | **7.6e-07** |
| **rebuilt**: states reconstructed, as in training | 1.1e-02 | 1.4e-02 |

The oracle column tests the *algorithm*. With the true states, the gradient matches autograd to fp32
rounding, for every rule and both weight scales. So the backward pass is correct. What remains is
*reconstruction error*, and that is the more interesting finding.

**Reversal is exact in real numbers, not in floating point.** For a linear rule, the growth of
rounding error per layer going backwards is `1/|r|`, where `r` solves `r² − b·r − a = 0`. For
midpoint(a) at a=0.5 the roots are 1 and −0.5, so running it backwards **doubles** the error at every
layer. The damping that makes the rule stable going forward is exactly what makes its reversal
ill-conditioned. In exact fp32 arithmetic (the CPU verifier) I measured **×1.90** per layer against a
prediction of ×2.00.

On the GPU that theory stops describing what happens. The blocks run in 16-bit. A rebuilt state
differs from the true one by about 1e-7, and when it is fed back through a 16-bit block, that tiny
difference occasionally flips a rounding decision. Each flip changes the output by a whole 16-bit
unit. So error grows **×4–5 per layer** at initialisation, whatever the rule. On the trained leapfrog
weights, the bottom state of the stack is rebuilt with:

| | fp16 (T4, op3) | bf16 (L4, op1) |
|---|---|---|
| bottom-state reconstruction error | **2.7e-02** | **1.8e-01** |
| gradient vs autograd, rebuilt states | 2.35e-02 | 1.86e-01 |
| gradient vs autograd, oracle (16-bit noise floor) | 1.30e-02 | 2.23e-02 |

**The ratio is 6.7×, close to the 2³ = 8 expected from bf16 having three fewer mantissa bits than
fp16.** So, against the usual advice, a reversible stack is *more* accurate in fp16 than in bf16.
The bf16 run's bottom block trained on gradients that were 19% off, and it still reached the same
loss as the fp16 runs. That also answers a question I could not otherwise settle. The loss gap to the
baseline is the same on both GPUs (+0.067 and +0.068), even though their reconstruction errors differ
by 6.7×. That makes it very likely the gap comes from **the leapfrog rule itself, not from
reconstruction noise**. A leapfrog model trained through plain autograd would confirm it directly,
and I list it below as the control I did not run.

The practical rule: **keep the residual stream in fp32.** With the states themselves rounded to fp16
after every layer, and the ×4 test weights, midpoint(a) rebuilds its bottom state with a relative
error of 6.8, meaning the reconstruction is garbage.

## 3 · Which variant worked

Every variant got a 5M-token screen at batch 32, with the same token order and learning rate:

| rule | T4, op3 | T4, op2 | L4, op1 |
|---|---|---|---|
| **leapfrog** (eq. 2.6, h=0.5) | **6.4595** | **6.4661** | **6.4313** |
| baseline | 6.5064 | 6.4896 | 6.4911 |
| midpoint (eq. 2.4, 2h=1) | 6.5418 | 6.5579 | 6.5817 |
| midpoint(a), no_kick | 6.5827 | 6.5839 | 6.5503 |
| midpoint(a), Euler bootstrap — Lightning LM production | 6.8024 | 6.7707 | 6.6462 |

**Leapfrog won all three times, and every rule trained stably.** The only loss-scale skip in the whole
screen was one step of midpoint on op3. Two further observations hold in every execution:

* **The Euler bootstrap made midpoint(a) worse**, by 0.10 to 0.22 against the same rule with
  `p1 = p0`. Lightning LM's production settings came last each time. That does not contradict their
  report. They tuned for a 120B mixture-of-experts model with 20 layers, and with a=0.5 and h=0.25
  each block contributes a quarter-sized update into a state that is already being averaged. In a
  10-layer, 20M-parameter model, that looks like too little update per layer. midpoint(a) also has
  the worst reconstruction error (§2), so the screen cannot separate those two causes.
* **The screen overstated leapfrog.** It beat the baseline at 5M tokens every time, and lost to it by
  about 0.068 at 50M every time. The notebook calls the screen "not a verdict", and this is why: it
  ranked the reversible rules correctly, and got their comparison with the baseline wrong.

## 4 · Where the memory goes

One training step at batch 32 × 512. The **saved-for-backward** column counts exactly the bytes
autograd keeps between the forward and backward passes, using `saved_tensors_hooks`. It is
device-independent, and it came out identical in all three executions.

| configuration | saved for backward | per token | CUDA peak |
|---|---|---|---|
| baseline, naive head | 4,652.9 MiB | 290.8 KiB | 11,198 MiB |
| baseline, **fused head** | 1,541.4 MiB | 96.3 KiB | 3,219 MiB |
| midpoint(a), plain autograd | 1,541.4 MiB | 96.3 KiB | 3,219 MiB |
| midpoint(a), **RevStack** | **97.4 MiB** | **6.1 KiB** | **1,790 MiB** |

At rest, memory is 258 MiB against the 239 MiB that parameters plus Adam should take, so nothing
is leaking (compare §7).

* **At this size, the output head is the memory problem, not the layers.** One token's logits are
  50,304 numbers, and its hidden state is 256. The standard head accounts for **67%** of everything
  the baseline saves. Left in place, it would set the batch limit for *both* models and hide
  reversibility entirely. So every run uses a fused linear + cross-entropy head that never builds
  the full logit table, and it is checked against `F.cross_entropy` to 1e-6 before use.
* **The rule does not change memory; the trick does.** The same midpoint(a) rule saves 1,541 MiB
  through autograd and 97 MiB through `RevStack`, **15.8× less**.
* **36.1 bytes per token per unit of hidden size per layer**, against the lesson's estimate of 34 for
  a slightly different block.

### Why 15.8× less storage became only 1.8× less peak and 4.6× more batch

| | T4 max batch |
|---|---|
| baseline, naive head | 37 |
| baseline, fused head | 272 |
| reversible (leapfrog) | **1,264** |
| reversible / fused baseline | **4.6×** |

The limits were found by a real out-of-memory search, in which every candidate batch ran two full
steps including the optimizer. The paper reports about 10×. Three things separate "saved for
backward" from "what fits":

1. **Peak memory is the maximum over the phases of a step, not a sum.** In the *head phase*, the
   fused head processes 2,048 tokens at a time. Each chunk's fp32 logits and probabilities, plus
   their 16-bit copies, come to about **1,468 MiB, whatever the batch**. In the *backward phase*, the
   one block being rebuilt holds its full activations, plus buffers for the state and its gradient:
   **10.1 MiB per sequence**. At batch 32 the head phase sets the peak for both models. That is why
   the peak ratio is only 1.80×.
2. **The floor of a reversible stack is one layer, not zero.** Above a batch of about 181 the
   backward phase takes over, and from then on each sequence costs 10.1 MiB against the baseline's
   46.7. 46.7 ÷ 10.1 = 4.6×, which is the ratio the search found. The same arithmetic predicts
   run 3's peak exactly: op2's leak-corrected 12.24 GiB at batch 1,216, plus 48 × 10.1 MiB, gives
   12.71 GiB at 1,264, which is what op3 measured.
3. **The comparison depends on the head.** Against the naive head, reversibility gives **34.2×**.
   The paper's GPT-2 small setup most likely used the standard head. So the honest statement is:
   4.6× at equal heads, and much more against a standard head.

The same analysis says what would help most next. A 512-token chunk would cut the head's
batch-independent temporaries by about four, and at batch 32 that is worth more than anything else
in this table.

## 5 · Speed: why reversibility cost 15%, not 30–50%

Rebuilding costs one extra forward pass of the **blocks**. At this size the blocks are only 7.87M
of the 20.88M parameters. The other **62%** is the embedding and output head, which are never
recomputed. So the extra work is 2 × 7.87M = 15.75M FLOPs per token, on top of 141.0M: a predicted
**11.2%**.

Measured: **+15.0% on the T4** (and +10.7% on the L4). The L4 matches the FLOP count almost exactly.
The T4 pays a few extra points, most likely for the Python-level loop of the reversible backward. The
paper's 30–50% is for models where the blocks carry most of the parameters. Model-FLOPs utilisation
was 10% on the T4. That is low, as expected for a 20M model with unfused elementwise operations, and
it is the same for every run, so it does not bias the comparison.

## 6 · Run 3: the largest batch, and why it is the wrong way to use the memory

At batch 1,264, one step processes 647,168 tokens, so the 50M-token budget allows **77 steps**. The
learning rate followed √(B/B₀), capped at 3×, with a 10% warmup. The run finished at **7.1387**,
where run 2 finished at 4.8539. Throughput was 1.02×, because a 20M model at batch 32 already keeps
the T4 busy, so the larger batch bought nothing.

This is the lesson's step 7 in miniature: *the token budget fixes the global batch*. The memory
reversibility frees is worth having only when something else needs it. Candidates are:

* a deeper or wider model;
* longer sequences;
* dropping gradient accumulation at a global batch the run needed anyway.

It is not a reason to raise the batch for its own sake. The Lightning LM report says the same about
its 2B model: on cards where memory was not the constraint, the standard path was faster.

## 7 · Defects the first GPU runs exposed, and what the rerun confirmed

The CPU verifier cannot reach CUDA code paths, so the first GPU runs (op1 and op2) were the first real
test of them. Going through their numbers turned up three defects in my code. I fixed them before op3.

| defect | seen in op1/op2 | op3, after the fix |
|---|---|---|
| **1.** On the GPU the fused head always computed in 16-bit, even in the check labelled "fp32" | "fp32" head error 5.3e-04 | **1.8e-06** — really fp32 now |
| **2.** An 810.6 MiB leak: the fp16 check left two fp32 chunks of shape 2,048 × vocabulary alive | at rest 1,067 MiB; peaks 4,025 and 2,596 MiB | at rest **258** MiB; peaks **3,219** and **1,790** MiB |
| **3.** The memory-per-sequence predictor was fitted at small batches | reversible limit predicted 6,154, measured 1,216 | predicted 6,584, measured 1,264 — **not fixed** |

For defect 2 I corrected op2's figures by arithmetic before the rerun: 3,214 and 1,785 MiB, and a
1.80× ratio. The rerun measured 3,219 and 1,790 MiB and 1.80×, within 5 MiB, which confirms both the
diagnosis and the correction. The limits the search found rose accordingly: 252 → 272 for the
baseline, and 1,216 → 1,264 for the reversible model.

**Defect 3 is only half fixed, and I now know why.** Moving the fit to batches 32–128 brought the
baseline prediction to within 4% (282 against 272). It did nothing for the reversible model, because
up to a batch of about 181 its peak is set by the head's batch-independent temporaries (§4). No line
fitted below that point can see the 10.1 MiB/sequence that applies above it. The search result,
which is a real measurement, is the number used everywhere. The prediction column is a diagnostic,
and this is the lesson from it: extrapolating peak memory from a linear fit is only safe when one
phase of the step dominates at every batch in the fitted range.

One fix did prove itself on the T4, in all three executions. Dividing the head's gradient by N
*before* the fp16 cast pushes every entry of softmax/N below fp16's smallest normal number. The
notebook reproduces that version on purpose: it has a **2.9e-02** gradient error, against **2.9e-04**
with the division done after the matmul. On the L4 the two are identical, because bf16 keeps fp32's
exponent range.

## What I would not claim

* **That leapfrog is worse than the baseline in general.** The gap of about 0.068 is reproducible here
  (three executions, two GPUs), but it comes from one seed, one size and 50M tokens. The paper found
  reversible rules equal or better at 124M and 772M.
* **That the loss gap is definitely architectural.** The two-precision argument in §2 is strong but
  indirect. The control that would settle it, leapfrog trained through plain autograd, was not run.
* **That the rankings will hold at scale.** The screen is 5M tokens on a 10-layer model. The
  leapfrog-versus-baseline reversal between 5M and 50M tokens shows how fragile such rankings are.
* **That checkpoint resume was exercised on the GPU.** Drive would not mount in any session
  (Colab Enterprise on the L4, a credential error on the T4), and no session disconnected. Resume is
  proven offline instead: `verify_local.py` interrupts a run, resumes it, and gets a **bit-identical**
  final loss and curve.

## Reproduce

```bash
python3 verify_local.py      # 20 cells, 43 checks, ~1 minute on CPU
```

The verifier runs the notebook's own cells with a miniature model and synthetic data, then re-derives
each claim independently. Examples:

* `RevStack` saves exactly **2 × B·T·d·4 bytes** at any depth, while plain autograd grows 4× from 2
  to 8 layers;
* the linear-stability growth factors are re-derived with numpy;
* an interrupted run resumes to a bit-identical result;
* source-level guards stop the three GPU defects from returning;
* no run-dependent number is typed into the findings cell.

Sources: Gal et al., [Reversing Large Language Models for Efficient Training and
Fine-Tuning](https://arxiv.org/abs/2512.02056) (2025) · Shravan, [Reversible Foundations: Training a
120B Sparse MoE through State-Preserving Scaling](https://arxiv.org/pdf/2606.07404) (2026) ·
Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models (2022).
