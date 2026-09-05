# Session 10 — The training loop, made to tell the truth about itself

**Stephen Raj Arokiasamy**

**Notebook:** [`S10_training_loop.ipynb`](S10_training_loop.ipynb) — runs top to bottom on a free
Colab T4. **Executed copy with all outputs:** [`S10_training_loop_op.ipynb`](S10_training_loop_op.ipynb).
**Model:** `HuggingFaceTB/SmolLM2-135M` — 134,515,008 params, V = 49,152, D = 576, 30 layers.
**Data:** the Session-4 cleaned OpenWebText shard, sha256-verified on download.
**Run:** Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, fp32, seed 10.

Every number below is from that executed run. Two of my predictions were wrong and are corrected
in place rather than quietly dropped — see items 4 and 5.

---

## 1 · Every tensor in a step, and what each dimension is

| Tensor | Shape | What the dimensions are |
|---|---|---|
| `batch` | `(4, 256)` | B sequences × T positions — token ids |
| `labels` | `(4, 256)` | same grid, `-100` where the position must not train |
| `hidden` | `(4, 256, 576)` | one residual-stream vector per position |
| `logits` | `(4, 256, 49152)` | one raw score per vocabulary entry, per position |
| `loss` | `()` | **scalar** — the single number the whole run is steered by |
| `lm_head.weight` | `(49152, 576)` | the parameter |
| `lm_head.weight.grad` | `(49152, 576)` | `dL/dw` — **one number per weight**, identical shape |
| `exp_avg` | `(49152, 576)` | Adam's running mean of the gradient |
| `exp_avg_sq` | `(49152, 576)` | Adam's running mean of the gradient *squared* |

**`loss is a scalar: 3.0017 — from 50,331,648 logits.`** Fifty million numbers collapse to one, and
that one number has to move 134 million weights.

The shape that matters is the repetition: gradient, master copy and both optimiser buffers all have
the weight's shape. That is the 16 bytes per weight:

| what must be held | bytes/weight | this model |
|---|---|---|
| weight (bf16) | 2 | 0.25 GiB |
| gradient (bf16) | 2 | 0.25 GiB |
| fp32 master copy | 4 | 0.50 GiB |
| Adam `exp_avg` + `exp_avg_sq` (fp32) | 8 | 1.00 GiB |
| **total** | **16** | **2.00 GiB** — before a single activation |

**And the part that table leaves out.** Parameters are a fixed cost you can compute on paper;
activations depend on batch and sequence length, and they are why "the model fits" and "training
fits" are different questions. Measured for one batch of 4 × 256:

| | MiB |
|---|---|
| forward only | 1,918.3 |
| forward + backward | **2,494.3** |
| the logits alone | 192.0 (8% of it) |

The 2.00 GiB of training state above is fixed. This 2.49 GiB scales with `B × T`, so doubling the
batch doubles it — which is the real constraint on batch size, and the reason activation
checkpointing exists.

**Where the 134,515,008 parameters actually are:**

| component | params | share |
|---|---|---|
| MLP | 79,626,240 | **59.2%** |
| embedding / lm_head (tied) | 28,311,552 | 21.0% |
| attention | 26,542,080 | 19.7% |
| norms / other | 35,136 | 0.0% |

The MLP is three times the attention block. Attention is the famous part; it is not the expensive
part.

---

## 2 · A gradient verified by hand

**2a — the lesson's chain**, `w1=3, w2=4, x=2, t=20`, in float64:

| | by hand | `backward()` | central difference | abs err |
|---|---|---|---|---|
| `dL/dw1` | 64 | **64.000000** | **64.000000** | 8.06e-09 |
| `dL/dw2` | 48 | **48.000000** | 48.000000 | 7.50e-09 |

**2b — one real weight inside SmolLM2**, the entry of `lm_head.weight` with the largest `|grad|`:

```
probe weight: lm_head.weight[2299, 17]
  value                 +0.03954078
  loss at w+0.001        2.9744260311
  loss at w-0.001        2.9740302563
  central difference    +0.19788742
  backward() reported   +0.19787554
  relative difference    6.002e-05      -> agreement to 4.2 decimal digits
```

**Why h = 1e-3 and not h = 1e-9.** The instinct that a smaller nudge gives a better derivative is
wrong, and the notebook sweeps `h` to show it. Two errors fight each other: truncation falls as
`h²`, floating-point cancellation grows as `ε/h`. Their sum is U-shaped, and on this weight the
minimum sits at exactly the `h = 1e-3` the probe uses.

| h | central difference | rel. error vs `backward()` |
|---|---|---|
| 1e-1 | 0.18279195 | 7.62e-02 |
| 1e-2 | 0.19773245 | 7.23e-04 |
| **1e-3** | **0.19788742** | **6.00e-05** ← best |
| 1e-4 | 0.19907951 | 6.08e-03 |
| 1e-5 | 0.17881393 | 9.63e-02 |
| 1e-6 | **0.00000000** | 1.00e+00 |
| 1e-7 | **1.19209290** | 5.02e+00 |

Read the bottom two rows. At `h = 1e-6` the two losses are **bit-identical in fp32**, so the
difference is exactly zero and the estimate is 0.00000000 — the derivative of a function that, as
far as fp32 can tell, did not change. At `h = 1e-7` one single ULP of difference survives and gets
divided by `2h`, producing **1.19209290** — which is not a coincidence: fp32 machine epsilon is
1.1920929e-07. The "derivative" there is pure quantisation noise amplified by a factor of ten
million.

A finite difference is a measurement, and measurements have a noise floor.

Two things that are the difference between this working and not: **`model.eval()` first** (a finite
difference compares two forward passes; anything stochastic between them is measured as gradient),
and **central rather than one-sided** differences.

---

## 3 · Gradient accumulation, broken on purpose

$$\text{correct}=\frac{\sum_i \ell_i}{\sum_i n_i} \qquad\qquad \text{wrong}=\frac{1}{k}\sum_i \frac{\ell_i}{n_i}$$

**The lesson's example, reproduced exactly:** `(4·2.0 + 4·2.0 + 2·5.0)/10 = 2.6000` against
`(2.0+2.0+5.0)/3 = 3.0000` — **+15.4%**.

**Measured on the real model**, four micro-batches with genuinely unequal supervised regions:

| micro-batch | valid tokens | mean loss |
|---|---|---|
| 1 | 510 | 2.9742 |
| 2 | 382 | 2.9255 |
| 3 | 126 | 3.4531 |
| 4 | 254 | 2.9460 |

| | value |
|---|---|
| token-weighted (correct) | **3.0014** |
| average of averages (wrong) | **3.0747** |
| **error** | **+2.44%** |
| equal-length control | **−0.00%** ← how it hid |

After 60 steps trained both ways (`accumulation.png`): correct **2.8569**, wrong **2.8858** — a gap
of **+0.0289 nats** that is still widening.

**The bug restated as a weight per token**, which is what it actually is. Neither formula is "an
average"; both assign a weight to every token and they disagree about what it should be:

| micro-batch | tokens | weight/token (correct) | weight/token (wrong) | ratio |
|---|---|---|---|---|
| 1 | 510 | 7.862e-04 | 4.902e-04 | 0.62× |
| 2 | 382 | 7.862e-04 | 6.545e-04 | 0.83× |
| 3 | 126 | 7.862e-04 | **1.984e-03** | **2.52×** |
| 4 | 254 | 7.862e-04 | 9.843e-04 | 1.25× |

Under the correct formula every token weighs the same, `1/1272`. Under the wrong one a token in the
shortest micro-batch counts for **4.05× one in the longest** (510/126), purely because of how
sequences happened to be grouped on that step. Nothing about the text justifies it.

**The mechanism, stated precisely.** The error size is driven by micro-batches having different
**mean losses**, not merely different token counts — equal means give zero error however unequal the
counts. Unequal counts are just the usual *reason* the means differ. That is also why an untrained
model cannot exhibit it: every token sits at ≈ ln V and there is nothing to distort.

**One methodological point.** Both curves are *reported* with the correct token-weighted loss; only
the gradient differs. Plotting the broken run's own broken loss would flatter it.

---

## 4 · The grad norm and the loss

A known shock at a known step — a batch of one token repeated — labelled as such, because a "found"
spike on natural data is easy to fool yourself with. Baseline is **median ± MAD**, not mean ± σ: on
a short window a 3σ band is tight enough that ordinary wobble crosses it, and my first detector did
exactly that and flagged a step *before* the injection.

| step | loss | dev | grad norm | dev |
|---|---|---|---|---|
| 19 | 3.4365 | +3.1 σ | 4.155 | +0.7 σ |
| **20** | **13.4386** | **+77.2 σ** | **1371.552** | **+4244.4 σ** |
| 21 | 3.5149 | +3.7 σ | 4.241 | +1.0 σ |

Baseline: loss 3.0154 ± 0.1350, norm 3.931 ± 0.322.

**The norm reacts 55× more strongly** — 4244 σ against 77 σ. There is a second difference worth as
much: **the norm is back to baseline at step 21** (+1.0 σ) while the loss is still elevated at +3.7 σ
and does not fully settle until step 23. The norm is a sharper instrument in both directions — it
spikes higher and it recovers faster, so it localises *which step* was bad rather than smearing the
event across the next few.

### Where I was wrong

I predicted the shock's loss would go **down**, on the reasoning that a repeated token is trivially
predictable after the first position, and wrote a whole paragraph about a monitor watching for the
loss to rise recording an improvement. **On the real pretrained model the loss went up, +77 σ.** The
offline random-weight run did show a drop, which is what misled me: a random model has no
expectations to violate, so the repetition just looks like more uniform noise. A pretrained model
finds a wall of one rare token genuinely astonishing.

So the honest finding is weaker than the one I wanted, and still worth having: **both traces fire,
but the norm fires 55× harder.** On a dashboard with alert thresholds, one of those is
unmissable and the other is a bad step.

### And on ordinary data, the norm does not lead

Cross-correlation on a clean run with no injection, lags −3 to +3: strongest at **lag −2, r = −0.250**
— a negative lag, meaning the loss leads if anything, and a correlation too weak to act on. Every
lag is under |0.26|.

The lesson says the grad norm "moves before the loss does". Over 40 steps of ordinary data I could
not reproduce that, and I am not going to report a 0.25 correlation as confirmation. What I *can*
show is that on an anomaly the norm is far more sensitive. The norm's value is in outlier detection,
not routine prediction.

### Choosing the clip threshold from data

The lesson lists this as an open question and says to pick it from the distribution rather than from
habit. Measured on the clean run:

| percentile | grad norm | | candidate cap | would clip |
|---|---|---|---|---|
| p5 | 3.579 | | 0.50 | 100.0% |
| p50 | **3.976** | | **1.00** | **100.0%** ← the usual default |
| p90 | 4.586 | | 2.00 | 100.0% |
| p95 | 4.889 | | 4.89 (p95) | 2.5% |
| p99 | **5.272** | | **5.27 (p99)** | **0.0%** |

**The usual default of 1.0 would clip every single step.** That is not a safety net, it is a
learning-rate change in disguise: the direction survives but the magnitude is set by the cap rather
than by the data, so the effective learning rate becomes `lr × cap / ‖g‖` and drifts with the
gradient norm. A cap at p99 = 5.27 leaves ordinary steps untouched and still catches the shock,
which at 1371.552 was **345× the median**.

---

## 5 · MFU, measured

```
20 steps, batch 4 x 256 = 1,024 tokens/step
  wall time        7.36 s   (368 ms/step)
  tokens/second    2,782
  achieved 6N·t/s  2.25 TFLOP/s
```

| denominator | peak | MFU |
|---|---|---|
| **T4 fp32** (the honest one — we train in fp32 on CUDA cores) | 8.1 TFLOP/s | **27.72%** |
| T4 fp16 tensor cores | 65 TFLOP/s | 3.45% |
| A100 bf16 tensor cores | 312 TFLOP/s | 0.72% |
| H100 bf16 tensor cores | 989 TFLOP/s | 0.23% |

The last two rows are arithmetic, not measurement — the same token rate judged against faster
silicon. **A run does not become efficient by being measured against a slower card**, which is the
reason to state the denominator every time an MFU is quoted.

**The `6N` approximation excludes the attention term**, which grows with sequence length. At T = 256
on a 135M model that term is small, but the figure is an approximation and flatters at long context.

### Where I was wrong, again

Before running this I wrote that a healthy 35–50% MFU was "not reachable by tuning" on a T4 and that
the number would say more about the card than the loop. **27.72% is far better than I expected** —
most of the way into the healthy band, not a different regime. That prediction was pessimistic and
the measurement corrected it.

The table also shows where the honesty lives: the same run is **27.72% or 3.45%** depending on which
T4 peak you quote, and neither is a lie. 27.72% is the fraction of the fp32 path we actually used;
3.45% is the fraction of the silicon we paid for. The second is the more uncomfortable number, and
it is the argument against training fp32 on a card that has tensor cores.

### What is costing me the distance to 40%

Ranked, largest first:

1. **Not using tensor cores at all.** fp32 on CUDA cores caps us at 8.1 TFLOP/s while 65 TFLOP/s of
   fp16 silicon sits idle. A T4 is Turing and has no bf16, and Session 9 showed fp16 is the wrong
   precision for AdamW at these learning rates — so on this card the choice is genuinely between a
   correct slow path and a fast wrong one. On Ampere or later this stops being a dilemma.
2. **The batch is tiny** — 1,024 tokens per step, so kernel launches, optimiser traversal and Python
   overhead amortise over almost nothing. The biggest lever available without changing hardware, and
   it costs only memory.
3. **The output head dominates a 135M model.** V/D = 85×, so the `(4, 256, 49152)` logits and their
   gradient are the largest tensors in the step and cross-entropy over them is bandwidth-bound.
   `6N` counts those FLOPs as useful work and is blind to the memory traffic they generate.
4. **No fused kernels** — stock attention, unfused AdamW, `.float()` casts on the logits.
5. **30 layers at 576 wide is thin**, so each matmul is small and the GPU is launch-latency bound.

### Where the time actually goes

MFU says how much of the machine you are using. It does not say what is using it. Timed separately:

| phase | time | share |
|---|---|---|
| forward | 108.0 ms | 33.5% |
| backward | 214.8 ms | 66.5% |
| optimiser step | 0.1 ms | 0.0% |
| **total** | **322.9 ms** | |

**backward / forward = 1.99×**, against the ~2× theory predicts — the backward pass does two matmuls
per forward one, computing gradients with respect to both the inputs and the weights. Seeing the
prediction land almost exactly is a useful check that nothing pathological is happening in autograd.

The optimiser is **0.03%** of the step, which is worth knowing in the other direction: AdamW touches
all 134M weights once per step regardless of batch size, so if that share were large it would mean
the batch was far too small. Here it is negligible, so batch size is a memory-bound choice rather
than an amortisation one.

---

## 6 · 0.1 in fp32, bf16 and fp8 E4M3

0.1 is `0.0001100110011…` repeating in binary — not representable in *any* of these. Normalised it
is `1.6 × 2⁻⁴`, so all three store exponent −4 and differ only in how well the mantissa approximates
1.6. Derived by hand, then checked against what the hardware stores; the notebook asserts the
reconstruction equals the stored value.

| format | sign | exponent | mantissa | hex | significand | stored | rel. error |
|---|---|---|---|---|---|---|---|
| **fp32** (1+8+23) | 0 | `01111011` | `10011001100110011001101` | `0x3DCCCCCD` | 1 + 5033165/8388608 | 0.10000000149011612 | 1.5e-08 |
| **bf16** (1+8+7) | 0 | `01111011` | `1001101` | `0x3DCD` | 1 + 77/128 = 1.6015625 | 0.10009765625 | 9.8e-04 |
| **fp8 E4M3** (1+4+3) | 0 | `0011` | `101` | `0x1D` | 1 + 5/8 = 1.625 | 0.1015625 | 1.6e-02 |

Exponent field: fp32/bf16 `123 − 127 = −4`; fp8 `3 − 7 = −4` (bias 7).

### Which I would train in, and why

**bf16, with an fp32 master copy of the weights** — which is exactly what the 16-bytes-per-weight
table in item 1 describes.

The reasoning is *not* the accuracy table above. On accuracy alone bf16 is a poor showing and fp8
worse. **Accuracy is the wrong axis.** The deciding property is range, and the notebook runs the
argument rather than asserting it:

| gradient | fp16 | bf16 | fp16 with ×1024 loss scaling |
|---|---|---|---|
| 1e-4 | 1.00e-04 | 1.00e-04 | 1.00e-04 |
| 1e-6 | 1.01e-06 | 9.98e-07 | 1.00e-06 |
| 1e-8 | **ZERO** | 1.00e-08 | 1.00e-08 |
| 1e-10 | **ZERO** | 1.00e-10 | 1.16e-10 |

**fp16 flushes to zero at 1e-8**, exactly where the lesson says it does; bf16 does not, because it
kept all eight of fp32's exponent bits. A gradient
of exactly zero means that weight stops moving — the model quietly stops learning precisely where
the remaining signal was faintest. The last column is loss scaling: it works, and it is one more
knob to tune and eventually get wrong. **bf16 deletes the knob**, which is why a format with 2.4
decimal digits beat one with 3.3.

**And why the master copy is not optional.** The notebook prints the gap between representable
numbers near 0.1. An AdamW update of ~1e-5 on a weight of ~1e-2 is a 0.1% change; in bf16 the
spacing at that magnitude is ~0.8%, so the update lands *below* the grid and rounds to nothing. The
weight never moves. Session 9 hit exactly this — pure-bf16 AdamW at lr 3e-5 left a head above `ln V`
after 300 steps, learning nothing while producing a perfectly plausible loss curve.

**fp8 is a production recipe in 2026 and NVFP4 is ~1.73× faster still**, but neither is a blanket
replacement: they need per-block shared exponents, and attention stays in higher precision because
softmax amplifies whatever noise you hand it. The rule is not "use fewer bits", it is *shrink where
the error does not accumulate*.

**On this hardware the question is moot** — a T4 has no bf16 path, which is item 5's answer too. So
the notebook pins fp32: correct for this card, and not what I would run at scale.

---

## A bug this notebook caught in itself

The first executed run printed:

```
real weight lm_head[2299,-0.18596420602103195]
```

A column index that is a negative float. The cross-correlation loop in item 4 used `c` as its
variable name, silently overwriting the column index `c` from item 2 several cells earlier. Nothing
raised; the summary just printed a number that could not possibly be an index.

It is a trivial bug with a non-trivial moral, and it belongs in this write-up rather than being
quietly fixed: a notebook is one long mutable namespace, the failure was silent, and the only reason
it was caught is that the value was printed somewhere a human would read it. Which is the session's
whole thesis. Renamed to `rho`, with the reason in a comment so nobody re-uses `c` there.

---

## Verification

```bash
python3 verify_local.py      # executes the notebook's own cells and re-derives every claim
```

`verify_local.py` parses `S10_training_loop.ipynb`, strips the Colab magics, and executes the
**actual cells** — a notebook edited since it last ran cannot pass. It then re-derives each claim
independently, including recomputing all three bit patterns for 0.1 from scratch rather than
trusting the notebook's own function. Offline it runs against a locally built `LlamaConfig` of the
same architecture (2 layers, for CPU) plus the Session-2 tokenizer, since that sandbox cannot reach
`huggingface.co`. **12 cells, 29 checks, all pass.**

### Two checks that were wrong before they were right

- **The accumulation gap.** I first asserted the measured error exceeds 1%. Offline it came out
  −0.13% — correctly, because the substitute model is random and mis-weighting identical means
  changes nothing. The assertion now covers the lesson's exact arithmetic, which is
  model-independent, and the magnitude is asserted only with real weights.
- **The grad-norm detector.** The first version used mean ± 3σ over the pre-shock window and flagged
  a step *before* the injection. Now median ± MAD at 5σ.

## Limitations, stated

- One seed. The accumulation gap, the MFU figure and the shock response are single measurements.
- 60 steps per accumulation arm shows the curves separating, not where they converge.
- MFU is measured on a T4 in fp32, which is not a realistic training configuration — see item 5.
- The `6N` FLOP estimate excludes attention and so overstates MFU, more so at longer context.
- The grad-norm shock is injected and labelled. It shows the trace works; it says nothing about how
  often this occurs naturally.
- The clean-run cross-correlation is 40 steps. That is too short to settle whether the norm leads
  the loss on ordinary data, which is why I report it as unresolved rather than as a negative result.
