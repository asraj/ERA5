# Session 11 — Optimizers and learning-rate schedules

**Stephen Raj Arokiasamy**

**Notebook:** [`S11_optimizers.ipynb`](S11_optimizers.ipynb) — runs top to bottom on a free Colab T4.
**Models:** `HuggingFaceTB/SmolLM2-135M` (items 3–4, fp32) and a small transformer built in the
notebook where width is a dial (item 5).
**Data:** the Session-4 cleaned OpenWebText shard, sha256-verified on download.

A gradient gives a direction, not a distance. Every method here is a different rule for choosing
that distance, and this notebook checks five of them against what they actually do.

**Executed copy with all outputs:** [`S11_optimizers_op.ipynb`](S11_optimizers_op.ipynb) —
Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, fp32, seed 11. Every number below is from that
run, with one flagged exception: section G's output cell was cleared before the notebook was saved,
so its figures are carried from the preceding run of the same unchanged cell.

**Item 5 failed on the first run and reproduces cleanly on the second.** Three bugs in my muP
implementation; all three are documented below with what each one did, because the diagnosis is the
useful part. After fixing them, standard parameterization halves its optimum per width doubling
(**0.56×**, predicted 0.50×) and muP holds it flat (**1.12×**, predicted 1.00×).

The notebook also carries a **"Beyond the assignment"** section that checks the eight lesson
mechanisms the assignment does not test — §3 curvature, §4/§5 the two averages, §7 decoupled decay
and the ηλ timescale, §8 optimiser memory, §9 the warmup arithmetic, §11 gradient noise, §13 Muon's
spectrum. Two of those disagree with the notes, and one of my own measurements was made with a
broken estimator; all three are called out rather than tidied away.

---

## 1 · Adam by hand

Adam keeps two exponential moving averages per weight and divides one by the square root of the
other. Both start at zero, so early on they read too low by exactly $(1-\beta^t)$ — dividing that
out is bias correction, and it is exact rather than approximate.

The lesson's own five gradients, every intermediate computed explicitly ($\eta = 0.001$,
$\beta_1 = 0.9$, $\beta_2 = 0.999$, $w_0 = 1$):

| t | g | m | v | m̂ | v̂ | step | w |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.0500 | 0.000250 | 0.5000 | 0.2500 | −0.001000 | 0.999000 |
| 2 | 0.40 | 0.0850 | 0.000410 | 0.4474 | 0.2050 | −0.000988 | 0.998012 |
| 3 | 0.60 | 0.1365 | 0.000769 | 0.5037 | 0.2567 | −0.000994 | 0.997018 |
| 4 | 0.45 | 0.1678 | 0.000971 | 0.4881 | 0.2431 | −0.000990 | 0.996028 |
| 5 | 0.55 | 0.2061 | 0.001273 | 0.5032 | 0.2550 | −0.000996 | 0.995031 |

**Against `torch.optim.Adam` in float64 the worst disagreement over five steps is `0.00e+00`** —
bit-identical, not merely close. All five weights agree to 15 decimal places
(`0.999000000020000`, `0.998011874237702`, …).

PyTorch's own state after the five steps matches the hand computation exactly:

| | PyTorch | by hand |
|---|---|---|
| `exp_avg` (m) | 0.2060650000 | 0.2060650000 |
| `exp_avg_sq` (v) | 0.0012725998 | 0.0012725998 |

Those are the two buffers per weight that make AdamW 8 of Session 10's 16 bytes.

**The property worth reading off the table:** the gradients range over **1.50×**, the steps over
**1.0120×**. The gradient chose the direction; $\eta$ chose the distance. That separation is what
made Adam the default.

### One correction to the lesson

The lesson's prose says these steps fall *"within half a percent of 0.001"*. Its own printed table
does not agree: step 2 is −0.000988, which is **1.20% below** $\eta$, and 1.20% is the worst case
across the five. The table is right and the sentence is about 2× too tight. The claim holds in
spirit — the steps are far more uniform than the gradients — but the number is off, and in a session
about checking rather than trusting, it seemed worth saying. The verifier asserts the measured
value rather than the prose, so nobody later "fixes" the notebook to match the sentence.

---

## 2 · Bias correction disabled

The assignment asks for twenty steps plotted both ways and the number of steps after which the
difference stops mattering. The second part has a surprising answer, so it is worth deriving before
measuring. For a steady gradient:

$$\frac{\text{corrected step}}{\text{uncorrected step}} = \frac{\sqrt{1-\beta_2^{\,t}}}{1-\beta_1^{\,t}}$$

| t | uncorrected step is |
|---|---|
| 1 | **3.16×** too large ← the figure usually quoted |
| 5 | 5.80× |
| 10 | 6.53× |
| **12** | **6.57×** ← the actual worst point |
| 15 | 6.51× |
| 20 | **6.24×** |
| 100 | 3.24× |
| 1,000 | 1.26× |
| 5,000 | 1.00× |

**Over the first twenty steps the difference does not stop mattering — it roughly doubles.** It
starts at 3.16×, peaks near step 12 at 6.57×, and is still 6.24× at step 20.

**When it actually stops mattering:**

| tolerance | steps |
|---|---|
| 10% | **1,660** |
| 5% | **2,327** |
| 1% | **3,916** |

### Why the question is a trap

Two corrections with very different timescales are pulling against each other. $\hat m$'s correction
has timescale $1/(1-\beta_1) = 10$ steps and is within 1% of 1 by **step 44**. $\hat v$'s has
timescale $1/(1-\beta_2) = 1000$ steps and does not get there until **step 3,925**.

Early on the $m$-correction is doing most of the work holding the ratio down. Once it retires around
step 40, the uncorrected step stays inflated by the $v$-term alone, and only decays on the
thousand-step timescale. So the answer is **thousands, not tens** — and which number you quote
depends on the tolerance, which is why all three are reported.

**The practical consequence** is not step size in isolation. Those first few thousand steps are
exactly when warmup is ramping, so an Adam without bias correction is silently running a *different*
warmup schedule from the one written in the config.

---

## 3 · Update-to-weight ratio per layer

$$\text{ratio} = \frac{\lVert \Delta w \rVert}{\lVert w \rVert}$$

measured before and after each optimiser step, for a `q_proj` and an `mlp.down_proj` from the first,
middle and last layers plus the embedding, under a linear warmup to $\eta = 3\times10^{-4}$.

Warmup: linear to $\eta = 3\times10^{-4}$ over 100 steps, 250 steps total.

| layer | ratio @ step 1 | @ warmup end | final | in band? |
|---|---|---|---|---|
| layer0.q_proj | 1.02e-05 | 2.48e-04 | 2.25e-04 | ok |
| layer0.mlp.down | 1.56e-05 | 3.99e-04 | 3.73e-04 | ok |
| layer15.q_proj | 1.53e-05 | 4.00e-04 | 4.48e-04 | ok |
| layer15.mlp.down | 1.62e-05 | 4.03e-04 | 4.77e-04 | ok |
| layer29.q_proj | 2.00e-05 | 4.87e-04 | **5.76e-04** | ok |
| layer29.mlp.down | 1.64e-05 | 3.85e-04 | 4.46e-04 | ok |
| embed_tokens | 1.62e-05 | 6.35e-04 | 3.92e-04 | ok |

Every final ratio lands between **2.25e-04 and 5.76e-04** — a 2.6× spread across seven tensors, and
all of them a factor of 1.7–4.4 below the $10^{-3}$ rule of thumb. Close enough to call healthy for
a 250-step fine-tune, and tight enough across layers that no tensor is running away.

Step 1 starts them all at 1.0–2.0e-05, roughly **25× below where they settle**. That is the ramp
doing its job: the first steps are small while the gradients are still maximally correlated.

**The layer ordering is the interesting part.** The final ratio rises with depth —
layer0 2.25e-04 → layer15 4.48e-04 → layer29 5.76e-04, a 2.6× spread from first to last. The deepest
layers move furthest relative to their size. This is precisely the quantity the lesson says to log
per layer rather than in aggregate; a single global number would have shown ~4e-04 and hidden it.

### The step at which warmup stops changing it

| layer | plateaus at | vs warmup end (100) |
|---|---|---|
| embed_tokens | **47** | −53 |
| layer0.mlp.down | 85 | −15 |
| layer29.mlp.down | 101 | +1 |
| layer15.mlp.down | 195 | +95 |
| layer0.q_proj | 207 | +107 |
| layer15.q_proj | 213 | +113 |
| layer29.q_proj | **230** | +130 |

**There is no single answer, and that is the finding.** The plateau ranges from step 47 to step 230
around a scheduled warmup end of 100 — median 195, roughly **2× later than the schedule**.

The embedding settles at 47, before warmup even finishes. The attention projections do not settle
until 200+, long after $\eta$ went flat. So "the step at which warmup stops changing the ratio" is
not a property of the schedule; it is a property of each tensor. Reading it off the config would
have given 100 for everything and been wrong for six of the seven tracked tensors.

**Warmup ends at a known step; the ratio does not have to stop moving there**, because it is
update over weight and both are still changing. So the notebook detects the plateau from the data —
first step after which a rolling window stays within 10% of the final level — rather than reading it
off the schedule.

**Why the ratio is not simply proportional to $\eta$.** Adam's step is
$\eta \cdot \hat m/\sqrt{\hat v}$, and that middle factor is near 1 only when gradients agree in
sign. Early in a run they do agree — the model is wrong in a consistent direction — so the ratio
tracks the ramp closely. Once gradients start disagreeing the factor falls well below 1 and the
ratio drops *even though $\eta$ is now constant*. Both effects are visible in `update_ratio.png`.

This is also the mechanism behind Section 9's warmup argument: correlated early gradients produce
Adam's largest possible step, which is precisely when you least want it.

---

## 4 · Cosine against WSD, both stopped at step 200

Both schedules configured for 300 steps, both stopped at 200. Same seed, same data order, same
optimiser — only the learning-rate trajectory differs.

| schedule | lr at stop | % of peak | loss (last 20) |
|---|---|---|---|
| cosine | 8.37e-05 | **27.9%** | **3.3284** |
| WSD | 3.00e-04 | 100.0% | **3.4751** |

difference **+0.1467 nats — cosine lower**.

### Which model I would keep

**Cosine won on loss, by 0.1467 nats — and I would still keep the WSD model.** That deserves
justifying rather than asserting, because it is the uncomfortable direction.

Cosine is *ahead on this metric for the same reason it is unusable*: it has decayed to 27.9% of peak
by step 200, and a lower learning rate always looks better on a short-horizon loss reading. It has
spent its decay to buy that number. The WSD model is still at 100% of peak, so it is being measured
mid-flight, with all of its annealing still in hand.

The fair comparison is not "loss at 200" but "loss at 200 after each is allowed to finish". Decay
the WSD checkpoint over 20 more steps and it gets the same noise-suppression cosine already took;
the cosine checkpoint has no such option, because its schedule assumed 300 steps and it has now
been stopped.

**I would keep the WSD model, for a structural reason.** Stopping cosine at 200 of a planned 300
catches it **mid-decay**, at a learning rate it was never meant to finish at. The lesson's phrasing
is exact: a run stopped early has not completed its decay, and the model it leaves behind is worse
than one trained to that shorter length deliberately. The cosine checkpoint at step 200 is not "a
200-step model" — it is an **unfinished 300-step model**.

The WSD checkpoint at 200 is on the flat phase, so it is a legitimate branch point. Decay it over the
next 20 steps and you have a finished 220-step model; or carry on to 1,000. **One run yields
finished models at many budgets**, and that optionality is worth more than a hundredth of a nat.

Honest caveat: this fine-tunes an already-trained model for 200 steps, so it exercises the mechanism
rather than the pretraining regime schedules are designed for. The structural argument does not
depend on the measurement, which is why I lead with it.

---

## 5 · Learning-rate sweep across widths — standard and muP

Sweeping on a small model and reusing the answer is cheap and **invalid**: under the standard
parameterization the best learning rate moves roughly as $1/\text{width}$, so carrying a value from
256 to 4,096 overstates it about sixteenfold.

The sweep is run **both ways** on the same model, same data, same seeds. The lesson's closing warning
applies here more than anywhere — an unfair baseline invents a result.

### The muP rules implemented

With width multiplier $m_d = \text{width}/256$:

| tensor | init variance | learning rate | forward multiplier |
|---|---|---|---|
| input embedding | fixed | fixed | — |
| hidden matrices | $\propto 1/m_d$ | $\propto 1/m_d$ | — |
| output head | $\propto 1/m_d$ | $\propto 1/m_d$ | $1/m_d$ |

Both halves are required — init scaling *and* per-tensor learning rates. Doing one without the other
is not muP and will not transfer. The verifier checks the rules structurally rather than by outcome:
that muP scales the readout init by $1/\sqrt{m_d}$, leaves the embedding alone, and produces two
parameter groups whose learning rates differ by exactly $m_d$.

**42 runs, 593 s.** Compact vocabulary of 4,096 covering 88.0% of tokens; 3 layers; 150 steps;
grid `3e-5 … 3e-2` at ~3.3× per point.

**Standard parameterization** — loss at each learning rate:

| width | 3e-05 | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 |
|---|---|---|---|---|---|---|---|
| 256 | 8.285 | 8.091 | 6.841 | 5.578 | **5.234** | 5.816 | 6.191 |
| 512 | 8.265 | 7.901 | 6.173 | **5.355** | 5.436 | 6.111 | 6.166 |
| 1024 | 8.226 | 7.541 | 5.854 | **5.322** | 6.105 | 6.176 | 6.468 |

**muP:**

| width | 3e-05 | 1e-04 | 3e-04 | 1e-03 | 3e-03 | 1e-02 | 3e-02 |
|---|---|---|---|---|---|---|---|
| 256 | 8.285 | 8.091 | 6.841 | 5.578 | **5.234** | 5.838 | 6.192 |
| 512 | 8.299 | 8.196 | 7.460 | 5.716 | **5.148** | 5.972 | 6.153 |
| 1024 | 8.306 | 8.248 | 7.846 | 5.931 | **5.235** | 5.802 | 6.122 |

Minima located by parabolic fit in log(lr), with every one now **bracketed** (the widened grid was
the point):

| parameterization | width | grid best | refined | bracketed? | ratio to prev |
|---|---|---|---|---|---|
| standard | 256 | 3.0e-03 | 2.72e-03 | yes | — |
| standard | 512 | 1.0e-03 | 1.55e-03 | yes | **0.57×** |
| standard | 1024 | 1.0e-03 | 8.51e-04 | yes | **0.55×** |
| muP | 256 | 3.0e-03 | 2.70e-03 | yes | — |
| muP | 512 | 3.0e-03 | 2.84e-03 | yes | **1.05×** |
| muP | 1024 | 3.0e-03 | 3.35e-03 | yes | **1.18×** |

| | predicted | measured (geometric mean) |
|---|---|---|
| standard, per width doubling | 0.50× | **0.56×** |
| muP, per width doubling | 1.00× | **1.12×** |

**The transfer reproduces.** Standard minima march left almost exactly as $1/\text{width}$ predicts;
muP minima stay put, spanning only **1.24×** across a 4× change in width against standard's 3.2×.
You can read it straight off the raw tables without any fitting: under muP the best column is
`3e-03` at all three widths.

### The three bugs that broke the first run

**1 · The positional table was classified matrix-like.** `self.pos` has shape `(1, T, width)`, so
`dim() == 3`. My split was `n.startswith("emb") or p.dim() < 2 → vector-like`, which files a rank-3
input-side table as a *matrix* and scales its learning rate by `1/m_d`. Classification by tensor
rank instead of by role. Now split by name, and the verifier asserts `pos` lands in the vector group.

**2 · The readout scaling was double-counted.** I divided the init by `√m_d` *and* multiplied the
output by `1/m_d`. muP-for-Adam does neither of the first: when the base init is already
`1/√fan_in`, that **is** muP's hidden-layer init, and the difference from SP is the per-tensor
learning rate plus the readout multiplier. The extra init rescale was my own invention. Init is now
identical in both parameterizations and the verifier asserts it.

**3 · `refine()` reported a confident number for an off-grid minimum.** At width 1024 the standard
minimum landed on `3e-4`, the **lowest point swept** — so it was never bracketed and the true optimum
is somewhere below. My parabolic fit clamped the index to the interior and returned `1.00e-03`,
which is not a location but an artefact. That single bad value is what produced the `0.98×` ratio
and made the standard drift look weaker than it is. `refine()` now returns a `bracketed` flag, the
table prints `NO (edge)`, and ratios built on an unbracketed endpoint are excluded.

The grid was also extended down to `3e-5`. On the re-run all six minima are bracketed and the
transfer reproduces — so the diagnosis was right, which is the only real evidence that a
post-mortem was worth writing.

### So what would I use at width 4,096?

**≈ 3.4e-03, carried across from the muP sweep.** Under muP that is a *transfer* rather than an
extrapolation: the optimum does not depend on width, so a measurement at 256 is a measurement for
4,096. The two candidate answers are now:

| route | value at width 4,096 |
|---|---|
| **muP, carried across unchanged** | **3.35e-03** |
| standard, `1/width` from 256 | 1.70e-04 |

They differ by **19.7×**, and that gap *is* the muP result — it is the size of the mistake you make by
sweeping small and reusing the number without changing the parameterization.

**How confident, honestly:**

- **The mechanism is demonstrated, across 4× of width.** 4,096 is another 4× beyond the largest
  width measured. Believing it there rests on the published results at several billion parameters,
  not on anything in this notebook.
- **Grid resolution is ~3.3× per point.** The parabolic refinement gives a sub-grid *estimate*, not
  sub-grid *accuracy*. Quoting 3.35e-03 rather than "about 3e-03" would be false precision.
- **One seed, 150 steps, 3 layers, 4,096-token vocabulary.** A proxy for the *shape* of the curve.
  Depth, data and run length all move the optimum.
- **The measured ratios are 0.56× and 1.12×, not 0.50× and 1.00×.** Both lean the same way — muP
  drifting slightly up, standard slightly less than halving — which is what you would expect if the
  proxy model is not perfectly in the asymptotic-width regime that muP's derivation assumes. At
  widths of 256–1024 it would be surprising if it were.

So: **3e-03 as the starting point for a width-4,096 run, with a narrow confirmation sweep around it
rather than blind trust.** The lesson's framing is right — the muP day does not buy a better model,
it makes the small model's measurement mean something.

---

## Beyond the assignment — the rest of the lesson, checked

Eight mechanisms the assignment does not test. Each reproduces a table or a claim from the notes,
so each one can be right or wrong rather than merely present.

### A · §3 — why one learning rate cannot serve every weight

`L = (w−5)²`, so one step multiplies the remaining distance by `(1−2η)`. Reproduced exactly:

| η | multiplier | w over five steps |
|---|---|---|
| 0.01 | +0.98 | 0.10, 0.20, 0.29, 0.39, 0.48 |
| 0.10 | +0.80 | 1.00, 1.80, 2.44, 2.95, 3.36 |
| 0.90 | −0.80 | 9.00, 1.80, 7.56, 2.95, 6.64 |
| 1.10 | −1.20 | 11.00, −2.20, 13.64, −5.37, **17.44** |

Then the failure that motivates everything after it — `L = ½(20u² + v²)`, condition number 20:

| η | u multiplier | v multiplier | after five steps |
|---|---|---|---|
| 0.01 | +0.80 | +0.99 | u=+0.328, **v=0.951** — v has barely moved |
| 0.09 | −0.80 | +0.91 | u=−0.328 oscillating, v=0.624 |
| 0.11 | −1.20 | +0.89 | **u=−2.488, diverged** |

The η that suits u leaves v stationary; the η that moves v makes u diverge.

### B · §4 and §5 — what each average buys

Momentum on gradients **5× apart in size** — one alternating ±1, one constant +0.2:

| step | steep g | m | shallow g | m |
|---|---|---|---|---|
| 1 | +1.0 | 0.100 | 0.2 | 0.020 |
| 3 | +1.0 | 0.091 | 0.2 | 0.054 |
| 5 | +1.0 | **0.084** | 0.2 | **0.082** |

Alternation cancels, consistency accumulates: the averages end **within 2%** of each other.

Per-parameter rates, two gradients 100× apart after 200 steps: parameter B is handed a learning
rate **100× larger** than A's, and the two then take **the same step**. The division removes
magnitude and keeps only sign and consistency — which is why a rare word's embedding still moves.

### C · §7 — decoupled decay, and why ηλ is one setting

| parameter | √v̂ | L2 route | decoupled | ratio |
|---|---|---|---|---|
| A | 1.00 | 5.00e-05 | 5.00e-05 | 1× |
| B | 0.01 | 5.00e-03 | 5.00e-05 | **100×** |

Under L2-inside-the-loss, B is regularised 100× harder than A for a reason unconnected to its size.
Decoupled decay verified geometric to **<1e-12** against the closed form `(1−ηλ)¹⁰⁰`.

And the 2025 result made concrete: at η=3e-4, λ=0.1 the timescale `1/(ηλ)` is **33,333 steps** — the
weights you finish with are an exponential moving average of the updates over roughly the last
thirty thousand steps. η and λ are not two knobs; their product is the knob.

### D · §8 — the optimiser is half the memory

| optimiser | bytes/weight | 135M model | 9B model | fits 80 GB? |
|---|---|---|---|---|
| SGD | 8 | 1.00 GiB | 67.1 GiB | yes |
| SGD + momentum | 12 | 1.50 GiB | 100.6 GiB | no |
| **AdamW** | **16** | **2.00 GiB** | **134.1 GiB** | no |
| 8-bit AdamW | 10 | 1.25 GiB | 83.8 GiB | no |

Largest model whose training state alone fits one 80 GB card: SGD **10.7B**, momentum 7.2B,
AdamW **5.4B**, 8-bit AdamW 8.6B. Measured live on SmolLM2 in fp32: parameters 513.1 MiB,
gradients 513.1 MiB, **AdamW state 1026.3 MiB — exactly 2.0× the parameters**.

### E · §9 — one claim confirmed, one not

| gradient behaviour | measured | closed form | lesson says |
|---|---|---|---|
| same sign every step | **1.000 η** | 1.000 η | 1.000 η |
| noisy, zero mean | **0.193 η** | **0.183 η** | 0.281 η |

The first row is exact and matches. **The second does not**, by 31%, and it is not sampling noise.

For iid zero-mean gradients, `m` is a weighted sum of many draws, so by CLT it is `N(0, σ²)` with
`σ² = (1−β₁)/(1+β₁)·σ_g²` while `√v → σ_g`, giving
`E|step|/η = √((1−β₁)/(1+β₁))·√(2/π) = 0.183` at β₁ = 0.9. My measurement matches that derivation to
6%, and the value is **insensitive to the gradient distribution** — normal 0.194, uniform 0.189,
bimodal 0.183 — because the CLT flattens the shape out and σ_g cancels.

Reaching 0.281 requires **β₁ ≈ 0.78**, not the 0.9 used everywhere else in the session. The lesson
labels this row "measured rather than derived" and does not state the settings, so the likeliest
explanation is a different β₁ or a shorter run. Reported as measured with its derivation, rather
than quoted as reproduced.

The warmup arithmetic does check out:

| d_model | init 1/√fan_in | η/init at 3e-4 | vs healthy 1e-3 |
|---|---|---|---|
| 576 | 0.0417 | 0.0072 | 7× |
| 4096 | 0.0156 | **0.0192** | **19×** |

matching the lesson's 0.0192 exactly.

### F · §11 — gradient noise, and how easy it is to measure it wrongly

Gradient of `layer0.q_proj`, 6 disjoint draws at each batch size, measured two ways on the *same*
gradients. The reference is a fixed gradient over 192 held-out sequences (`‖g_ref‖ = 0.0051`).

| batch | mean ‖g‖ | err vs reference | vs batch 1 | 1/√N | naive spread | vs batch 1 |
|---|---|---|---|---|---|---|
| 1 | 0.0175 | 3.2325 | 1.000 | 1.000 | 1.5701 | 1.000 |
| 2 | 0.0123 | 2.2802 | **0.705** | 0.707 | 1.4928 | 0.951 |
| 4 | 0.0102 | 1.8228 | **0.564** | 0.500 | 1.2772 | 0.813 |
| 8 | 0.0098 | 1.6379 | **0.507** | 0.354 | 1.2643 | 0.805 |

**The two right-hand columns are the cautionary tale.** The naive estimator — spread of the draws
about their own mean, `‖gᵢ − ḡ‖ / ‖ḡ‖` — reaches only 0.805 by batch 8 where 1/√N predicts 0.354.
It cannot do better: at batch 1 the spread is **1.57**, meaning each draw deviates from the mean by
more than the mean's own length, so `ḡ` is itself mostly noise. As N grows the reference gets
cleaner along with the samples and the ratio compresses. The estimator saturates exactly where the
effect it is meant to show is largest.

**Against a fixed reference it moves.** Batch 2 lands on 0.705 against a predicted 0.707. Then it
flattens: 0.564 against 0.500, and 0.507 against 0.354.

That residual is worth a sentence rather than a shrug. Fitting
`err² = a²/N + c²` — a sampling term that does fall as 1/√N, sitting on a constant floor — gives
**a = 3.01, c = 1.08 with R² = 0.988**, and removing the floor leaves 1.000 / 0.659 / 0.481 / 0.404
against 1/√N's 1.000 / 0.707 / 0.500 / 0.354. So the 1/√N component is there; something
batch-independent is sitting underneath it.

The reference's own residual noise is too small to explain a floor that size — 192 sequences at a
per-sequence relative noise of 3.23 leaves about **0.23**, not 1.08. The likelier cause is that the
draws are contiguous slices of a document-ordered shard rather than iid samples, so the sampled
rows and the held-out reference rows are not exchangeable, and a systematic offset between them
does not average away with N. Fixing that means shuffling before slicing. **1/√N is a statement
about independent samples, and a contiguous slice of a corpus is not one** — which is the same
reason shuffling matters in the training loop, arriving here from the measurement side.

The scaling table itself is arithmetic and holds either way: ×4 global batch permits ×4 learning
rate under SGD and **×2 under Adam**, because Adam has already divided out the gradient's magnitude
and only its consistency is left to improve.

### G · §13 — why Muon treats a matrix as a matrix

The momentum matrix of `layer0.q_proj`, 576×576, after five real steps. *(This cell's output was
cleared before the executed notebook was saved; the numbers are from the preceding run. The cell is
byte-identical between the two runs and depends only on state built by cells that reproduced
exactly, but it is carried rather than freshly confirmed, so it is marked.)*

| | |
|---|---|
| largest singular value | 3.31e-02 |
| median | 8.09e-06 |
| smallest | 4.71e-12 |
| condition number | **7.0 × 10⁹** |
| top 1 direction carries | 26.6% |
| top 8 carry | 59.1% |
| top 32 carry | 79.7% |
| **effective rank** | **10.1 of 576** |

**The update lives in about 2% of the available directions.** Adam scales each of the 331,776
entries separately, which cannot see this at all — it is a property of the matrix as a map, not of
its entries.

Orthogonalised (Muon's step direction): singular values all **1.0002–1.0004**, condition number
1.0002, cosine **0.133** with the original momentum. Same rough direction, every axis advancing
equally.

That is also the failure mode: making every direction equally strong lets Q and K grow without
limit, attention logits pass 1,000, softmax saturates and the run dies. MuonClip rescales Q and K
after each update — the fix that carried Kimi K2 through 15.5T tokens without a loss spike. And
Muon applies **only** to 2-D matrices; embeddings, norms and the output head stay on AdamW, because
they do not amplify directions and the argument does not apply to them.

---

## Verification

```bash
python3 verify_local.py      # executes the notebook's own cells and re-derives every claim
```

Same discipline as S9/S10: the verifier parses `S11_optimizers.ipynb`, strips the Colab magics, and
executes the **actual cells** — a notebook edited since it last ran cannot pass. It then re-derives
each claim independently, including recomputing Adam from scratch and comparing against the lesson's
printed table at that table's own precision.

Offline it runs against a locally built `LlamaConfig` of the same architecture plus the Session-2
tokenizer, since the sandbox cannot reach `huggingface.co`. **21 cells, 38 checks, all pass.**

Notable checks:

- Adam by hand matches PyTorch to **0.00e+00** in float64, and matches the lesson's table at its
  printed precision.
- The bias-correction answers (1,660 / 2,327 / 3,916 steps) are recomputed independently and
  asserted to be **in the thousands**, so a plausible-looking two-digit answer cannot slip through.
- The muP rules are checked **structurally** — init scaling, embedding exemption, and the exact
  $m_d$ ratio between parameter-group learning rates. Checking muP by whether the minima happen to
  align would be checking the outcome you are trying to measure.

### What the verifier did and did not catch

It caught the Adam arithmetic, the bias-correction timescales, and the LR-group ratios — all of
which were right. It **did not** catch the three muP bugs, and it is worth being precise about why:
every structural check I wrote asserted the rule I had implemented, not the rule muP actually
specifies. A test that encodes the same misunderstanding as the code passes for the wrong reason.

Only the *outcome* caught it — minima drifting 1.52× per doubling when they should have been flat.
That is the argument for keeping an end-to-end measurement even when every unit check is green, and
against my own earlier instinct on item 5 to check muP structurally *instead of* by outcome. Both
were needed. The structural checks now encode the corrected rules and additionally assert `pos` is
vector-like and that init is identical across parameterizations.

### Run-to-run reproducibility

The notebook was run twice end to end on separate T4 sessions. Items 1–4 came back
**byte-identical** — every Adam intermediate, all three bias-correction thresholds, all seven
update-to-weight ratios and plateau steps, both schedule losses to four decimals.

The sweep did not, and *where* it moved is the informative part:

| | run 1 | run 2 |
|---|---|---|
| minima (all six, grid) | 3e-03 / 1e-03 / 1e-03 · 3e-03 / 3e-03 / 3e-03 | identical |
| refined muP at 1024 | 3.38e-03 | 3.35e-03 |
| standard at 1024, lr 3e-02 | 6.348 | **6.468** |
| muP at 256, lr 1e-02 | 5.813 | **5.838** |
| wall clock | 501 s | 593 s |

Same seed, same data order. The drift is confined to the **diverged high-learning-rate columns** —
where the loss is chaotic and a different cuDNN kernel choice is enough to separate trajectories —
and is invisible near the minima, which is the region the conclusion rests on. Conclusions unchanged
to the precision they are quoted at. Worth checking rather than assuming, since "I re-ran it and got
the same answer" is a claim with a precision attached to it.

### A check that failed for the right reason

My first verifier asserted every Adam step lands within 1% of $\eta$, taking the figure from the
lesson's prose. It failed at 1.19%. The lesson's own table says 1.20%, so the code was right and the
sentence was wrong. The assertion now encodes the measurement and explicitly flags the discrepancy.

## Limitations, stated

- One seed throughout. The schedule comparison and every sweep point are single measurements.
- Items 3 and 4 fine-tune a pretrained model, which is not the regime warmup and schedules are
  designed for. The mechanisms are real; the magnitudes would differ from scratch.
- Item 5's model is 3 layers at widths 256–1024 with a 4,096-token compact vocabulary — a proxy
  chosen so the embedding does not dominate at the smallest width and swamp the width effect.
- muP transfer is shown across 4× of width, not the 16× the extrapolation needs.
