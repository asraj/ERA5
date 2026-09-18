# Session 12 — Distributed training I: data parallel and ZeRO

**Stephen Raj Arokiasamy**

**Notebook:** [`S12_zero.ipynb`](S12_zero.ipynb) — pure PyTorch, runs top to bottom in about a
minute. No downloads, no dataset, no accelerator required.
**Executed copy with outputs:** [`S12_zero_op.ipynb`](S12_zero_op.ipynb) — all 25 cells, clean, in
order, on a free Colab **Tesla T4** (torch 2.11.0+cu128), including the `torch.distributed`
cross-check.
**Verifier:** `python3 verify_local.py` — 25 cells, 45 checks.

The assignment: build 32 virtual GPUs, run a demo model on them, simulate ZeRO-1/2/3, show how
memory and computation change.

The trap in that assignment is that every number involved is already printed in the lesson. A
notebook that prints 16.00 / 5.50 / 3.75 / 2.00 has demonstrated nothing, because those four
numbers can be typed. So the rule I set myself was: **every byte reported here is either a tensor
that was really allocated on a rank, or a byte that was really moved between two ranks by a
collective I wrote and then checked against `torch.distributed`.** Where that was impossible — you
cannot measure an InfiniBand link in Colab — the number comes from a stated cost model with its
bandwidth written down, never from a stopwatch.

---

## The one sentence the session rests on

**A reduce-scatter followed by an all-gather is an all-reduce.**

Everything else falls out of that. Data parallelism averages gradients with an all-reduce. A ring
all-reduce is *already implemented* as those two phases — it reduces into slices, then gathers the
slices back. So ZeRO-1 and ZeRO-2 are not a new algorithm and not a new cost: they run the two
phases data parallelism was running anyway, and **keep the intermediate slice instead of throwing
it away**. Ten of the sixteen bytes, for free, because the communication was already happening.

That reframing is the thing I actually took from this session. ZeRO-1 and ZeRO-2 sound like they
should cost something. They do not, and the reason is that data parallelism was already paying for
them and discarding the receipt.

I did not want to take that on faith, so the notebook verifies the identity directly:
reduce-scatter the same buffer, all-gather the slices back, and compare against a plain all-reduce
— `0.00e+00`.

---

## What a "virtual GPU" is here, and what it is not

A GPU in this notebook is a rank index and a dict of tensors. That is the honest description, and
it is sufficient, because the two quantities the session is about are **what each rank stores** and
**what each rank sends** — neither of which needs silicon.

What it is **not** is a simulation of speed. Compute is serialised across the 32 ranks in one
process, so wall-clock time here is meaningless and I never report it as a result.

Three places a simulation like this can quietly cheat, and what I did about each:

1. **Pretending to shard.** If ZeRO-3 kept the full weights and only *reported* a shard, every
   number would still look right. So the ranks genuinely hold `1/32`-sized tensors, and the forward
   pass genuinely reassembles them with an all-gather before it can run.
2. **Asserting the byte counts.** The `2P` figure is easy to hardcode. Instead the cluster charges
   each collective by the size of the buffers actually passed to it, and a separate ring
   implementation walks all `2(W−1)` sends one at a time so the cost factor is *derived*.
3. **Trusting my own collectives.** Everything downstream rests on them, so they are checked
   against real `torch.distributed` gloo in separate OS processes over a socket. All four agree to
   float tolerance. (gloo has no native reduce-scatter, so that leg is built as all-reduce + slice
   — which is the section-4 identity being *used*, not worked around.)

---

## 1 · Why this is a problem at all

One weight does not cost one number. It costs five:

| what is stored for one weight | bytes |
|---|---|
| the weight, in the 16-bit format used for arithmetic | 2 |
| its gradient | 2 |
| a 32-bit copy of the weight, kept for accuracy | 4 |
| Adam's two 32-bit running averages | 4 + 4 |
| **total** | **16** |

The fp32 master copy is the one worth understanding rather than memorising. Repeatedly adding a
very small update to a 16-bit number loses it to rounding — at bf16's 8 mantissa bits, an update
smaller than about `2⁻⁸` of the weight simply does not land. So the authoritative value is kept at
full precision and a 16-bit copy is made from it for the arithmetic. The two averages are Adam's
`m` and `v` from Session 11; they are the reason the optimizer, not the model, is the largest
single consumer of memory.

`30e9 × 16 = 480 GB = 447.0 GiB`. A card holds 74.5 GiB. **Six cards to hold the model before a
single calculation happens**, and activations on top of that.

`P`, one full copy of the parameters in bf16, is **60 GB**. Every communication cost below is a
multiple of it.

---

## 2 · The four arrangements, measured

Measured on a real transformer — 2 layers, `d_model` 64, vocabulary 256 — trained across 32 virtual
ranks. Bytes per weight are obtained by walking each rank's tensors and summing
`numel × element_size`. No formula is applied; the formula is what this is checked *against*.

| arrangement | weight | gradient | master + m + v | **measured B/weight** | 30B per GPU | fits 74.5 GiB? |
|---|---|---|---|---|---|---|
| data parallel | 2 | 2 | 12 | **16.000** | 447.0 GiB | no |
| ZeRO-1 | 2 | 2 | 12/32 | **4.375** | 122.2 GiB | no |
| ZeRO-2 | 2 | 2/32 | 12/32 | **2.438** | 68.1 GiB | yes |
| ZeRO-3 | 2/32 | 2/32 | 12/32 | **0.500** | 14.0 GiB | yes |

At the world size the lesson tabulates (W=8) the same formula gives **16.00 / 5.50 / 3.75 / 2.00**,
matching exactly.

**ZeRO-3 at 32 ranks is 0.5 bytes per weight, and `0.5 × 32 = 16`.** The state has not been
compressed — it has been *divided*. That check is in the notebook because it is the thing the name
says and the tables hide: **Zero Redundancy** Optimizer. Nothing here makes the model cheaper to
train. It makes it possible to train at all, by refusing to store the same thing thirty-two times.

### What crosses the wire

| arrangement | collectives per step | measured | predicted |
|---|---|---|---|
| data parallel | all-reduce | **1.9375 P** | 2P |
| ZeRO-1 | reduce-scatter, all-gather | **1.9375 P** | 2P |
| ZeRO-2 | reduce-scatter, all-gather | **1.9375 P** | 2P |
| ZeRO-3 | all-gather, all-gather, reduce-scatter | **2.9062 P** | 3P |

The measured figures are deliberately *just under* the round ones, and that is not an error.
`2P` is shorthand for `2(W−1)/W · P`, which at W=32 is `1.9375P`. A ring never sends the last
`1/W` — a rank does not mail itself its own slice. I count the exact factor everywhere.

Stage 3's extra `P` is one all-gather of the weights in the forward pass and one more in the
backward, because the gathered weights are freed in between. It does **not** need to all-gather the
updated weights at the end of the step the way stages 1 and 2 do — the next forward pass's gather
does that job. That is why stage 3 is 3P and not 4P.

### And the computation?

**It does not change.** Every rank does one forward and one backward over its own 2 sequences in
every arrangement — identical FLOPs, to the last operation, across all four. ZeRO does not save
compute and does not cost compute.

This is worth saying plainly because "how does computation change" invites an answer, and the
correct answer is *it doesn't*. **ZeRO trades memory for communication and touches nothing else.**
That is the entire shape of the technique.

---

## 3 · The equivalence claim, split in two

The lesson says averaging gradients from 8 GPUs that each saw 32 sequences produces *exactly* the
gradient one GPU would produce from all 256. Two separate questions hide in that sentence and they
have different answers, which is why the notebook asks them separately.

**Question 1 — does the sharding itself change anything?** Compare each ZeRO stage against plain
data parallelism: same batches, same reduction, the only difference is where the state lives.

| | max \|loss − DP\| | max \|weights − DP\| |
|---|---|---|
| ZeRO-1 | **0.00e+00** | **0.00e+00** |
| ZeRO-2 | **0.00e+00** | **0.00e+00** |
| ZeRO-3 | **0.00e+00** | **0.00e+00** |

Exactly zero. Splitting the optimizer state, then the gradients, then the weights themselves
changes nothing about the model being trained. The right slice reaches the right rank and the
pieces reassemble bit-for-bit.

**Question 2 — is the 32-way split identical to one GPU on the whole batch?** No — and I think
rounding that away would be the single easiest way to fake this assignment.

| weights in | max \|loss − single\| | max \|weights − single\| |
|---|---|---|
| bf16 | 1.287e-05 | **6.001e-04** |
| fp32 | 6.258e-07 | **1.468e-06** |

Two candidates for the disagreement: summation order (one mean over 64 sequences versus 32 means
of 2, then averaged) and bf16 rounding after every step. They are separable — re-run with fp32
weights and change nothing else, and whatever survives is summation order. **bf16 accounts for
roughly 409× of it** on the T4. What remains in fp32 is irreducible: `a + b + c` summed in two
orders is two different numbers.

That multiplier is the one figure here that moves with the backend — the same notebook on CPU gives
196×, because CUDA and CPU accumulate differently. The *conclusion* does not move, and the notebook
now says so in place rather than leaving a reader to wonder why two runs disagree. Everything else
in this write-up is either exact or a closed form, and reproduces identically on both.

So the claim is **exactly true of the mathematics and approximately true of the arithmetic**, and
the approximation is dominated by the precision of the weights rather than by the distribution.

A third place the order matters, measured separately: my collectives reduce with
`torch.stack(...).mean(0)`, but a ring accumulates sequentially, so rank 5's contribution enters
the sum at a different point. Ring versus direct differs by **1.86e-09 in fp32 and 1.53e-05 in
bf16** — the latter about 1.1% of the signal. Neither is a bug. Both are what "the same
computation in a different order" costs, and together they are why two runs at different world
sizes are not expected to match bit-for-bit even with the same seed.

---

# Beyond the assignment

Five more mechanisms from the lesson, each reduced to something that can be right or wrong. Three
of them said something I had not expected.

## A · The floor that nothing gets under

The lesson claims data parallelism and ZeRO-1 never fit *at any world size*. More GPUs usually
fixes memory, so that deserved deriving rather than reading.

Both leave the **weights and the gradients replicated** on every card. That is `2 + 2 = 4` bytes
per weight, and **there is no `W` anywhere in that expression**:

| W | replicated part | sharded part | ZeRO-1 total |
|---|---|---|---|
| 8 | 111.8 GiB | 41.9 GiB | 153.7 GiB |
| 64 | 111.8 GiB | 5.2 GiB | 117.0 GiB |
| 512 | 111.8 GiB | 0.7 GiB | 112.4 GiB |
| 8,192 | 111.8 GiB | 0.0 GiB | **111.8 GiB** |

So "how big a model can data parallelism train" has an answer that does not mention the cluster:
**4 bytes/weight fills a 74.5 GiB card at exactly 20.0B parameters.** V5 at 30B is 1.5× past that
line. One division eliminates two of the four arrangements before any benchmark is run, and it is
the most useful thing in the section.

The full ladder reproduces the lesson's table to within **0.04 GiB** at every cell.

## B · The lesson's two step times are one assumption

The lesson gives 7.10 s per step on 64 H100 and 3.12 s on 64 B200. I rebuilt both from `6ND` and
published peak throughput rather than taking them on trust:

| card | 64 × peak (bf16 dense) | ideal step | lesson says | **implied MFU** |
|---|---|---|---|---|
| H100 | 6.33e16 FLOP/s | 2.84 s | 7.10 s | **40.0%** |
| B200 | 1.44e17 FLOP/s | 1.25 s | 3.12 s | **40.1%** |

**They land on the same number.** The two step times are not two measurements — they are one
assumption, MFU ≈ 40%, applied to two peak figures. That is a reasonable assumption and the
conclusion drawn from them survives it, but it is worth knowing before quoting them as evidence
about hardware.

And the conclusion is the uncomfortable one:

| | compute | 2P over InfiniBand | comm/compute | sum if unhidden |
|---|---|---|---|---|
| H100 | 7.10 s | 2.40 s | **34%** | 9.50 s |
| B200 | 3.12 s | 2.40 s | **77%** | 5.52 s |

The volume did not change. The compute it has to hide behind got 2.3× shorter. **Buying faster
cards makes a run more network-bound, not less** — the only line in this session where the right
answer gets *harder* as the hardware improves.

## C · The 83% overlap figure is 5/6, and it is a ceiling

The backward pass runs last layer to first, so the last layer's gradients are ready long before the
first layer's and can start moving immediately. Buckets control how often that happens.

The lesson states three things. Two reproduce; the third turned out to be structural.

**The 83% is not a measurement.** With N buckets, the final bucket's gradients **do not exist**
until the backward pass has finished, so its transfer cannot begin before then. No more than
`(N−1)/N` of the traffic can ever be hidden. At a bucket of 2 layers over 12 layers that is 6
buckets, and `5/6 = 83.3%`:

| bucket | buckets | ceiling (N−1)/N | H100 measured | B200 measured |
|---|---|---|---|---|
| 1 | 12 | 91.7% | **91.7%** | 64% |
| 2 | 6 | 83.3% | **83.3%** | 64% |
| 3 | 4 | 75.0% | **75.0%** | 60% |
| 6 | 2 | 50.0% | **50.0%** | 42% |

**H100 sits exactly on its ceiling at every bucket size; B200 sits below it.** That one comparison
is the whole diagnosis: on H100 the backward lasts 4.73 s against 2.40 s of transfer, so the link
has slack and finishing earlier is always better. On B200 the backward is 2.08 s and the transfer
is still 2.40 s — the link is now the critical path.

Which is why the best bucket **reverses**: smallest on H100, two layers on B200. Same configuration
file, opposite right answer, decided entirely by which side is longer.

That reversal cannot happen with a free link, so the model has exactly one free parameter — the
fixed cost `α` of starting a transfer. Rather than pick a flattering value I solved for the range
that makes both statements true: **α ∈ [29.0, 86.5] ms**. That band is far above a bare NCCL launch
latency of tens of microseconds, so the lesson's "fixed cost of starting a transfer" is standing in
for more than launch overhead. I use 50 ms and say so, rather than presenting a fitted parameter as
a measured one.

## D · Offload, priced in PCIe seconds

The optimizer state is 12 of the 16 bytes and is touched exactly once per step, which makes it the
obvious thing to move off the card. PCIe carries ~60 GB/s — comparable to a network cable and 7.5×
slower than NVLink. Offload converts a memory problem into a bandwidth problem.

At ZeRO-1 on 32 GPUs, per card:

| scheme | bytes over PCIe | time | % of an H100 step |
|---|---|---|---|
| park the state, update on the GPU | 22.5 GB | 0.38 s | 5% |
| park the state, **update on the CPU** | 3.8 GB | **0.06 s** | **1%** |

Doing the update on the CPU wins by 6×, because the state then never crosses PCIe at all — only the
gradient goes down and the new weight comes back.

But both free the same 10.5 GiB, taking the card from 122.2 to **111.8 GiB** — and it *still* does
not fit, because 111.8 GiB is exactly the 4-byte floor from section A. **Offloading the optimizer
cannot touch the term that is blocking you.** Offload is worth buying when memory is the binding
constraint; here it is merely a large one.

## E · What 8-bit actually saves

| | bytes/weight | 30B model |
|---|---|---|
| bf16 weights and gradients | 16.0000 | 447.0 GiB |
| MXFP8 weights and gradients | **14.0625** | 392.9 GiB |

A block of 32 values shares one 8-bit exponent, so the scale costs `2 × 1/32 = 0.0625` bytes per
parameter across the two tensors. The 12 bytes of optimizer state are untouched — the update still
needs the accuracy — and they are **75% of the total**, which is exactly why the saving is only
**12.1%**.

Where it does pay is the wire, because `P` halves:

| | comm/compute in bf16 | in MXFP8 |
|---|---|---|
| H100 | 34% | **17%** |
| B200 | **77%** | **38%** |

On B200 that is the difference between a transfer filling three quarters of the step and one
filling well under half. **12% off the stored state is the least interesting thing 8-bit does**;
halving the wire volume is what it is for, and it lands hardest on exactly the configuration
section B showed to be in trouble.

Softmax stays in fp32. The exponential turns a modest gap between two scores into a very large
ratio, so it amplifies input error — and it is bandwidth-bound rather than arithmetic-bound, so the
precision is nearly free there. The rule is not "use 8-bit everywhere"; it is **"use high precision
where it costs nothing"**, which is a different and better rule.

---

## F · What I would run for V5, and where I disagree

V4 ran ZeRO-2 on 8 GPUs and never needed stage 3. At V4's scale, ZeRO-2 on 8 cards needs about
1 GiB per card against 74.5 available — it had room to spare. The same arrangement for V5 needs
104.8 GiB.

The session frames the choice as ZeRO-2 on 32 or ZeRO-3 on 8:

| option | GiB/GPU | headroom for activations | wire | IB seconds | % of an H100 step |
|---|---|---|---|---|---|
| ZeRO-2 on 32 | 68.1 | **6.4 GiB** | 2P | 2.40 s | 34% |
| ZeRO-3 on 8 | 55.9 | 18.6 GiB | 3P | 3.60 s | 51% |
| **ZeRO-3 on 32** | **14.0** | **60.5 GiB** | 3P | 3.60 s | 51% |

**My reading of my own numbers is ZeRO-3 on 32, which is neither of the two on offer.** It costs
the identical 3P that ZeRO-3 on 8 costs, and leaves 60.5 GiB for activations rather than 18.6. The
session explicitly notes that activation memory is missing from the table and pushes the practical
threshold up — and recomputation buys activation memory at ~30% more compute, which is a worse
trade than spending GPUs I have already been given. ZeRO-2 on 32 leaving 6.4 GiB for activations
looks to me like the genuinely risky option, not the conservative one.

The caveat I would attach to my own recommendation: I have not measured activation memory for V5's
architecture, and "60.5 GiB of headroom" is only decisive if activations actually fit in it.

On the four open questions:

| question | what this notebook says |
|---|---|
| ZeRO-2 on 32, or ZeRO-3 on 8? | wrong pair — ZeRO-3 on 32 dominates ZeRO-3 on 8 at identical wire cost |
| how many GPUs per node? | NVLink is 9× InfiniBand: a 3P step costs 0.40 s inside a node and 3.60 s across |
| 8-bit from the start? | the argument is the wire, not the memory — and it helps most where §B says we are in trouble |
| does state go to system memory? | not at ZeRO-3 on 32; we are not memory-bound there |

The thread through all four: **once the arrangement is chosen, every remaining decision is about
the interconnect, not about the cards.**

---

## Verification

```bash
python3 verify_local.py      # executes the notebook's own cells and re-derives every claim
```

Same discipline as S9/S10/S11: the verifier parses `S12_zero.ipynb`, strips the Colab magics and
executes the **actual cells** — a notebook edited since it last ran cannot pass. It then recomputes
each claim from scratch rather than reading the notebook's variables where that is possible.
**25 cells, 45 checks, all pass**, in about 6 seconds on CPU.

The checks that matter most:

- **The section-4 identity**, recomputed independently: reduce-scatter then all-gather equals
  all-reduce to `0.00e+00`.
- **The ring cost factor** is asserted to be exactly `2(W−1)/W`, so a byte-counting bug cannot hide
  behind a rounded "2P".
- **Bytes per weight is re-derived from first principles** in the verifier and compared against
  what the notebook *measured* from real tensors — two independent routes to the same number.
- **DP and ZeRO-1 never fit**, checked at W = 8, 64 and 4,096, and the 4-byte floor asserted to be
  invariant at W up to a million.
- **The 40% MFU** recovered from both cards' step times independently.
- **The 83% overlap** asserted to equal 5/6 exactly, so it is tested as a structural ceiling rather
  than as a fitted number.
- **Determinism across runs.** Two independent T4 executions were diffed line by line: **every
  measured value is bit-identical**, and the only differences are wall-clock jitter, which this
  write-up never quotes as a result. The byte accounting and the closed forms are also identical
  between CPU and GPU; the one figure that legitimately moves with the backend is called out in
  §3 and in the notebook itself.
- **A stale-literal guard.** Reviewing the first executed run caught a real defect: the closing
  recap cell had a hardcoded "~2,500×" for the bf16 factor while the cell above it *measured* a
  different number. The figure is now interpolated, and the verifier scans every non-f-string
  `print()` in that cell and fails on any number that is not a lesson constant. I confirmed the
  guard bites by reintroducing the original bug — it fails. This is the same class of error as the
  stale dimension comment in S7, so it now has a test rather than an apology.

### What I would not claim from this

- **Wall-clock time means nothing here.** 32 ranks share one process. Every time in this write-up
  comes from a stated bandwidth and a stated FLOP count, never from a measurement.
- **The bucket model has a free parameter.** Section C reports the range of `α` that reproduces the
  lesson rather than a fitted point estimate, but it is still a model, not a measurement.
- **The gloo cross-check runs at world size 4, not 32.** It establishes that my collectives compute
  what the real library computes; it does not exercise a 32-process run.
- **ZeRO-3's gather is at whole-model granularity**, not per layer. The same bytes move, in fewer
  and larger pieces. Real FSDP2 gathers per `fully_shard` unit, which is what makes the transient
  memory small — my version's transient peak is one full model, so it demonstrates the
  communication correctly and the peak-memory benefit optimistically.
- **One seed, one model, one shape.** The byte accounting is exact and shape-independent; the
  training curves are a single run.
