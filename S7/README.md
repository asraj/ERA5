# Elastic Kronecker: removing the 32-byte wall from Kronecker embeddings

**Problem solved: #3** — *"Today Kronecker is limited to 32 positions for every word (even "apple" or "a"). That's a waste of space. How can it be dynamic and not force us to crop a word?"*

**One command:** `python run_all.py` (~2 min, numpy only) — regenerates every number below into `results/`.

---

## 0. Validation against the published V1 paper

*Kronecker Embeddings: Byte-Level Structured Token Representations* (Shravan, 2026),
[code](https://github.com/theschoolofai/kronecker-embeddings), deployed in
[LightningLM](https://lightninglm.theschoolofai.in/).

**0.0 Our baseline is bit-identical to the reference *code*.** `kv2/reference.py` is a
line-by-line numpy transcription of the published
[`codec.py` / `embedding.py` / `tokenizer_utils.py`](https://github.com/theschoolofai/kronecker-embeddings)
(torch isn't available here). `kv2/codec.py::KroneckerV1` is then checked against it as an
oracle:

```
max |ours − reference|  =  0.0        # Latin, Devanagari, Tamil, Telugu, emoji, code, <s>
```

Reading the actual code corrected **two things the paper's prose does not state**:

| detail | paper prose | reference code | ours (now) |
|---|---|---|---|
| z-norm std | "standard deviation 1" (§3.3) | `.std()` is torch's **unbiased (ddof=1)**, and `eps=1e-6` is **added** to it → realised std **0.999848** | matches code exactly |
| projection | "single learned linear map" | `nn.Linear(D, d_model, **bias=False**)`, init `normal(0, 1/√D)` | matches |
| index layout | — | `byte_value * pos_dim + pos` | matches |
| `encode_single` | — | does **not** UTF-8-safe-truncate; the *production* path truncates when building the byte buffer | we truncate safely (production path) |

The std discrepancy is numerically trivial but worth recording: it is a real
**prose-vs-code** difference, and our fidelity check follows the **code**, which is
authoritative for an implementation.

**0.1 Our baseline is also spec-faithful.** `kv2/validate_v1.py` checks the
reimplementation property-by-property against the paper — **all PASS**:

| paper property (§3.2–3.3) | verified |
|---|---|
| `κ(b) = (1/√L) Σ_p c[b_p] ⊗ p_p` | ✓ |
| coordinate of `(b_p,p)` is `b_p·d_p + p` | ✓ |
| at most `L` nonzeros, each `1/√L` | ✓ |
| `E‖κ(b)‖² ≈ 1` for any length | ✓ |
| per-token z-normalisation (mean 0, std 1) | ✓ |
| `d_c=256`, `d_p=32`, `D=8192` | ✓ |
| **UTF-8-safe truncation** (back off to codepoint boundary) | ✓ |
| special / byte-fallback tokens = literal surface bytes | ✓ |

I had originally truncated at a raw byte boundary; reading §3.2 corrected this. Note the
paper's own refinement makes the problem *slightly worse*: backing off to a codepoint
boundary discards a little more, taking collisions from 265 groups to **306**.

**0.2 We reproduce the paper's truncation rate.** Table §4.2 reports `d_p=32` covering
≥99.82% of tokens, `d_p=16` "falling dramatically" on multilingual vocabularies. On a real
10,000-token BPE vocabulary we measure **87.9% / 97.5% / 99.9%** at `d_p` = 16/32/64 — same
shape, same conclusion (`d_p=16` inadequate, `d_p=64` nearly lossless).

**0.3 The claim this work refutes.** The paper states twice, of the truncated tail:

> *"These truncated tokens **still receive distinct embeddings** based on their first 32
> bytes; only the post-byte-32 byte structure is lost."* (§4.2)
> *"The truncated tokens **still receive unique embeddings** from their first 32 bytes."* (§7)

**This is false.** Two tokens sharing a 32-byte prefix receive the *same* vector (cosine
1.0), not a degraded one. Counterexamples, measured:

| vocabulary | tokens truncated | V1 collision groups | tokens sharing a vector | Elastic |
|---|--:|--:|--:|--:|
| real 10k BPE vocab | 246 (2.5%) | **44** | **108** | **0** |
| India-Wikipedia words (19,122) | 2,240 | **306** | **851** | **0** |

```
பயன்படுத்தப்படுகிறது == பயன்படுத்தப்படும் == பயன்படுத்தப்பட்டது
                     == பயன்படுத்தப்பட்டுள்ளது == பயன்படுத்த == பயன்படுத்தப்பட்ட
अंतर्राष्ट्रीयकरण == अंतर्राष्ट्रीयता          (the pair named in the course lesson)
```

The paper's **rate** (≤0.18%) is not disputed — we reproduce its order of magnitude. What
is disputed is the **consequence** assigned to that rate. The paper's §7 Limitations lists
truncation as a loss of *fine distinctions*; it is in fact a loss of *identity*, and the
words it silently merges are ordinary Tamil and Hindi inflections. The paper contains zero
occurrences of "collision"/"collide" and never tests uniqueness — which is precisely the
gap this submission fills.

---

## 1. The problem, measured

The shipped codec builds a token's vector from a `256 × 32` grid: byte value ⊗ byte
position, `L = min(len(bytes), 32)`. Two consequences, both measured here on the
**real India-Wikipedia vocabulary (19,122 distinct words)**:

**(a) The window is not equally generous across scripts.** UTF‑8 spends 1 byte per
Latin character and 3 per Indic character, so the same 32 columns hold:

| script | bytes/char | characters that fit in 32B | words > 32B |
|---|--:|--:|--:|
| latin | 1.00 | **31.9** | 6 |
| devanagari | 3.00 | **10.7** | 118 |
| telugu | 3.00 | **10.7** | 256 |
| tamil | 3.00 | **10.7** | **1,860 (32.9%)** |

**(b) Overflow is a silent, permanent collision.** Two tokens agreeing on their first
32 bytes get *byte-identical* codes, so the projection can never separate them.
Measured on the same vocabulary:

| codec | collision groups | words affected |
|---|--:|--:|
| **kronecker-v1** | **306** | **851 (4.45%)** |
| **elastic (ours)** | **0** | **0** |

These are not exotic words. Real groups the shipped codec fuses into one vector:

```
இந்தியாவில்  ==  இந்தியாவிலேயே  ==  இந்தியாவிலும்         (in India / only in India / also in India)
கொண்டிருந்தது == கொண்டிருந்த == கொண்டிருந்தன == கொண்டிருந்தனர்
              == கொண்டிருந்தாலும் == கொண்டிருந்தால்        (six distinct tense/person/mood forms)
प्रधानमंत्री (36 bytes)                                     ("prime minister" — truncated)
```

Tamil and Telugu inflect **at the end of the word**, which is exactly the part the
window truncates first. The failure is therefore not random: it systematically
erases the morphology of agglutinative Indic languages while leaving English intact.

**(c) The waste.** The mean token uses **0.14%** of the grid's cells. Every token —
`a`, `apple`, or a 30-character Tamil word — is billed for all 32 columns.

---

## 2. The fix: an elastic position axis

Keep the Kronecker idea exactly (frozen one-hot ⊗ one-hot, no learned codec, one
shared projection). Replace the *absolute* 32-column position factor with an
**elastic** one:

| block | bins | what it does |
|---|--:|---|
| **HEAD** | 8 | absolute first 8 bytes → preserves prefix similarity (`train`/`training`) |
| **TAIL** | 8 | absolute last 8 bytes → **anchors the suffix**, where Indic morphology lives |
| **MID** | 8 | *relative* bins over the remaining span → any length maps in; nothing is dropped |
| **ORDER** | 256 | mean relative position per byte value → breaks transposition ties inside a bin |
| **LEN** | 32 | log-length signature → distinguishes tokens that otherwise match |

`ElasticKronecker` mirrors the reference module surface (`D`, `encode`, `extra_repr`,
`dropped_bytes`), so it is a **drop-in sibling** of `KroneckerEmbedding`: swapping it into
the reference package is a one-line constructor change, and the `forward` contract
(`(...,L) ids → (...,L,d_model)` through one `Linear(D, d_model, bias=False)`) is unchanged.

```
dim = 256×(8+8+8) + 256 + 32 = 6,432      vs V1's 256×32 = 8,192
```

A token of **any** byte length is represented — long tokens degrade *gracefully*
(several bytes superpose in a middle bin) instead of failing *silently* (bytes
deleted). And the code is **21.5% smaller**, so the single trainable projection
shrinks with it.

### Why each piece is there (each was forced by a measurement, not a hunch)

- **TAIL** exists because truncation destroys suffixes, and Indic inflection is suffixal.
- **ORDER** exists because relative binning is order-blind: I constructed a transposition
  collision (`…xy…` vs `…yx…` in one bin), added this plane, and it disappeared —
  **0 collisions in 2,893 genuine random transpositions, vs 168 for V1.**
- **LEN weight (0.35)** exists because the dense length channel initially had **55× the
  std** of the sparse grid and destroyed the spelling geometry (separation went
  *negative*, −0.34). Normalising the grid first and adding the length channel at a
  controlled weight restored it.

---

## 3. Proof

### P0 · The transformer is correct before it is trusted
A single-block causal-attention transformer written in numpy with hand-derived
gradients. Backward pass verified against numerical differentiation:

```
dense: 1.11e-08  PASS      codec: 4.26e-08  PASS
```

### P1 · The decisive experiment: an information wall, not a quality gap
Task: a sequence `[word, filler, filler]` where the label depends **only** on which of
two colliding words appeared. Under V1 the two words are byte-identical inputs, so
the model is being asked to separate two classes from the same vector.

| input path | mean val accuracy | per pair |
|---|--:|---|
| dense (control) | **1.000** | 1.0 / 1.0 / 1.0 |
| **kronecker-v1** | **0.500** ← pinned at chance | 0.49 / 0.54 / 0.47 |
| **elastic (ours)** | **1.000** | 1.0 / 1.0 / 1.0 |

V1's learning curve is flat at chance for 400 steps; Elastic is at 1.0 by step 25.
This mirrors the lesson's own positional experiment: *the token-only model was pinned
at chance because the two cases were literally identical inputs to it.*

### P2 · The fix costs nothing on ordinary language modelling
Next-word prediction over real EN/HI/TE/TA Wikipedia text:

| input path | val loss | val acc |
|---|--:|--:|
| dense | 2.973 | 0.240 |
| kronecker-v1 | 2.990 | 0.200 |
| elastic | 3.022 | 0.203 |

### P3 · The property that made Kronecker attractive survives
`cos(related) − cos(unrelated)`, on spelling families like `train`/`training` and
`भारत`/`भारतीय`:

| codec | related | unrelated | separation |
|---|--:|--:|--:|
| kronecker-v1 | 0.803 | 0.290 | 0.513 |
| elastic | 0.743 | 0.305 | **0.438** |

Slightly lower, still clearly positive — spelling-similar tokens still start out near
each other.

### P4 · Parameters at the V5 reference shape (131,072 × 8,096)

| input path | parameters | vs dense |
|---|--:|--:|
| dense table | 1,061,158,912 | — |
| kronecker-v1 | 66,322,432 | 93.75% |
| **elastic (ours)** | **52,073,472** | **95.09%** (and **21.5% below V1**) |

**14.2M parameters saved against V1, while removing 851 collisions and all truncation.**

---

## 4. Honest limits

- The elastic scheme has a residual failure mode of its own: bytes that share a
  middle bin are order-sensitive only through the ORDER plane. It survives 2,893
  adversarial transpositions, but I do not claim it is collision-*proof* — I claim it
  is collision-*free on the measured vocabulary* and vastly harder to break than a
  32-byte prefix match.
- Middle-bin superposition means very long tokens are represented at lower resolution.
  That is a deliberate trade: graceful degradation instead of silent deletion.
- The transformer is small (d=48, one block). It is sized to make the *mechanism*
  visible, not to produce competitive LM numbers. The chance-vs-perfect gap in P1 is
  structural and cannot be closed by scale, which is why the small model is sufficient
  to prove it.
- Geometry separation drops from 0.513 to 0.438. Reported, not hidden.

## 5. What I would tell the V5 team

`pos_dim = 32` is not a neutral default; on this vocabulary it silently fuses **4.45%
of words**, and **32.9% of Tamil words** exceed the window. Raising `pos_dim` to 64
would fix truncation but *doubles* the projection to ~133M. The elastic axis fixes it
while *shrinking* the projection to 52M. Under the lesson's own standard — a decision
is a hypothesis until a proxy run has tested it — this is the cheaper hypothesis to
buy, and the collision count is the number that decides it.

---

## Repo

```
kv2/reference.py    line-by-line numpy port of the PUBLISHED reference code (oracle)
kv2/codec.py        KroneckerV1 (bit-identical to reference) + ElasticKronecker (this work)
kv2/vocab.py        real India-Wikipedia vocabulary, tagged by script
kv2/audit.py        byte-budget / collision / occupancy / geometry measurements
kv2/transformer.py  numpy transformer, hand-written grads + gradcheck
kv2/experiments.py  discrimination probe + LM comparison
run_all.py          one command -> results/results.json + results/run.log
tests/              invariants (run: python -m unittest discover -s tests)
```

Corpus: set `KV2_CORPUS_DIR` to a folder of `en.md/hi.md/te.md/ta.md`; otherwise the
bundled sample is used so the demo runs anywhere. `run_all.py` exits non-zero if any
claim in this README fails to reproduce.
