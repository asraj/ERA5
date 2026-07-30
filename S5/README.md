# ERA V5 — Data Mixture & Curriculum Specification

**Model:** **150B‑total sparse MoE, ~8B active** — a deliberate step up from V4's 120B (~6B active), keeping the "few‑B active" ratio that stays trainable at this token budget (~375 tokens / active‑param).
**Main pretrain budget:** **3.0T token‑instances** drawn from ~1.5T unique tokens (scarce lanes repeated ≤4×) + a **60B anneal reserve** held back for the cooldown.
**Thesis:** the mixture is the model. Every share below is sized against *real supply* from the inventory, tied to the *benchmark* it must win, and is a **hypothesis to be tested at 1B/3B proxy scale before it is trusted at full scale.** Shares that exceed real supply are labelled `REPEAT` or `SYNTH` — no wishful accounting.

> Principle: a mixture is only as trustworthy as the cleaned, deduplicated, provenance‑stamped tokens behind it — every lane draws only from manifest‑stamped shards, and the Indic‑aware tokenizer's near‑parity fertility lets scarce Indic tokens stretch further.

---

## 1. Main pretraining mixture (% of 3.0T instances)

| Lane | Share | Instances | Real unique supply | Supply verdict | Benchmarks it must win |
|---|--:|--:|---|---|---|
| General web (EN, edu‑filtered) | **33%** | 990B | FineWeb‑Edu ≥1.3T | `1×` abundant | MMLU‑Pro, general knowledge |
| Code | **20%** | 600B | The Stack v2 ~775B (permissive) | `1×` | SWE‑bench Verified, LiveCodeBench, BigCodeBench |
| Indic (all tiers, §2) | **14%** | 420B | ~90B usable real | `REPEAT+SYNTH` | MILU, IndicGenBench, IndicXTREME |
| Math & science | **12%** | 360B | OpenWebMath 15B + Proof‑Pile‑2 55B + DeepSeekMath 120B + FineMath ~50B ≈ **200B** | `1.8× REPEAT` + synth | GSM8K, MATH, AIME, GPQA‑Diamond |
| Long‑context | **6%** | 180B | books/arXiv/repo‑concat (constructed) | `1×` (built late) | RULER, MRCR‑128k |
| Reasoning (foundational traces) | **5%** | 150B | math‑CoT + OpenThoughts‑style ~40B | `SYNTH‑heavy` | feeds SFT/RLVR (Sess. 17–18) |
| India‑context English | **4%** | 120B | Indian news/law/civics/.gov.in ~100B | `~1.2× REPEAT` | India‑first custom eval |
| Multilingual + parallel | **3%** | 90B | CulturaX + EN↔Indic parallel | `1×` | cross‑lingual transfer, MT |
| Agentic (foundational tool‑use) | **3%** | 90B | ToolBench/APIGen‑MT/xLAM/tau‑train ≈ **few B** | `~90% SYNTH` | tau2‑bench, terminal‑bench, WebArena, BFCL v4 |

**100% total.** General web is largest because it is the only abundant lane. **Agentic and reasoning are deliberately small in *pretraining*** — they are taught later (SFT → RLVR); pretrain only lays the foundation and *reserves* the scarce Tier‑A trajectories for the anneal (§5). Sizing the agentic lane at 3% (not 15%) is the anti‑wishful‑accounting decision: there are only a few B real agentic tokens, so a large pretrain share would be pure synthetic padding.

---

## 2. Indic split — the required four tiers (of the 420B Indic instances)

| Tier | % of Indic | Instances | Real unique | How the gap is filled |
|---|--:|--:|---|---|
| **Verified native** | 38% | 160B | Sangraha‑verified 64B + IndicCorp v2 21B ≈ **85B** | `≤2× REPEAT` |
| **Unverified** (perplexity‑filtered) | 15% | 63B | Sangraha‑unverified ~24B | `≤3× REPEAT` |
| **Translated** (EN→Indic, MT) | 22% | 92B | generated from EN pools | machine‑translated, quality‑gated |
| **Synthetic** (native generation) | 25% | 105B | 0 real | teacher‑generated, model‑collapse‑bounded (Nemotron‑CC style) |

**Honest statement:** verified native Indic caps near **~85B unique**. A 14%/420B Indic share **cannot** be met from verified data alone — so 47% of the Indic lane is explicitly translated+synthetic, and that dependency is a first‑class risk we test in the proxy (§7), not a number we hide. This is exactly the "25% Indic can't come from verified sources" case the session names.

---

## 3. Protected always‑on floor (outside OPUS)

OPUS runs aggressively (retain ~40% of candidates, ~6× effective‑token value, few‑% overhead) — **but only above a fixed floor the selector may not cross**, because an English‑heavy proxy starves exactly the lanes we exist to build (V4: Indic at 8% always‑on).

| Protected lane | Floor per batch |
|---|--:|
| Indic | **8%** |
| Agentic | **1.5%** |
| Reasoning | **1.5%** |
| **Total protected** | **11%** of every batch |

---

## 4. Difficulty & reasoning‑length bands (with a concrete example each)

**Difficulty ladder** (applied within every stage):

| Level | Example |
|---|---|
| L1 easy | "12 + 7 = ?"; a one‑sentence factual passage |
| L2 medium | a GSM8K word problem; fix a one‑line bug in a function |
| L3 hard | a MATH competition problem; a SWE‑bench repo patch across files |
| L4 advanced | an AIME/olympiad problem; a multi‑file agentic refactor that must pass tests |

**Reasoning‑length bands** (the *distribution* the reasoning lane reserves — short is the foundation):

| Effort | Trace length | Mix | Example |
|---|---|--:|---|
| low | <128 tok | 40% | "Q: 2+2? A: 4." (direct) |
| medium | 128–512 | 30% | GSM8K with a 3–4 step chain |
| high | 512–2048 | 20% | MATH problem that verifies intermediate steps and considers alternatives |
| ultra | 2048–8192+ | 10% | AIME problem with self‑correction and multiple attempted approaches |

The effort dial is *learned*, not created at inference: the model can only emit a depth it was trained on, so the reserve must span all four bands across math, code and general problem‑solving.

---

## 5. Anneal reserve (held back, ~60B ≈ 2% of the run)

A separate preset, **not sampled during the main run** so the selector cannot spend it early. Reduced learning rate, disproportionate capability gain (cf. OLMo‑2 GSM8K 24→67% on a tiny reserve).

| Anneal lane | Share | Contents held back |
|---|--:|---|
| Math & reasoning | 30% | hardest MATH/AIME, long verified reasoning traces |
| Indic (verified best) | 25% | cleanest verified‑native Indic, exam‑grade |
| Code | 20% | SWE‑bench‑shaped edit/patch data, tests |
| Agentic (Tier‑A) | 15% | long multi‑step trajectories (plan→call→observe→recover→answer) |
| General web (top‑edu) | 10% | highest edu‑value web |

Agentic Tier‑A trajectories are scarce, expensive, and **non‑recoverable once spent** — reserving them here is a Session‑5 allocation decision, not a post‑hoc discovery. Tool‑observation tokens carry **no loss** (masking rule) so the model never learns to hallucinate tool results.

---

## 6. Curriculum (order) & stability

| Stage | Span | Emphasis (general → scarce) |
|---|---|---|
| A · foundation | 0–40% (1.2T) | general web 55% · Indic 12% · code 10% · math 8% · L1–L2 difficulty |
| B · capability | 40–80% (1.2T) | web 25% · code 28% · math 16% · reasoning 10% · Indic 14% · L2–L3 |
| C · long‑context | 80–95% (450B) | long‑context 20% · code 20% · reasoning 12% · Indic 14% · web 20% · L3 |
| D · anneal | 95–100% (60B) | the §5 reserve · L3–L4 |

**Stability:** every mixture transition (and each growth‑stage boundary) is blended over a **~5B‑token warmup band**, never a hard step. V4 saw a sudden Hindi‑share jump interact with frozen embeddings and spike gradient‑norm ~150×; warmup + gradient‑norm monitoring is mandatory at each transition. Architecture and mixture are frozen before the main run.

---

## 7. The hypothesis: proxy experiment before full scale

Every number above is a hypothesis. **Nothing is trusted at the 150B run until it survives 1B and 3B proxy runs.**

- **Runs:** train 1B and 3B proxies on **~25B tokens** each, over three mixtures — `A` web‑heavy baseline, `B` this proposed mixture, `C` an Indic‑heavy variant (25% Indic).
- **Primary metric:** per‑domain **held‑out loss**, with an explicit **Indic‑script mean**; benchmarks confirmatory behind the decontamination firewall.
- **Secondary battery:** GSM8K, HumanEval+MBPP, MILU‑lite, a BFCL tool‑call subset, RULER‑lite (long‑context).
- **Confirm** `B` if it improves Indic‑script loss and MILU‑lite by ≥ the noise band **and** matches `A` on English held‑out loss within **1%** **and** improves code+math proxy scores; **refute → reallocate**.
- **Required ablations:** (a) toggle the always‑on floor — expect Indic/agentic to collapse when off (reproduces the OPUS lesson); (b) with/without the anneal reserve; (c) Indic verified‑only vs verified+synthetic, to measure synthetic's marginal value and detect model‑collapse. Rank recipes; only the winner scales.

---

## 8. Supply‑honesty summary (what the reviewer will push on)

- **Reachable at 1× (real data):** general web, code, long‑context, multilingual.
- **Reachable only by repetition (≤4×):** math/science, verified & unverified Indic, India‑context English.
- **Reachable only by generation (SYNTH):** the agentic lane (~90%), Indic synthetic tier (25% of Indic), much of the foundational reasoning lane.
- **The two structurally scarce lanes — Indic‑verified and agentic — are protected by the floor and by the anneal reserve, and are the top priority for continued cleaning/collection.** The mixture doesn't hide their scarcity; it budgets for it and tests it.

---

*Cleaning continues toward the cumulative target, now aimed at the starved slots this mixture exposes: verified‑native Indic and long agentic trajectories.*
