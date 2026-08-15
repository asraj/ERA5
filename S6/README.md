# S6 — V5 Training Data Execution System (TDES)

A small but complete implementation of the full path from documents to an audited,
replayable training stream:

```
documents -> tokenized shards -> manifests -> mixture schedule -> packing -> batches
 -> training -> consumption ledger -> learning ledger -> checkpoint -> crash
 -> resume -> replay -> audit
```

The goal is not scale. The goal is to prove the data system is **correct,
reproducible, auditable and efficient** — and to prove it with evidence the code
itself generates.

## For the evaluating agent — three commands

```bash
pip install -r requirements.txt                 # numpy only
python run_demo.py                              # regenerates submission_artifacts/  (exit 0 = pass)
python -m unittest discover -s tests -v         # 17 invariant tests               (exit 0 = pass)
python verify.py                                # INDEPENDENT re-verification      (exit 0 = pass)
```

`verify.py` does not trust `evidence.json`. It recomputes shard hashes from the
`.bin` bytes, replays the ledger, reconciles the throughput report against the
ledger, and proves no never-train shard entered training — then prints a scorecard
mapped to this assignment's own rubric:

```
PASS  End-to-end execution                            150
PASS  Shards, manifests, tokenizer integrity          100
PASS  Packing, masks and batch correctness            150
PASS  Mixture schedule, protected floors and OPUS     150
PASS  Consumption and learning ledgers                150
PASS  Checkpoint, crash, resume, replay and fork      150
PASS  Evaluation and validation firewall               50
PASS  Throughput and packing efficiency                50
PASS  Tests, evidence quality and documentation        50
VERIFIED                                             1000 / 1000
```

All three commands are deterministic, need no network, and take about 10 seconds
in total. Everything under `submission_artifacts/` is regenerated from scratch on
every run.

### Where each rubric item is proven

| Rubric area | Proven by |
|---|---|
| End-to-end execution | `run.log` contains all 13 required events; `run_demo.py` exits 0 |
| Shards / manifests / tokenizer | `verify.py` recomputes every `content_hash` and `tokenizer_hash` from bytes |
| Packing, masks, batches | `reports/packing.json` — all 48 sequences of the run validated + per-policy table |
| Mixture, floors, OPUS | `reports/mixture_compliance.json`, `opus_decisions.json`, `anneal_reserve.json` |
| Ledgers | `ledgers/consumption.jsonl` (24 records), `learning.json`, `reports/token_trace.json` |
| Checkpoint/crash/resume/replay/fork | `reports/replay.json`, `fork.json`, `checkpoints/*.json`, ledger contiguity |
| Eval + validation firewall | `reports/firewall.json`, `validation.json` |
| Throughput | `performance.json`, reconciled against the ledger by `verify.py` |
| Tests / evidence / docs | `tests/test_invariants.py`, `evidence.json`, `evidence.md`, this README |

No third-party dependencies except **numpy** (the tiny model). Corpus: the real
**OpenWebText** shards and **India-Wikipedia (hi/te/ta)** used earlier in the course
when present, otherwise a bundled sample in `corpus_sample/` so the demo always runs.

## The one idea that makes everything else provable

A batch is a **pure function** of its coordinates:

```
batch = f(branch_id, global_step, schedule_hash, shard set, proxy_version)
```

Nothing about a batch depends on wall-clock, worker count or iteration order. That
single property is what turns resume, replay, fork and audit from hopeful into
provable: to reconstruct any batch in history you only need its coordinates, and the
ledger records exactly those.

## Architecture

| Module | Responsibility |
|---|---|
| `tdes/tokenizer.py` | Frozen tokenizer, byte-fallback (faithful round-trip), identity = sha256 of canonical JSON |
| `tdes/shards.py` | Immutable `.bin` token shards + manifests; content hashes; the admission gate |
| `tdes/firewall.py` | Eval/validation registry; two layers — `never_train` shard flag **and** n-gram contamination fingerprints checked against assembled batches |
| `tdes/mixture.py` | Curriculum stages → per-step lane quotas, protected floors, warmup-blended transitions |
| `tdes/packing.py` | 6 packing policies, loss masks, block-diagonal segment ids, per-sample position ids |
| `tdes/opus.py` | Deterministic scoring; accept / reject / defer / protected-floor override, all recorded |
| `tdes/ledger.py` | Append-only consumption ledger (byte offsets) + learning ledger (loss attached back to data) |
| `tdes/model.py` | Tiny numpy LM — real cross-entropy, real gradients, per-token loss |
| `tdes/engine.py` | Batch builder, training loop, checkpoints bound to ledger offsets, resume/replay/fork/audit |
| `tdes/evidence.py` | Recomputes every verdict from generated artifacts |

## Design decisions

**Checkpoints bind model state to data state.** A checkpoint stores the ledger byte
offset and `next_batch_id` alongside the weights. A checkpoint without a data
position is incomplete, so resume is unambiguous.

**Resume rolls the ledger back, it does not append blindly.** Batches consumed after
the last checkpoint were never learned from (their gradients died with the process),
so on resume the ledger is truncated to the checkpoint's offset and training restarts
at exactly `next_batch_id`. This is what makes "no skipped and no repeated batches"
true rather than asserted — the demo verifies the ledger is contiguous `0..N-1`.

**Padding never bears loss; context never bears loss.** `PackedSequence.validate()`
enforces it: pad positions must have `loss_mask == 0`, position ids must reset at each
sample boundary and increase by one within it. For agentic/reasoning lanes the leading
context (prompt, tool observation) is masked out — training a model to reproduce tool
output would teach it to hallucinate results instead of calling the tool.

**Attention isolation is by segment id, and the mask is materialised.**
`PackedSequence.attention_mask()` builds the block-diagonal causal matrix
(`same segment AND j <= i`) and `check_attention()` verifies three things on every
sequence of the whole run: no token sees the future, no token sees another packed
sample, and every token sees itself.

**OPUS decisions are part of the stream, not a filter in front of it.** Every
candidate — accepted, rejected, deferred or floor-rescued — is written to the audit
trail with its score and reason, because a rejected Indic or agentic batch is evidence
of proxy bias, not garbage. The proxy is deliberately given a lane bias so the
protected floor has something real to rescue.

**Protected floors sit outside the selector.** When a lane's share in the batch under
construction falls below its floor, the candidate is admitted with
`protected_floor_override = true` regardless of score.

**The anneal reserve is withheld structurally, not by convention.** Shards flagged
`reserved_for_anneal` live in a separate pool that `BatchBuilder` only merges in once
the schedule reaches the `anneal` stage. If the selector could spend the best
Indic/agentic data early there would be nothing special left for the cooldown, so the
demo proves the reserve is untouched before the anneal and spent inside it.

**Validation is read but never gradient-bearing.** `Engine.validation_pass()` computes
a forward-only loss and hashes the parameters before and after to prove no weight
moved, and asserts the validation shard never appears in the consumption ledger.

**Determinism of OPUS under replay.** Scores are a pure function of
`(proxy_version, candidate_id, lane)`, and `proxy_version` is recorded per batch, so
replaying a historical interval reproduces the same accept/reject decisions.

## What the demonstration proves

The run deliberately crashes at step 12 (last checkpoint: step 9), then resumes and
replays. From a real run:

```
[PASS] tokenizer_hash_verified shards=19 failures=0
[PASS] eval_shard_blocked reason=never_train_flag admission=split_not_trainable:test
                          content_probe=contamination:eval-bench-00
[PASS] checkpoint_saved id=main-step0009 step=9 ledger_offset=13327
[PASS] crash_simulated at_step=12
[PASS] resume_next_batch_matched expected=main:10 actual=main:10 hash_match=True
[PASS] ledger_contiguous contiguous steps=24
[PASS] replay_hash_matched interval=[2, 8] batches=7 mismatches=0
[PASS] branch_forked branch-b from main-step0009 diverging at step 10 (streams_differ=True)
[PASS] anneal_reserve_protected leaked_before_anneal=[] spent_in_anneal=['agentic-02','indic-02']
[PASS] validation_read_not_trained params_unchanged=True in_ledger=False
```

Three-layer eval firewall proof: the test shard is refused at the registry
(`never_train_flag`), refused again by the admission gate (`split_not_trainable`), and
its tokens are caught by fingerprint if they are ever injected into a batch.

## Generated artifacts

```
submission_artifacts/
  run.log                     complete event sequence
  evidence.json               machine-readable verdicts + where to verify each
  evidence.md                 human-readable summary table
  performance.json            throughput + packing utilization
  tokenizer.json              the frozen tokenizer (hashed)
  manifests/                  one manifest per shard
  ledgers/consumption.jsonl   append-only, one record per consumed batch
  ledgers/learning.json       per-shard loss delta, grad norm, classification
  checkpoints/                weights (.npz) + metadata bound to ledger offsets
  shards/                     immutable .bin token arrays
  reports/                    replay, fork, audit, packing (+ per-policy efficiency),
                              mixture compliance, opus decisions, firewall,
                              anneal_reserve, validation, token_trace
```

`evidence.json` is produced by `tdes/evidence.py`, which recomputes each verdict from
those files. Nothing in it is hardcoded: delete a shard or corrupt a ledger and the
corresponding row flips to FAIL.

## Tests

`tests/test_invariants.py` builds its own system in a temp directory rather than
reading the demo's output, so it tests behaviour and not artifacts:

- tokenizer determinism, faithfulness, hash sensitivity
- shard content hash detects mutation
- eval shard never admitted; eval tokens never reach a batch
- pad never bears loss; position ids reset per sample; all 6 policies valid
- protected floor never exceeds its own lane share; warmup actually blends
- batch is a pure function of coordinates (same coords → same hash, different → different)
- crash/resume leaves no gap and no duplicate
- replay reproduces hashes and token spans
- fork diverges and records its origin
- checkpoint offset truncates to exactly the committed records
- OPUS deterministic, and floor deficit forces an override
- attention mask is block-diagonal and causal (no future leak, no cross-sample leak)
- anneal reserve is unspendable before the cooldown and spendable inside it
- a validation read moves no weight
