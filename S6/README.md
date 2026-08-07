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

## Run it

```bash
python run_demo.py                              # full demonstration, ~5 seconds
python -m unittest discover -s tests -v         # 14 invariant tests
```

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

**Attention isolation is by segment id.** Packing several samples into one window is
only safe if they cannot see each other; `segment_ids` expresses the block-diagonal
mask, recorded in the ledger as `attention_policy`.

**OPUS decisions are part of the stream, not a filter in front of it.** Every
candidate — accepted, rejected, deferred or floor-rescued — is written to the audit
trail with its score and reason, because a rejected Indic or agentic batch is evidence
of proxy bias, not garbage. The proxy is deliberately given a lane bias so the
protected floor has something real to rescue.

**Protected floors sit outside the selector.** When a lane's share in the batch under
construction falls below its floor, the candidate is admitted with
`protected_floor_override = true` regardless of score.

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
  reports/                    replay, fork, audit, packing, mixture compliance,
                              opus decisions, firewall
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
