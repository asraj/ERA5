# Evidence Bundle — V5 Training Data Execution System

**Overall: PASS** (12/12 checks passed)

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | **PASS** | `submission_artifacts/manifests/*.manifest.json` |
| Shard immutability | **PASS** | `content hash recomputed over mutated bytes` |
| Evaluation firewall | **PASS** | `submission_artifacts/reports/firewall.json` |
| Crash recovery | **PASS** | `submission_artifacts/ledgers/consumption.jsonl + checkpoints/*.json` |
| Replay | **PASS** | `submission_artifacts/reports/replay.json` |
| Fork from earlier checkpoint | **PASS** | `submission_artifacts/reports/fork.json` |
| Packing correctness | **PASS** | `submission_artifacts/reports/packing.json` |
| Mixture compliance | **PASS** | `submission_artifacts/reports/mixture_compliance.json` |
| OPUS audit trail | **PASS** | `submission_artifacts/reports/opus_decisions.json` |
| Learning trace | **PASS** | `submission_artifacts/ledgers/learning.json` |
| Throughput | **PASS** | `submission_artifacts/performance.json` |
| Consumption ledger | **PASS** | `submission_artifacts/ledgers/consumption.jsonl` |

## Key numbers

- **corpus_source**: real:openwebtext+india-wikipedia
- **documents**: 233
- **vocab**: 4096
- **tokenizer_hash**: 03c3294cb3e7683378a694726727eb0a
- **schedule_hash**: a134a1cf86f3b63c
- **steps**: 24
- **crash_at**: 12
- **resumed_from**: main-step0009
- **useful_loss_bearing_tokens_per_sec**: 12757.2
- **packing_utilization**: 0.7571

## Detail

**Tokenizer integrity** — `tokenizer_hash`=03c3294cb3e7683378a694726727eb0a, `shards_verified`=19, `content_hash_failures`=0, `roundtrip_faithful`=True
**Shard immutability** — `original`=95dd000fee267b81, `mutated`=213dcda0da1bd408
**Evaluation firewall** — `shard_blocked_reason`=never_train_flag, `admission_gate`=split_not_trainable:test, `content_probe`=contamination:eval-bench-00, `fingerprints`=107
**Crash recovery** — `crash_at`=12, `resumed_from`=main-step0009, `expected_next_batch`=main:10, `actual_next_batch`=main:10, `hash_match`=True, `ledger_contiguous`=contiguous, `steps`=0..23, `duplicates_or_gaps`=False
**Replay** — `interval`=[2, 8], `batches_checked`=7, `mismatches`=[]
**Fork from earlier checkpoint** — `origin_branch`=main, `origin_checkpoint`=main-step0009, `divergence_step`=10, `new_branch`=branch-b, `origin_next_hash`=abffc3510ae4fb3cd3efc6617c61b14d5195fd7c7be349fc4aee6efa77e3a8f4, `fork_next_hash`=260411b937e374eeeaee4791a2cd795d72256e1cfabd113d3df1ae3b9774a38a, `streams_differ`=True
**Packing correctness** — `utilization`=1.0, `useful_loss_bearing_tokens`=384, `total_positions`=384, `context_tokens_masked_from_loss`=False
**Mixture compliance** — `planned_share`={'agentic': 0.0875, 'code': 0.1875, 'general_web': 0.3937, 'indic': 0.2125, 'reasoning': 0.1188}, `actual_share`={'agentic': 0.0938, 'code': 0.1615, 'general_web': 0.3542, 'indic': 0.2448, 'reasoning': 0.1458}, `protected_floors`={'indic': 0.08, 'agentic': 0.015, 'reasoning': 0.015}, `floors_met`={'indic': True, 'agentic': True, 'reasoning': True}, `max_abs_deviation`=0.0395
**OPUS audit trail** — `rejected`=110, `protected_override`=49, `accepted`=255, `deferred`=13, `total`=427
**Learning trace** — `first_loss`=8.3161, `last_loss`=8.1939, `shards_with_token_level_trace`=16, `shards_scored`=16, `classifications`=['neutral', 'useful']
**Throughput** — `batches`=27, `wall_seconds`=0.5667, `total_positions`=10368, `useful_loss_bearing_tokens`=7230, `pad_positions`=2518, `packing_utilization`=0.7571, `raw_tokens_per_sec`=18294.2, `useful_loss_bearing_tokens_per_sec`=12757.2
**Consumption ledger** — `records`=24, `fields_per_record`=23
