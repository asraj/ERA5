# Evidence Bundle — V5 Training Data Execution System

**Overall: PASS** (15/15 checks passed)

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | **PASS** | `submission_artifacts/manifests/*.manifest.json` |
| Shard immutability | **PASS** | `content hash recomputed over mutated bytes` |
| Evaluation firewall | **PASS** | `submission_artifacts/reports/firewall.json` |
| Crash recovery | **PASS** | `submission_artifacts/ledgers/consumption.jsonl + checkpoints/*.json` |
| Replay | **PASS** | `submission_artifacts/reports/replay.json` |
| Fork from earlier checkpoint | **PASS** | `submission_artifacts/reports/fork.json` |
| Packing correctness | **PASS** | `submission_artifacts/reports/packing.json` |
| Attention & position masks | **PASS** | `submission_artifacts/reports/packing.json` |
| Mixture compliance | **PASS** | `submission_artifacts/reports/mixture_compliance.json` |
| Anneal reserve held back | **PASS** | `submission_artifacts/reports/anneal_reserve.json` |
| Validation firewall (read, never trained) | **PASS** | `submission_artifacts/reports/validation.json` |
| OPUS audit trail | **PASS** | `submission_artifacts/reports/opus_decisions.json` |
| Learning trace | **PASS** | `submission_artifacts/ledgers/learning.json` |
| Throughput | **PASS** | `submission_artifacts/performance.json` |
| Consumption ledger | **PASS** | `submission_artifacts/ledgers/consumption.jsonl` |

## Key numbers

- **corpus_source**: bundled_sample
- **documents**: 310
- **vocab**: 828
- **tokenizer_hash**: 859ab002083519f11653509c26ba2e06
- **schedule_hash**: a134a1cf86f3b63c
- **steps**: 24
- **crash_at**: 12
- **resumed_from**: main-step0009
- **useful_loss_bearing_tokens_per_sec**: 48163.1
- **packing_utilization**: 0.885

## Detail

**Tokenizer integrity** — `tokenizer_hash`=859ab002083519f11653509c26ba2e06, `shards_verified`=20, `content_hash_failures`=0, `roundtrip_faithful`=True
**Shard immutability** — `original`=999441b6a81877a9, `mutated`=02becc71e3db6684
**Evaluation firewall** — `shard_blocked_reason`=never_train_flag, `admission_gate`=split_not_trainable:test, `content_probe`=contamination:eval-bench-00, `fingerprints`=41
**Crash recovery** — `crash_at`=12, `resumed_from`=main-step0009, `expected_next_batch`=main:10, `actual_next_batch`=main:10, `hash_match`=True, `ledger_contiguous`=contiguous, `steps`=0..23, `duplicates_or_gaps`=False
**Replay** — `interval`=[2, 8], `batches_checked`=7, `mismatches`=[]
**Fork from earlier checkpoint** — `origin_branch`=main, `origin_checkpoint`=main-step0009, `divergence_step`=10, `new_branch`=branch-b, `origin_next_hash`=d363fa1f60abc4d0342c2814029411039425c063762c0c968a391277e0598c5c, `fork_next_hash`=83a18ea349f13dd9bb475c20fda2dd7b82d42cdcc5a7dcf9a4e632cb406e7a53, `streams_differ`=True
**Packing correctness** — `utilization`=0.8021, `useful_loss_bearing_tokens`=293, `total_positions`=384, `context_tokens_masked_from_loss`=True, `attention_check`=block_diagonal_causal_ok, `sequences_validated_across_whole_run`=48, `all_sequences_valid`=True
**Attention & position masks** — `block_diagonal_causal`=block_diagonal_causal_ok, `sequences_checked`=48, `policies_compared`=6
**Mixture compliance** — `planned_share`={'agentic': 0.0875, 'code': 0.1875, 'general_web': 0.3937, 'indic': 0.2125, 'reasoning': 0.1188}, `actual_share`={'agentic': 0.0885, 'code': 0.1719, 'general_web': 0.3438, 'indic': 0.2448, 'reasoning': 0.151}, `protected_floors`={'indic': 0.08, 'agentic': 0.015, 'reasoning': 0.015}, `floors_met`={'indic': True, 'agentic': True, 'reasoning': True}, `max_abs_deviation`=0.0499
**Anneal reserve held back** — `reserved_shards`=['agentic-02', 'indic-03'], `anneal_starts_at_step`=21, `spent_before_anneal`=[], `spent_during_anneal`=['agentic-02'], `held_back_correctly`=True
**Validation firewall (read, never trained)** — `shard_id`=valid-00, `val_loss`=6.562878, `params_unchanged`=True, `gradient_bearing`=False, `appears_in_consumption_ledger`=False
**OPUS audit trail** — `rejected`=187, `protected_override`=70, `accepted`=434, `deferred`=30, `total`=721
**Learning trace** — `first_loss`=6.7148, `last_loss`=6.632, `shards_with_token_level_trace`=16, `shards_scored`=16, `classifications`=['neutral', 'useful']
**Throughput** — `batches`=24, `batches_discarded_by_crash`=0, `wall_seconds`=0.1564, `total_positions`=9216, `useful_loss_bearing_tokens`=7534, `pad_positions`=1060, `packing_utilization`=0.885, `raw_tokens_per_sec`=58915.7, `useful_loss_bearing_tokens_per_sec`=48163.1
**Consumption ledger** — `records`=24, `fields_per_record`=23
