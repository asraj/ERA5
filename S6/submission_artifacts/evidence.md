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

- **corpus_source**: real:openwebtext+india-wikipedia
- **documents**: 233
- **vocab**: 4096
- **tokenizer_hash**: 03c3294cb3e7683378a694726727eb0a
- **schedule_hash**: a134a1cf86f3b63c
- **steps**: 24
- **crash_at**: 12
- **resumed_from**: main-step0009
- **useful_loss_bearing_tokens_per_sec**: 13091.5
- **packing_utilization**: 0.7788

## Detail

**Tokenizer integrity** — `tokenizer_hash`=03c3294cb3e7683378a694726727eb0a, `shards_verified`=19, `content_hash_failures`=0, `roundtrip_faithful`=True
**Shard immutability** — `original`=95dd000fee267b81, `mutated`=213dcda0da1bd408
**Evaluation firewall** — `shard_blocked_reason`=never_train_flag, `admission_gate`=split_not_trainable:test, `content_probe`=contamination:eval-bench-00, `fingerprints`=107
**Crash recovery** — `crash_at`=12, `resumed_from`=main-step0009, `expected_next_batch`=main:10, `actual_next_batch`=main:10, `hash_match`=True, `ledger_contiguous`=contiguous, `steps`=0..23, `duplicates_or_gaps`=False
**Replay** — `interval`=[2, 8], `batches_checked`=7, `mismatches`=[]
**Fork from earlier checkpoint** — `origin_branch`=main, `origin_checkpoint`=main-step0009, `divergence_step`=10, `new_branch`=branch-b, `origin_next_hash`=09570b2bbc3f98eecb6557273729996ea09f7d7f3e066e4956e211901f2a16b9, `fork_next_hash`=3a6a911822906c4b7eb03bc839a6c03a8ea8faa61c0bee249893993d520c8b2e, `streams_differ`=True
**Packing correctness** — `utilization`=0.9219, `useful_loss_bearing_tokens`=354, `total_positions`=384, `context_tokens_masked_from_loss`=False, `attention_check`=block_diagonal_causal_ok, `sequences_validated_across_whole_run`=48, `all_sequences_valid`=True
**Attention & position masks** — `block_diagonal_causal`=block_diagonal_causal_ok, `sequences_checked`=48, `policies_compared`=6
**Mixture compliance** — `planned_share`={'agentic': 0.0875, 'code': 0.1875, 'general_web': 0.3937, 'indic': 0.2125, 'reasoning': 0.1188}, `actual_share`={'agentic': 0.0938, 'code': 0.1615, 'general_web': 0.3438, 'indic': 0.2396, 'reasoning': 0.1615}, `protected_floors`={'indic': 0.08, 'agentic': 0.015, 'reasoning': 0.015}, `floors_met`={'indic': True, 'agentic': True, 'reasoning': True}, `max_abs_deviation`=0.0499
**Anneal reserve held back** — `reserved_shards`=['agentic-02', 'indic-02'], `anneal_starts_at_step`=21, `spent_before_anneal`=[], `spent_during_anneal`=['agentic-02', 'indic-02'], `held_back_correctly`=True
**Validation firewall (read, never trained)** — `shard_id`=valid-00, `val_loss`=8.214162, `params_unchanged`=True, `gradient_bearing`=False, `appears_in_consumption_ledger`=False
**OPUS audit trail** — `rejected`=184, `protected_override`=76, `accepted`=428, `deferred`=30, `total`=718
**Learning trace** — `first_loss`=8.3161, `last_loss`=8.1872, `shards_with_token_level_trace`=16, `shards_scored`=16, `classifications`=['neutral', 'useful']
**Throughput** — `batches`=27, `wall_seconds`=0.5669, `total_positions`=10368, `useful_loss_bearing_tokens`=7422, `pad_positions`=2293, `packing_utilization`=0.7788, `raw_tokens_per_sec`=18287.8, `useful_loss_bearing_tokens_per_sec`=13091.5
**Consumption ledger** — `records`=24, `fields_per_record`=23
