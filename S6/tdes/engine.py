"""The execution engine: deterministic batch construction, training, checkpoints
bound to ledger offsets, crash/resume, replay, fork and audit.

The central invariant: a batch is a PURE FUNCTION of
    (branch_id, global_step, schedule_hash, shard set, proxy_version)
so the same coordinates always rebuild byte-identical tokens, masks and hashes.
That single property is what makes resume, replay and audit provable rather than
hopeful.
"""
from __future__ import annotations
import os, json, hashlib, random, time
import numpy as np

from .packing import Packer
from .model import TinyLM


class CrashSignal(Exception):
    """Deliberate, simulated crash used by the demonstration."""


def _rng_for(branch: str, step: int, schedule_hash: str) -> random.Random:
    seed = int(hashlib.sha256(f"{branch}|{step}|{schedule_hash}".encode()).hexdigest()[:16], 16)
    return random.Random(seed)


class BuiltBatch:
    def __init__(self, batch_id, step, branch, stage, sequences, samples, decisions):
        self.batch_id, self.step, self.branch = batch_id, step, branch
        self.stage, self.sequences, self.samples = stage, sequences, samples
        self.decisions = decisions

    @property
    def shard_ids(self):
        return sorted({s["shard_id"] for s in self.samples})

    @property
    def sample_ids(self):
        return [s["sample_id"] for s in self.samples]

    @property
    def token_span_ids(self):
        return [s["span_id"] for s in self.samples]

    @property
    def lanes(self):
        return [s["lane"] for s in self.samples]

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.batch_id.encode())
        for sq in self.sequences:
            h.update(sq.hash().encode())
        h.update(bytes(str(self.token_span_ids), "utf-8"))
        return h.hexdigest()

    @property
    def useful_tokens(self):
        return sum(sq.useful_tokens for sq in self.sequences)

    @property
    def total_positions(self):
        return sum(len(sq.tokens) for sq in self.sequences)


class BatchBuilder:
    def __init__(self, store, schedule, firewall, tokenizer, opus, *,
                 microbatch: int = 2, samples_per_seq: int = 4):
        self.store, self.schedule, self.firewall = store, schedule, firewall
        self.tok, self.opus = tokenizer, opus
        self.microbatch, self.samples_per_seq = microbatch, samples_per_seq
        self.packer = Packer(schedule.stages[0].sequence_length, tokenizer.pad_id, tokenizer.eos_id)
        # lane -> [(shard_id, span_idx)]. Two pools: the ordinary stream, and the
        # anneal reserve which the main run is NOT allowed to spend. If the selector
        # could consume the best Indic/agentic data early there would be nothing
        # special left for the cooldown, so the reserve is withheld structurally.
        self.pools: dict[str, list] = {}
        self.reserve_pools: dict[str, list] = {}
        self.reserved_shards: set = set()
        for sid, m in sorted(store.manifests.items()):
            if m.split != "train":
                continue
            ok, _ = firewall.check_shard(sid)
            if not ok:
                continue
            target = self.reserve_pools if m.reserved_for_anneal else self.pools
            target.setdefault(m.capability_lane, []).extend(
                (sid, i) for i in range(len(m.spans)))
            if m.reserved_for_anneal:
                self.reserved_shards.add(sid)

    def _pools_for(self, stage_name: str) -> dict:
        """The reserve only becomes spendable during the anneal stage."""
        if stage_name != "anneal":
            return self.pools
        merged = {l: list(v) for l, v in self.pools.items()}
        for l, v in self.reserve_pools.items():
            merged.setdefault(l, []).extend(v)
        return merged

    def _candidate(self, rng, lane, pools) -> dict:
        pool = pools[lane]
        sid, si = pool[rng.randrange(len(pool))]
        span = self.store.manifests[sid].spans[si]
        toks = self.store.span_tokens(sid, si)
        ctx = 0
        if lane in ("agentic", "reasoning"):
            ctx = min(len(toks) // 3, max(1, len(toks) - 2))   # observation/prompt = context only
        return {"tokens": toks, "sample_id": f"{sid}#{si}", "span_id": f"{sid}:{span['start']}+{span['length']}",
                "shard_id": sid, "lane": lane, "ctx_len": ctx}

    def build(self, branch: str, step: int, proxy_version: str) -> BuiltBatch:
        weights, stage = self.schedule.weights_for(step)
        rng = _rng_for(branch, step, self.schedule.hash)
        pools = self._pools_for(stage.name)
        lanes = [l for l in sorted(weights) if l in pools]
        wts = [weights[l] for l in lanes]
        need = self.microbatch * self.samples_per_seq

        accepted, decisions, lane_counts = [], [], {l: 0 for l in lanes}
        tries = 0
        while len(accepted) < need and tries < need * 12:
            tries += 1
            lane = rng.choices(lanes, weights=wts, k=1)[0]
            cand = self._candidate(rng, lane, pools)
            cid = f"{branch}:{step}:{tries}"
            floors = stage.protected_floors
            share_now = lane_counts.get(lane, 0) / need
            deficit = lane in floors and share_now < floors[lane]
            d = self.opus.evaluate(cid, lane=lane, stage=stage.name,
                                   shard_ids=[cand["shard_id"]], proxy_version=proxy_version,
                                   effective_tokens=len(cand["tokens"]), floor_deficit=deficit)
            decisions.append(d)
            if d.status in ("accepted", "protected_override"):
                ok, _ = self.firewall.check_shard(cand["shard_id"])
                if ok:
                    accepted.append(cand); lane_counts[lane] += 1

        sequences = []
        for k in range(self.microbatch):
            chunk = accepted[k * self.samples_per_seq:(k + 1) * self.samples_per_seq]
            if not chunk:
                chunk = accepted[:1]
            sq = self.packer.pack(chunk, stage.packing_policy)
            sq.validate(self.packer.seq_len, self.tok.pad_id)
            sequences.append(sq)
        return BuiltBatch(f"{branch}:{step}", step, branch, stage.name,
                          sequences, accepted[:need], decisions)


class Engine:
    def __init__(self, root, store, schedule, firewall, tokenizer, opus,
                 cledger, lledger, *, microbatch=2, samples_per_seq=4, seed=0, log=print):
        self.root, self.store, self.schedule = root, store, schedule
        self.firewall, self.tok, self.opus = firewall, tokenizer, opus
        self.cl, self.ll, self.log = cledger, lledger, log
        self.builder = BatchBuilder(store, schedule, firewall, tokenizer, opus,
                                    microbatch=microbatch, samples_per_seq=samples_per_seq)
        self.model = TinyLM(len(tokenizer), seed=seed)
        self.ckpt_dir = os.path.join(root, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # per-step accounting keyed by (branch, step). Batches consumed after the
        # last checkpoint are rolled back on resume, so throughput must be summed
        # over the COMMITTED ledger only - otherwise the report double-counts the
        # work the crash threw away.
        self.step_perf: dict = {}

    # ---------- checkpointing ----------
    def save_checkpoint(self, branch: str, step: int) -> dict:
        ck_id = f"{branch}-step{step:04d}"
        path = os.path.join(self.ckpt_dir, ck_id + ".npz")
        np.savez(path, **self.model.state())
        meta = {"checkpoint_id": ck_id, "branch_id": branch, "global_step": step,
                "ledger_offset": self.cl.offset(), "next_batch_id": f"{branch}:{step+1}",
                "param_hash": self.model.param_hash(), "tokenizer_hash": self.tok.hash,
                "schedule_hash": self.schedule.hash, "weights_file": os.path.basename(path)}
        with open(os.path.join(self.ckpt_dir, ck_id + ".json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self.log(f"[PASS] checkpoint_saved id={ck_id} step={step} ledger_offset={meta['ledger_offset']}")
        return meta

    def load_checkpoint(self, ck_id: str) -> dict:
        with open(os.path.join(self.ckpt_dir, ck_id + ".json")) as fh:
            meta = json.load(fh)
        z = np.load(os.path.join(self.ckpt_dir, meta["weights_file"]))
        self.model.load({"E": z["E"], "W": z["W"], "b": z["b"]})
        assert self.model.param_hash() == meta["param_hash"], "checkpoint weights corrupted"
        return meta

    # ---------- the training loop ----------
    def run(self, branch: str, start_step: int, end_step: int, *,
            checkpoint_every: int = 5, crash_at: int | None = None,
            proxy_of=lambda step: f"proxy-v1@ckpt{step//5*5}") -> dict:
        last_ckpt = None
        for step in range(start_step, end_step):
            t0 = time.perf_counter()
            pv = proxy_of(step)
            batch = self.builder.build(branch, step, pv)

            # --- eval firewall, content layer: no test tokens may reach the optimizer
            for sq in batch.sequences:
                ok, why = self.firewall.check_tokens(sq.tokens, f"{branch}:{step}")
                if not ok:
                    raise RuntimeError(f"firewall breach {why}")

            losses, grads, tops = [], [], []
            for sq in batch.sequences:
                ml, per_tok, gn = self.model.step(sq.tokens, sq.loss_mask, sq.segment_ids)
                losses.append(ml); grads.append(gn)
                tops += TinyLM.top_perplexity(per_tok, sq.tokens)
            dt = time.perf_counter() - t0
            mean_loss = float(np.mean(losses)) if losses else 0.0
            gnorm = float(np.max(grads)) if grads else 0.0

            pads = sum(1 for sq in batch.sequences for t in sq.tokens if t == self.tok.pad_id)
            self.step_perf[(branch, step)] = {
                "positions": batch.total_positions, "useful_tokens": batch.useful_tokens,
                "pad_positions": pads, "seconds": dt}

            acc = [d for d in batch.decisions if d.status in ("accepted", "protected_override")]
            rec = {
                "run_id": os.path.basename(self.root), "branch_id": branch,
                "global_step": step, "batch_id": batch.batch_id,
                "checkpoint_id": last_ckpt["checkpoint_id"] if last_ckpt else None,
                "rank": 0, "microbatch_ids": [f"{batch.batch_id}/mb{i}" for i in range(len(batch.sequences))],
                "packed_sample_ids": batch.sample_ids, "shard_ids": batch.shard_ids,
                "token_span_ids": batch.token_span_ids, "lanes": batch.lanes,
                "curriculum_stage": batch.stage,
                "loss_mask_hash": hashlib.sha256(
                    bytes(str([sq.loss_mask for sq in batch.sequences]), "utf-8")).hexdigest()[:32],
                "attention_policy": "block_diagonal_by_segment", "position_policy": "reset_per_sample",
                "batch_hash": batch.hash(), "tokenizer_version": self.tok.hash[:16],
                "dataloader_version": "tdes/1.0", "proxy_version": pv,
                "opus_decision_ids": [d.candidate_id for d in acc],
                "mean_loss": round(mean_loss, 6), "useful_tokens": batch.useful_tokens,
                "total_positions": batch.total_positions,
            }
            self.cl.append(rec)
            self.ll.record(step=step, stage=batch.stage, shard_ids=batch.shard_ids,
                           lane=(batch.lanes[0] if batch.lanes else "mixed"),
                           mean_loss=mean_loss, useful_tokens=batch.useful_tokens,
                           top_ppl=tops, grad_norm=gnorm,
                           opus_score=float(np.mean([d.score for d in acc])) if acc else 0.0)

            if (step + 1) % checkpoint_every == 0:
                last_ckpt = self.save_checkpoint(branch, step)
            if crash_at is not None and step == crash_at:
                self.log(f"[EVENT] crash_simulated at step={step} (uncommitted since "
                         f"checkpoint {last_ckpt['checkpoint_id'] if last_ckpt else 'none'})")
                raise CrashSignal(str(step))
        return {"last_checkpoint": last_ckpt, "final_step": end_step - 1}

    # ---------- validation: read, never trained on ----------
    def validation_pass(self, shard_id: str, seq_len: int = 192) -> dict:
        """Validation data may be READ for evaluation but must never become
        gradient-bearing. We prove that by hashing the parameters before and
        after: the loss is computed, no weight moves."""
        before = self.model.param_hash()
        toks = self.store.tokens(shard_id)[:seq_len]
        segs = [0] * len(toks)
        mask = [1] * len(toks)
        loss = self.model.evaluate(toks, mask, segs)      # forward only
        after = self.model.param_hash()
        self.firewall.note_access(shard_id, "validation_eval")
        return {"shard_id": shard_id, "val_loss": round(loss, 6),
                "params_unchanged": before == after,
                "gradient_bearing": False,
                "appears_in_consumption_ledger":
                    any(shard_id in r["shard_ids"] for r in self.cl.read())}

    def token_trace(self, branch: str, step: int, proxy_version: str, limit: int = 60) -> dict:
        """Token-level learning signal for one batch: loss and perplexity per
        loss-bearing position, tied back to its shard and lane."""
        b = self.builder.build(branch, step, proxy_version)
        sq = b.sequences[0]
        _, per_tok, _ = self.model.step(sq.tokens, sq.loss_mask, sq.segment_ids, lr=0.0)
        rows = []
        for pos in sorted(per_tok)[:limit]:
            seg = sq.segment_ids[pos]
            rows.append({"position": pos, "token_id": int(sq.tokens[pos]),
                         "decoded": self.tok.decode([sq.tokens[pos]])[:24],
                         "segment": int(seg),
                         "sample_id": sq.sample_ids[seg] if seg < len(sq.sample_ids) else None,
                         "lane": sq.lanes[seg] if seg < len(sq.lanes) else None,
                         "loss": round(per_tok[pos], 5),
                         "perplexity": round(float(np.exp(min(per_tok[pos], 20))), 3),
                         "loss_bearing": True})
        return {"batch_id": b.batch_id, "stage": b.stage,
                "loss_bearing_tokens": len(per_tok), "shown": len(rows), "tokens": rows}

    # ---------- replay / fork / audit ----------
    def replay(self, branch: str, lo: int, hi: int) -> dict:
        """Rebuild a historical interval and compare against what was recorded."""
        recs = {r["global_step"]: r for r in self.cl.interval(branch, lo, hi)}
        checked, mismatches = [], []
        for step, r in sorted(recs.items()):
            rebuilt = self.builder.build(branch, step, r["proxy_version"])
            same = (rebuilt.hash() == r["batch_hash"]
                    and rebuilt.batch_id == r["batch_id"]
                    and rebuilt.token_span_ids == r["token_span_ids"])
            checked.append({"step": step, "batch_id": r["batch_id"],
                            "original_hash": r["batch_hash"], "replay_hash": rebuilt.hash(),
                            "token_spans_match": rebuilt.token_span_ids == r["token_span_ids"],
                            "match": same})
            if not same:
                mismatches.append(step)
        return {"interval": [lo, hi], "checked": checked,
                "all_match": not mismatches, "mismatches": mismatches}

    def fork(self, from_ckpt: str, new_branch: str) -> dict:
        meta = self.load_checkpoint(from_ckpt)
        step = meta["global_step"]
        old = self.builder.build(meta["branch_id"], step + 1, "proxy-v1@fork")
        new = self.builder.build(new_branch, step + 1, "proxy-v1@fork")
        info = {"origin_branch": meta["branch_id"], "origin_checkpoint": from_ckpt,
                "divergence_step": step + 1, "new_branch": new_branch,
                "origin_next_hash": old.hash(), "fork_next_hash": new.hash(),
                "streams_differ": old.hash() != new.hash()}
        self.log(f"[PASS] branch_forked {new_branch} from {from_ckpt} "
                 f"diverging at step {step+1} (streams_differ={info['streams_differ']})")
        return info

    def audit(self, branch: str, lo: int, hi: int) -> dict:
        recs = self.cl.interval(branch, lo, hi)
        lanes = {}
        for r in recs:
            for l in r["lanes"]:
                lanes[l] = lanes.get(l, 0) + 1
        return {"interval": [lo, hi], "batches": len(recs),
                "shards_influencing": self.cl.shards_between(branch, lo, hi),
                "lane_exposure": dict(sorted(lanes.items())),
                "stages": sorted({r["curriculum_stage"] for r in recs})}

    def performance(self) -> dict:
        """Throughput over the COMMITTED stream: summed only over batches that
        survive in the consumption ledger, so the report reconciles exactly with
        the ledger even after a crash rolled work back."""
        committed = {(r["branch_id"], r["global_step"]) for r in self.cl.read()}
        keys = [k for k in self.step_perf if k in committed]
        pos = sum(self.step_perf[k]["positions"] for k in keys)
        use = sum(self.step_perf[k]["useful_tokens"] for k in keys)
        pad = sum(self.step_perf[k]["pad_positions"] for k in keys)
        sec = sum(self.step_perf[k]["seconds"] for k in keys)
        discarded = len(self.step_perf) - len(keys)
        s = max(sec, 1e-9)
        return {
            "batches": len(keys),
            "batches_discarded_by_crash": discarded,
            "wall_seconds": round(sec, 4),
            "total_positions": pos,
            "useful_loss_bearing_tokens": use,
            "pad_positions": pad,
            "packing_utilization": round(1 - pad / max(pos, 1), 4),
            "raw_tokens_per_sec": round(pos / s, 1),
            "useful_loss_bearing_tokens_per_sec": round(use / s, 1),
        }
