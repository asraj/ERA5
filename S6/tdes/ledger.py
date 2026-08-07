"""Append-only ledgers.

ConsumptionLedger - what the run actually consumed (the run's memory). Every
record is one served batch, with enough provenance to reconstruct it exactly.
Offsets are byte offsets into the JSONL file, so a checkpoint can bind model
state to an exact data position and a resume can roll back uncommitted events.

LearningLedger - what the model got out of it: loss attached back to the source
data (shard / lane / token cluster) so V5 can teach V6 what to collect.
"""
from __future__ import annotations
import os, json


class ConsumptionLedger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    # ---- append-only write ----
    def append(self, rec: dict) -> int:
        """Returns the byte offset AFTER this record (the resumable position)."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            return f.tell()

    def offset(self) -> int:
        return os.path.getsize(self.path)

    def read(self, branch: str | None = None) -> list:
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if branch is None or r.get("branch_id") == branch:
                    out.append(r)
        return out

    def rollback_to(self, offset: int) -> int:
        """Crash recovery: discard events written after the checkpoint's offset,
        so no batch is double-counted on resume."""
        cur = self.offset()
        if offset < cur:
            with open(self.path, "r+b") as f:
                f.truncate(offset)
        return cur - offset

    # ---- audit ----
    def interval(self, branch: str, lo: int, hi: int) -> list:
        return [r for r in self.read(branch) if lo <= r["global_step"] <= hi]

    def contiguous(self, branch: str) -> tuple[bool, str]:
        steps = [r["global_step"] for r in self.read(branch)]
        if steps != sorted(steps):
            return False, "out_of_order"
        if len(steps) != len(set(steps)):
            return False, "duplicate_batch"
        for a, b in zip(steps, steps[1:]):
            if b != a + 1:
                return False, f"gap_between_{a}_and_{b}"
        return True, "contiguous"

    def shards_between(self, branch: str, lo: int, hi: int) -> dict:
        """Which shards influenced the model over a step range (audit question)."""
        agg = {}
        for r in self.interval(branch, lo, hi):
            for sid in r["shard_ids"]:
                agg[sid] = agg.get(sid, 0) + 1
        return dict(sorted(agg.items()))


class LearningLedger:
    """Attaches the training outcome back to the data that produced it."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.by_shard: dict[str, dict] = {}
        self.by_lane: dict[str, dict] = {}

    def record(self, *, step: int, stage: str, shard_ids: list, lane: str,
               mean_loss: float, useful_tokens: int, top_ppl: list,
               grad_norm: float, opus_score: float):
        for sid in shard_ids:
            e = self.by_shard.setdefault(sid, {
                "shard_id": sid, "lane": lane, "exposures": 0, "useful_tokens": 0,
                "first_loss": None, "last_loss": None, "loss_sum": 0.0,
                "max_grad_norm": 0.0, "opus_score_sum": 0.0,
                "high_ppl_examples": []})
            e["exposures"] += 1
            e["useful_tokens"] += useful_tokens
            e["loss_sum"] += mean_loss
            e["last_loss"] = mean_loss
            if e["first_loss"] is None:
                e["first_loss"] = mean_loss
            e["max_grad_norm"] = max(e["max_grad_norm"], grad_norm)
            e["opus_score_sum"] += opus_score
            if top_ppl and len(e["high_ppl_examples"]) < 5:
                e["high_ppl_examples"].append({"step": step, **top_ppl[0]})
        l = self.by_lane.setdefault(lane, {"lane": lane, "exposures": 0,
                                           "loss_sum": 0.0, "useful_tokens": 0})
        l["exposures"] += 1; l["loss_sum"] += mean_loss; l["useful_tokens"] += useful_tokens

    def finalize(self) -> dict:
        """Classify each shard useful / neutral / harmful for the next corpus version."""
        shards = []
        for e in self.by_shard.values():
            delta = (e["first_loss"] - e["last_loss"]) if e["exposures"] > 1 else 0.0
            e = dict(e)
            e["mean_loss"] = e["loss_sum"] / max(e["exposures"], 1)
            e["loss_delta"] = delta
            e["mean_opus_score"] = e["opus_score_sum"] / max(e["exposures"], 1)
            if e["max_grad_norm"] > 50:
                e["classification"] = "harmful_gradient_spike"
            elif delta > 0.01:
                e["classification"] = "useful"
            elif e["exposures"] > 1 and delta <= 0:
                e["classification"] = "neutral_repetition_exhausted"
            else:
                e["classification"] = "neutral"
            shards.append(e)
        lanes = []
        for l in self.by_lane.values():
            l = dict(l); l["mean_loss"] = l["loss_sum"] / max(l["exposures"], 1)
            lanes.append(l)
        out = {"shards": sorted(shards, key=lambda x: x["shard_id"]),
               "lanes": sorted(lanes, key=lambda x: x["lane"])}
        json.dump(out, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return out
