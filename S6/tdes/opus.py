"""OPUS candidate selection with a full audit trail.

Every candidate batch is scored against a proxy direction and lands in one of four
ledgers: accepted, rejected, deferred, protected-override. Rejected clean data is
never thrown away - it is recorded, because a batch that is low-value now may be
valuable later, and a rejected Indic/agentic batch is evidence of proxy bias.

Determinism: the score is a pure function of (proxy_version, candidate features),
so replaying a recorded interval reproduces the same decisions exactly.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, asdict


@dataclass
class OpusDecision:
    candidate_id: str
    shard_ids: list
    lane: str
    stage: str
    proxy_version: str
    score: float
    status: str            # accepted | rejected | deferred | protected_override
    reason: str
    protected_floor_override: bool
    effective_tokens: int


class Opus:
    def __init__(self, accept_rate: float = 0.5, defer_band: float = 0.10,
                 lane_bias: dict | None = None):
        """lane_bias models a real proxy's blind spot: an English-heavy proxy
        systematically under-scores Indic and agentic candidates."""
        self.accept_rate = accept_rate
        self.defer_band = defer_band
        self.lane_bias = lane_bias or {"indic": -0.25, "agentic": -0.30, "reasoning": -0.10}
        self.decisions: list[OpusDecision] = []

    def score(self, candidate_id: str, lane: str, proxy_version: str) -> float:
        h = hashlib.blake2b(f"{proxy_version}|{candidate_id}".encode(), digest_size=8).digest()
        base = int.from_bytes(h, "big") / 2 ** 64          # deterministic uniform [0,1)
        return max(0.0, min(1.0, base + self.lane_bias.get(lane, 0.0)))

    def evaluate(self, candidate_id: str, *, lane: str, stage: str, shard_ids: list,
                 proxy_version: str, effective_tokens: int,
                 floor_deficit: bool) -> OpusDecision:
        s = self.score(candidate_id, lane, proxy_version)
        thr = 1.0 - self.accept_rate
        if s >= thr:
            status, reason, override = "accepted", "high_proxy_utility", False
        elif s >= thr - self.defer_band:
            status, reason, override = "deferred", "borderline_utility_defer_to_later_phase", False
        else:
            status, reason, override = "rejected", "low_proxy_utility", False
        # protected floor is outside the selector's control: rescue the batch
        if status != "accepted" and floor_deficit:
            status, reason, override = "protected_override", "protected_floor_rescue", True
        d = OpusDecision(candidate_id, shard_ids, lane, stage, proxy_version,
                         round(s, 6), status, reason, override, effective_tokens)
        self.decisions.append(d)
        return d

    # ---- audit helpers ----
    def counts(self) -> dict:
        c = {}
        for d in self.decisions:
            c[d.status] = c.get(d.status, 0) + 1
        return c

    def by_lane(self) -> dict:
        out = {}
        for d in self.decisions:
            out.setdefault(d.lane, {}).setdefault(d.status, 0)
            out[d.lane][d.status] += 1
        return out

    def as_records(self) -> list:
        return [asdict(d) for d in self.decisions]
