"""Curriculum stages -> executable per-step lane quotas.

Session 5 gave the plan in human terms. Here it becomes a compiled schedule with
warmup-blended transitions (never a hard step) and protected floors the selector
may not cross.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .tokenizer import canonical_json, sha256_str


@dataclass
class Stage:
    name: str
    step_start: int
    step_end: int
    sequence_length: int
    mixture: dict            # lane -> share (sums to 1)
    protected_floors: dict   # lane -> min share of every batch
    warmup_steps: int = 0
    packing_policy: str = "concat_chop"


class MixtureSchedule:
    def __init__(self, stages: list[Stage], anneal_reserve_lanes: tuple = ("agentic", "indic")):
        self.stages = stages
        self.anneal_reserve_lanes = anneal_reserve_lanes
        assert stages[0].step_start == 0
        for a, b in zip(stages, stages[1:]):
            assert a.step_end == b.step_start, "stages must be contiguous"
        for s in stages:
            tot = sum(s.mixture.values())
            assert abs(tot - 1.0) < 1e-6, f"stage {s.name} mixture sums to {tot}"
            for lane, fl in s.protected_floors.items():
                assert s.mixture.get(lane, 0) >= fl, f"{lane} share below its own floor"

    @property
    def total_steps(self) -> int:
        return self.stages[-1].step_end

    def stage_for(self, step: int) -> Stage:
        for s in self.stages:
            if s.step_start <= step < s.step_end:
                return s
        return self.stages[-1]

    def weights_for(self, step: int) -> tuple[dict, Stage]:
        """Lane weights at this step, blended across the warmup band of a transition."""
        s = self.stage_for(step)
        prev = None
        for i, st in enumerate(self.stages):
            if st is s and i > 0:
                prev = self.stages[i - 1]
        if prev and s.warmup_steps > 0 and step < s.step_start + s.warmup_steps:
            f = (step - s.step_start + 1) / s.warmup_steps          # 0..1 ramp
            lanes = set(prev.mixture) | set(s.mixture)
            w = {l: (1 - f) * prev.mixture.get(l, 0.0) + f * s.mixture.get(l, 0.0) for l in lanes}
            tot = sum(w.values())
            return {l: v / tot for l, v in w.items()}, s
        return dict(s.mixture), s

    @property
    def hash(self) -> str:
        return sha256_str(canonical_json([asdict(s) for s in self.stages]))

    def to_json(self) -> str:
        return canonical_json({"schedule_hash": self.hash,
                               "anneal_reserve_lanes": list(self.anneal_reserve_lanes),
                               "stages": [asdict(s) for s in self.stages]})


def default_schedule(total_steps: int, seq_len: int = 256) -> MixtureSchedule:
    """A compact 4-stage curriculum mirroring the V5 plan: foundation -> capability
    -> long-context -> anneal. General web falls, scarce lanes rise, floors hold."""
    a = int(total_steps * 0.40); b = int(total_steps * 0.75); c = int(total_steps * 0.90)
    floors = {"indic": 0.08, "agentic": 0.015, "reasoning": 0.015}
    return MixtureSchedule([
        Stage("foundation", 0, a, seq_len,
              {"general_web": 0.55, "code": 0.10, "indic": 0.20, "reasoning": 0.08, "agentic": 0.07},
              floors, warmup_steps=0, packing_policy="concat_chop"),
        Stage("capability", a, b, seq_len,
              {"general_web": 0.35, "code": 0.25, "indic": 0.20, "reasoning": 0.12, "agentic": 0.08},
              floors, warmup_steps=4, packing_policy="best_fit"),
        Stage("long_context", b, c, seq_len,
              {"general_web": 0.30, "code": 0.25, "indic": 0.20, "reasoning": 0.15, "agentic": 0.10},
              floors, warmup_steps=3, packing_policy="long_context"),
        Stage("anneal", c, total_steps, seq_len,
              {"general_web": 0.15, "code": 0.20, "indic": 0.30, "reasoning": 0.20, "agentic": 0.15},
              floors, warmup_steps=3, packing_policy="structure_preserving"),
    ])
