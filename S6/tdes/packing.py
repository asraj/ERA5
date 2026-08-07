"""Packing policies + masks.

A packed sequence carries more than token ids: it carries the *training meaning*
of those ids.
  loss_mask      1 = token contributes to the gradient, 0 = context/pad only
  segment_ids    which packed sample each position belongs to (block-diagonal
                 attention -> unrelated samples cannot attend to each other)
  position_ids   reset at every sample boundary
Policies differ by data type, as the lesson requires: plain text tolerates
concat-and-chop; SFT/agentic must preserve structure and mask tool observations.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field

POLICIES = ["pad_only", "concat_chop", "greedy", "best_fit",
            "structure_preserving", "long_context"]


@dataclass
class PackedSequence:
    tokens: list
    loss_mask: list
    segment_ids: list
    position_ids: list
    sample_ids: list = field(default_factory=list)   # provenance per packed sample
    lanes: list = field(default_factory=list)

    def validate(self, seq_len: int, pad_id: int):
        assert len(self.tokens) == seq_len, "wrong length"
        assert len(self.loss_mask) == len(self.segment_ids) == len(self.position_ids) == seq_len
        for t, m in zip(self.tokens, self.loss_mask):
            if t == pad_id:
                assert m == 0, "pad token must never bear loss"
        # position ids restart at each new segment and increase by 1 within it
        last_seg, expect = None, 0
        for s, p in zip(self.segment_ids, self.position_ids):
            if s != last_seg:
                assert p == 0, "position ids must reset at a sample boundary"
                last_seg, expect = s, 1
            else:
                assert p == expect, "position ids must increase by 1 inside a sample"
                expect += 1
        return True

    @property
    def useful_tokens(self) -> int:
        return sum(self.loss_mask)

    def utilization(self, pad_id: int) -> float:
        return sum(1 for t in self.tokens if t != pad_id) / len(self.tokens)

    def hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.tokens, self.loss_mask, self.segment_ids, self.position_ids):
            h.update(bytes(str(arr), "utf-8"))
        h.update(bytes(str(sorted(self.sample_ids)), "utf-8"))
        return h.hexdigest()


class Packer:
    def __init__(self, seq_len: int, pad_id: int, eos_id: int):
        self.seq_len, self.pad_id, self.eos_id = seq_len, pad_id, eos_id

    def pack(self, samples: list, policy: str) -> PackedSequence:
        """samples: list of dicts {tokens, sample_id, lane, ctx_len(optional)}.
        ctx_len marks leading context tokens (prompt / tool observation) that must
        NOT bear loss - the agentic masking rule."""
        assert policy in POLICIES, f"unknown policy {policy}"
        if policy == "pad_only":
            chosen = samples[:1]
        elif policy in ("greedy", "structure_preserving", "long_context"):
            chosen = self._greedy(samples)
        elif policy == "best_fit":
            chosen = self._best_fit(samples)
        else:                                     # concat_chop
            return self._concat_chop(samples)
        return self._assemble(chosen, truncate=(policy == "long_context"))

    # ---- policies ----
    def _greedy(self, samples):
        out, used = [], 0
        for s in samples:
            n = len(s["tokens"])
            if used + n <= self.seq_len:
                out.append(s); used += n
        return out or samples[:1]

    def _best_fit(self, samples):
        remaining = sorted(samples, key=lambda s: -len(s["tokens"]))
        out, used = [], 0
        for s in remaining:
            n = len(s["tokens"])
            if used + n <= self.seq_len:
                out.append(s); used += n
        return out or samples[:1]

    def _concat_chop(self, samples):
        """Documents joined with EOS boundaries; fixed window cut from the stream.
        Safe for plain text: the EOS + segment id tell the model one text ended."""
        toks, segs, pos, loss, ids, lanes = [], [], [], [], [], []
        for seg, s in enumerate(samples):
            j = 0
            for t in s["tokens"]:
                if len(toks) >= self.seq_len:
                    break
                toks.append(t); loss.append(1); segs.append(seg); pos.append(j); j += 1
            ids.append(s["sample_id"]); lanes.append(s["lane"])
            if len(toks) >= self.seq_len:
                break
        return self._finish(toks, loss, segs, pos, ids, lanes)

    def _assemble(self, chosen, truncate=False):
        toks, loss, segs, pos, ids, lanes = [], [], [], [], [], []
        for seg, s in enumerate(chosen):
            st = s["tokens"]
            if truncate:
                st = st[: self.seq_len]
            ctx = s.get("ctx_len", 0)
            for j, t in enumerate(st):
                if len(toks) >= self.seq_len:
                    break
                toks.append(t)
                loss.append(0 if j < ctx else 1)     # context / tool observations: no loss
                segs.append(seg)
                pos.append(j)
            ids.append(s["sample_id"]); lanes.append(s["lane"])
        return self._finish(toks, loss, segs, pos, ids, lanes)

    def _finish(self, toks, loss, segs, pos, ids, lanes):
        pad_seg = (segs[-1] + 1) if segs else 0
        p = 0
        while len(toks) < self.seq_len:              # right padding, never loss-bearing
            toks.append(self.pad_id); loss.append(0); segs.append(pad_seg); pos.append(p); p += 1
        return PackedSequence(toks[:self.seq_len], loss[:self.seq_len],
                              segs[:self.seq_len], pos[:self.seq_len], ids, lanes)


def _seg_start(segs, seg):
    for i, s in enumerate(segs):
        if s == seg:
            return i
    return 0
