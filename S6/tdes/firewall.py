"""Evaluation / validation firewall.

Test shards are *registered* so the system knows they exist, precisely so it can
refuse to train on them. Validation shards may be read for evaluation but never
become gradient-bearing. Enforcement is two-layer:
  1. shard level  - never_train flag + content hash
  2. content level- contamination fingerprints (n-gram hashes) checked against
                    the actual token stream of a packed batch
"""
from __future__ import annotations
import hashlib, json, os


def ngram_fingerprints(tokens, n: int = 8, stride: int = 4) -> set:
    out = set()
    for i in range(0, max(0, len(tokens) - n + 1), stride):
        h = hashlib.blake2b(bytes(str(tokens[i:i + n]), "utf-8"), digest_size=8).hexdigest()
        out.add(h)
    return out


class EvalFirewall:
    def __init__(self):
        self.registry: dict[str, dict] = {}     # shard_id -> record
        self.fingerprints: dict[str, str] = {}  # fingerprint -> shard_id
        self.events: list[dict] = []            # blocked/allowed audit events

    def register(self, shard_id: str, *, split: str, content_hash: str,
                 benchmark_id: str, tokens=None, never_train: bool = True):
        self.registry[shard_id] = {
            "shard_id": shard_id, "split": split, "content_hash": content_hash,
            "benchmark_id": benchmark_id, "never_train": never_train,
            "access_log": [],
        }
        if tokens:
            for fp in ngram_fingerprints(tokens):
                self.fingerprints[fp] = shard_id

    # ---- layer 1: shard admission ----
    def check_shard(self, shard_id: str) -> tuple[bool, str]:
        rec = self.registry.get(shard_id)
        if rec and rec["never_train"]:
            self._event("shard_blocked", shard_id, rec["benchmark_id"], "never_train_flag")
            return False, "never_train_flag"
        return True, "ok"

    # ---- layer 2: content contamination in an assembled batch ----
    def check_tokens(self, tokens, where: str) -> tuple[bool, str]:
        hits = ngram_fingerprints(tokens) & set(self.fingerprints)
        if hits:
            sid = self.fingerprints[sorted(hits)[0]]
            self._event("content_blocked", sid, where, f"{len(hits)}_fingerprint_hits")
            return False, f"contamination:{sid}"
        return True, "ok"

    def note_access(self, shard_id: str, purpose: str):
        if shard_id in self.registry:
            self.registry[shard_id]["access_log"].append(purpose)

    def _event(self, kind, shard_id, benchmark_id, reason):
        self.events.append({"event": kind, "shard_id": shard_id,
                            "benchmark_id": benchmark_id, "reason": reason})

    def dump(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"registry": self.registry, "events": self.events,
                   "fingerprint_count": len(self.fingerprints)},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
