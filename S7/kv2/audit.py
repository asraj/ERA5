"""Measurements. Every number in the paper is produced here.

A1  byte-budget audit  - how many characters the 32-byte window actually holds
                         per script, and how many real words overflow it.
A2  collision audit    - words whose codes are EXACTLY equal, per script. A
                         collision is permanent: no amount of training can
                         separate two tokens the codec maps to one vector.
A3  occupancy audit    - what fraction of the code's cells a token actually
                         uses (the "waste" in problem 3).
A4  geometry audit     - does the fix preserve the property that made Kronecker
                         attractive (prefix-similar tokens start out similar)?
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from .codec import code_key

SCRIPTS = ["latin", "devanagari", "telugu", "tamil"]


def byte_budget(words) -> dict:
    out = {}
    for s in SCRIPTS:
        ws = [w for w, sc, _ in words if sc == s]
        if not ws:
            continue
        ch = np.array([len(w) for w in ws])
        by = np.array([len(w.encode()) for w in ws])
        out[s] = {"words": len(ws), "mean_chars": round(float(ch.mean()), 2),
                  "bytes_per_char": round(float((by / ch).mean()), 2),
                  "mean_bytes": round(float(by.mean()), 2),
                  "max_bytes": int(by.max()),
                  "chars_that_fit_in_32B": round(float(32 / (by / ch).mean()), 1),
                  "words_over_32B": int((by > 32).sum()),
                  "pct_over_32B": round(100 * float((by > 32).mean()), 2)}
    return out


def collisions(words, codec) -> dict:
    """Exact-code collisions. Reported per script and with real examples."""
    groups = defaultdict(list)
    for w, s, _ in words:
        groups[code_key(codec.encode(w))].append((w, s))
    colliding = [g for g in groups.values() if len(g) > 1]
    per_script = defaultdict(int)
    words_affected = 0
    examples = []
    for g in colliding:
        words_affected += len(g)
        for _, s in g:
            per_script[s] += 1
        if len(examples) < 12:
            examples.append([w for w, _ in g])
    return {"codec": codec.name, "distinct_words": len(words),
            "distinct_codes": len(groups),
            "collision_groups": len(colliding),
            "words_affected": words_affected,
            "pct_words_affected": round(100 * words_affected / max(len(words), 1), 3),
            "per_script": dict(sorted(per_script.items())),
            "examples": examples}


def occupancy(words, codec, sample: int = 4000) -> dict:
    """How much of the fixed code a token actually occupies."""
    used, dropped = [], []
    for w, _, _ in words[:sample]:
        v = codec.grid(w)
        used.append(float((v != 0).sum()) / v.size)
        dropped.append(codec.dropped_bytes(w))
    return {"codec": codec.name, "dim": codec.dim,
            "mean_cells_used_pct": round(100 * float(np.mean(used)), 3),
            "mean_bytes_dropped": round(float(np.mean(dropped)), 3),
            "total_bytes_dropped": int(np.sum(dropped)),
            "words_truncated": int(np.sum(np.array(dropped) > 0))}


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def geometry(codec) -> dict:
    """Kronecker's selling point: shared spelling => similar starting vectors.
    A fix that destroys this would be a bad trade, so we measure it."""
    fam = [("train", "training"), ("train", "trainer"), ("play", "playing"),
           ("भारत", "भारतीय"), ("இந்தியா", "இந்தியாவின்")]
    unrel = [("train", "zebra"), ("play", "quantum"), ("भारत", "कंप्यूटर"),
             ("இந்தியா", "கணினி")]
    fs = [_cos(codec.encode(a), codec.encode(b)) for a, b in fam]
    us = [_cos(codec.encode(a), codec.encode(b)) for a, b in unrel]
    return {"codec": codec.name,
            "mean_cos_related": round(float(np.mean(fs)), 4),
            "mean_cos_unrelated": round(float(np.mean(us)), 4),
            "separation": round(float(np.mean(fs) - np.mean(us)), 4),
            "related_pairs": {f"{a}~{b}": round(c, 3) for (a, b), c in zip(fam, fs)}}


def find_collision_pairs(words, codec, want: int = 6) -> list:
    """Real word pairs that the codec maps to one vector - used to build the
    discrimination task the transformer is trained on."""
    groups = defaultdict(list)
    for w, s, _ in words:
        groups[code_key(codec.encode(w))].append((w, s))
    out = []
    for g in groups.values():
        if len(g) > 1:
            for i in range(len(g) - 1):
                out.append((g[i][0], g[i + 1][0], g[i][1]))
                if len(out) >= want:
                    return out
    return out
