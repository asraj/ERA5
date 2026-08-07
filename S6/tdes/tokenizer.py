"""Frozen tokenizer with a content hash.

Contract (Session 2): the same raw text must always produce the same token IDs.
Pure-python, no third-party deps, byte-fallback so encode/decode round-trips
every character (faithful). Once built, the tokenizer is written to disk and
identified by sha256 over its canonical JSON; every shard manifest records that
hash, and training refuses to run if the hash does not match.
"""
from __future__ import annotations
import json, hashlib, re, os
from collections import Counter

SPECIALS = ["<|pad|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|think|>"]
WORD_RE = re.compile(r"\s*\S+")          # Metaspace-like: keep leading space with the word


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class FrozenTokenizer:
    """Word-level vocabulary + 256 byte-fallback tokens. Deterministic and faithful."""

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.byte_base = self.tok2id["<0x00>"]
        self.pad_id = self.tok2id["<|pad|>"]
        self.eos_id = self.tok2id["<|eos|>"]

    # ---------- construction ----------
    @staticmethod
    def train(texts, max_vocab: int = 4096) -> "FrozenTokenizer":
        cnt = Counter()
        for t in texts:
            cnt.update(WORD_RE.findall(t))
        # deterministic ordering: frequency desc, then lexicographic
        words = [w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
        vocab = list(SPECIALS) + [f"<0x{b:02X}>" for b in range(256)]
        room = max_vocab - len(vocab)
        vocab += words[:max(0, room)]
        return FrozenTokenizer(vocab)

    def to_json(self) -> str:
        return canonical_json({"format": "tdes-frozen/1", "vocab": self.vocab})

    @property
    def hash(self) -> str:
        return sha256_str(self.to_json())

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return self.hash

    @staticmethod
    def load(path: str) -> "FrozenTokenizer":
        with open(path, encoding="utf-8") as f:
            return FrozenTokenizer(json.load(f)["vocab"])

    # ---------- codec ----------
    def encode(self, text: str) -> list[int]:
        ids = []
        for w in WORD_RE.findall(text):
            i = self.tok2id.get(w)
            if i is not None:
                ids.append(i)
            else:                                   # byte fallback -> always faithful
                ids.extend(self.byte_base + b for b in w.encode("utf-8"))
        return ids

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            t = self.vocab[i]
            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                out.append(int(t[3:5], 16))
            elif t in SPECIALS:
                if t != "<|pad|>":
                    out += t.encode("utf-8")
            else:
                out += t.encode("utf-8")
        return out.decode("utf-8", errors="replace")

    def __len__(self):
        return len(self.vocab)
