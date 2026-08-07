"""Immutable tokenized shards + manifests, and the admission gate.

A shard is a sealed object: a binary token array (.bin, uint32) plus an index of
document spans. Its identity is the sha256 of the .bin bytes. Mutating a shard
produces a new hash and therefore a new shard, never an edited one.
"""
from __future__ import annotations
import os, json, hashlib, struct
from dataclasses import dataclass, asdict, field

from .tokenizer import canonical_json, FrozenTokenizer

MANIFEST_REQUIRED = [
    "shard_id", "source_ids", "doc_ids", "tokenizer_hash", "token_count",
    "language", "capability_lane", "license", "provenance_tier",
    "cleaning_pipeline_hash", "dedup_status", "contamination_status",
    "eval_overlap", "content_hash", "parent_shard_ids", "split",
    "reserved_for_anneal",
]


@dataclass
class DocSpan:
    doc_id: str
    start: int
    length: int


@dataclass
class ShardManifest:
    shard_id: str
    source_ids: list
    doc_ids: list
    tokenizer_hash: str
    token_count: int
    language: str
    capability_lane: str
    license: str
    provenance_tier: str
    cleaning_pipeline_hash: str
    dedup_status: str
    contamination_status: str
    eval_overlap: bool
    content_hash: str
    parent_shard_ids: list
    split: str                      # train | validation | test
    reserved_for_anneal: bool = False   # held back for the cooldown, unspendable earlier
    spans: list = field(default_factory=list)

    def to_json(self) -> str:
        return canonical_json(asdict(self))


class ShardWriter:
    """Writes an immutable shard: <id>.bin + <id>.manifest.json"""

    def __init__(self, root: str, tokenizer: FrozenTokenizer, cleaning_pipeline_hash: str):
        self.root = root
        self.tok = tokenizer
        self.clean_hash = cleaning_pipeline_hash
        os.makedirs(os.path.join(root, "shards"), exist_ok=True)
        os.makedirs(os.path.join(root, "manifests"), exist_ok=True)

    def write(self, shard_id: str, docs: list, *, lane: str, language: str,
              license: str, tier: str, split: str = "train",
              dedup_status: str = "deduplicated", contamination_status: str = "scanned",
              eval_overlap: bool = False, source_ids=None, parents=None,
              reserved_for_anneal: bool = False) -> ShardManifest:
        """docs: list of (doc_id, text). Tokens are laid out contiguously with EOS between docs."""
        toks, spans, doc_ids = [], [], []
        for doc_id, text in docs:
            ids = self.tok.encode(text) + [self.tok.eos_id]
            spans.append(asdict(DocSpan(doc_id, len(toks), len(ids))))
            toks.extend(ids)
            doc_ids.append(doc_id)

        blob = struct.pack(f"<{len(toks)}I", *toks)
        content_hash = hashlib.sha256(blob).hexdigest()
        bin_path = os.path.join(self.root, "shards", f"{shard_id}.bin")
        if os.path.exists(bin_path):                  # allow a clean overwrite on re-run
            try:
                os.chmod(bin_path, 0o666)
            except OSError:
                pass
        with open(bin_path, "wb") as f:
            f.write(blob)
        try:
            os.chmod(bin_path, 0o444)                 # sealed: read-only on disk
        except OSError:
            pass                                      # some filesystems refuse chmod

        man = ShardManifest(
            shard_id=shard_id, source_ids=source_ids or [shard_id.split("-")[0]],
            doc_ids=doc_ids, tokenizer_hash=self.tok.hash, token_count=len(toks),
            language=language, capability_lane=lane, license=license,
            provenance_tier=tier, cleaning_pipeline_hash=self.clean_hash,
            dedup_status=dedup_status, contamination_status=contamination_status,
            eval_overlap=eval_overlap, content_hash=content_hash,
            parent_shard_ids=parents or [], split=split,
            reserved_for_anneal=reserved_for_anneal, spans=spans)
        with open(os.path.join(self.root, "manifests", f"{shard_id}.manifest.json"), "w",
                  encoding="utf-8") as f:
            f.write(man.to_json())
        return man


class ShardStore:
    """Read-side: loads manifests, verifies integrity, serves token spans."""

    def __init__(self, root: str, tokenizer_hash: str):
        self.root = root
        self.tokenizer_hash = tokenizer_hash
        self.manifests: dict[str, ShardManifest] = {}
        mdir = os.path.join(root, "manifests")
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith(".manifest.json"):
                continue
            with open(os.path.join(mdir, fn), encoding="utf-8") as fh:
                d = json.load(fh)
            self.manifests[d["shard_id"]] = ShardManifest(**d)
        self._cache: dict[str, list] = {}

    # ---------- integrity ----------
    def verify(self, shard_id: str) -> tuple[bool, str]:
        m = self.manifests[shard_id]
        with open(os.path.join(self.root, "shards", f"{shard_id}.bin"), "rb") as fh:
            blob = fh.read()
        if hashlib.sha256(blob).hexdigest() != m.content_hash:
            return False, "content_hash_mismatch"
        if m.tokenizer_hash != self.tokenizer_hash:
            return False, "tokenizer_hash_mismatch"
        if any(getattr(m, k, None) is None for k in MANIFEST_REQUIRED):
            return False, "manifest_incomplete"
        return True, "ok"

    def tokens(self, shard_id: str) -> list:
        if shard_id not in self._cache:
            with open(os.path.join(self.root, "shards", f"{shard_id}.bin"), "rb") as fh:
                blob = fh.read()
            self._cache[shard_id] = list(struct.unpack(f"<{len(blob)//4}I", blob))
        return self._cache[shard_id]

    def span_tokens(self, shard_id: str, span_idx: int) -> list:
        s = self.manifests[shard_id].spans[span_idx]
        t = self.tokens(shard_id)
        return t[s["start"]: s["start"] + s["length"]]

    def by_lane(self, lane: str, split: str = "train") -> list:
        return sorted(sid for sid, m in self.manifests.items()
                      if m.capability_lane == lane and m.split == split)


# ---------------- admission gate ----------------
UNSAFE_LICENSES = {"unknown", "noncommercial", "proprietary"}


def admission_check(m: ShardManifest, tokenizer_hash: str) -> tuple[bool, str]:
    """Session-3/4 contract: only cleaned, licensed, uncontaminated, correctly
    tokenized train shards may enter the stream."""
    if m.split != "train":
        return False, f"split_not_trainable:{m.split}"
    if m.eval_overlap:
        return False, "eval_overlap"
    if m.contamination_status != "scanned":
        return False, f"contamination_{m.contamination_status}"
    if m.license.lower() in UNSAFE_LICENSES:
        return False, f"unsafe_license:{m.license}"
    if not m.cleaning_pipeline_hash:
        return False, "unknown_cleaning_lineage"
    if m.tokenizer_hash != tokenizer_hash:
        return False, "tokenizer_hash_mismatch"
    if m.dedup_status != "deduplicated":
        return False, "not_deduplicated"
    return True, "admitted"
