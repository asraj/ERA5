#!/usr/bin/env python3
"""Export a Colab-sized slice of the Session-4 CLEANED OpenWebText corpus.

Session 4 ran the 8-stage pipeline but only persisted stats.json, so this script
re-runs the same cleaning stages (imported from S4, not reimplemented) over a
fresh slice of the local OpenWebText copy and writes the surviving documents to
a gzipped JSONL the Session-9 notebook can fetch from GitHub raw.

Output: s9_owt_clean.jsonl.gz  +  s9_owt_clean.manifest.json (sha256, counts)
"""
import os, sys, json, gzip, hashlib, glob
sys.path.insert(0, "/sessions/serene-dazzling-volta/mnt/ERA/s4_cleaning")
from clean_pipeline import normalize, unify_format, quality_ok, scrub, dedup, detect_lang

OWT  = "/sessions/serene-dazzling-volta/mnt/ERA/openwebtext"
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_WORDS = 3_000_000            # ~4M SmolLM2 tokens: plenty for a few hundred steps
MIN_CHARS, MAX_CHARS = 400, 12_000  # drop stubs; cap monsters so batches stay even

def read_owt(target):
    """Stream the local HF-datasets arrow copy without loading 13 GB."""
    import pyarrow.parquet as pq, pyarrow.ipc as ipc, pyarrow as pa
    files = sorted(glob.glob(OWT + "/**/*.arrow", recursive=True))
    assert files, f"no .arrow files under {OWT}"
    words = 0
    for f in files:
        if words >= target: break
        with pa.memory_map(f, "rb") as src:
            try:    reader = ipc.open_stream(src)
            except Exception: reader = ipc.open_file(src)
            for batch in (reader if hasattr(reader, "__iter__") else
                          (reader.get_batch(i) for i in range(reader.num_record_batches))):
                for t in batch.column("text").to_pylist():
                    t = (t or "").strip()
                    if not (MIN_CHARS <= len(t) <= MAX_CHARS): continue
                    yield t
                    words += t.count(" ") + 1
                    if words >= target: return

def main():
    raw = list(read_owt(TARGET_WORDS))
    print(f"1 extraction      : {len(raw):>7,} docs")

    # NOTE: the S4 stage functions return (text, counters) tuples and dedup()
    # reads d["text_clean"] - call them exactly as S4 does rather than guessing.
    docs, garbage, ghosts = [], 0, 0
    for t in raw:
        t, bad = normalize(t);      garbage += bad
        t, hits = unify_format(t);  ghosts  += hits
        docs.append({"text_clean": t, "lang_claimed": "en", "src": "openwebtext"})
    print(f"2 normalize       : {len(docs):>7,} docs  ({garbage:,} garbage chars removed)")
    print(f"3 format          : {len(docs):>7,} docs  ({ghosts:,} ghost markers unified)")

    docs = [d for d in docs if quality_ok(d["text_clean"], "en")]
    print(f"4 quality         : {len(docs):>7,} docs")
    docs, exact, near = dedup(docs)
    print(f"5 dedup           : {len(docs):>7,} docs  ({exact:,} exact, {near:,} near-dup)")
    docs = [d for d in docs if detect_lang(d["text_clean"]) == "en"]
    print(f"6 language-ID     : {len(docs):>7,} docs")
    pii = 0
    for d in docs:
        d["text_clean"], c = scrub(d["text_clean"]); pii += sum(c.values())
    print(f"7 PII scrub       : {len(docs):>7,} docs  ({pii:,} spans redacted)")

    # 8 decontamination is a no-op here: no eval set travels with this shard

    out = os.path.join(HERE, "s9_owt_clean.jsonl.gz")
    words = 0
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for i, d in enumerate(docs):
            words += d["text_clean"].count(" ") + 1
            fh.write(json.dumps({"id": i, "text": d["text_clean"]}, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    man = {"file": os.path.basename(out), "sha256": sha,
           "docs": len(docs), "whitespace_words": words,
           "bytes_gz": os.path.getsize(out),
           "source": "OpenWebText (Skylion007) local copy",
           "cleaning": "Session-4 pipeline stages 2-7 (normalize, format, quality, "
                       "dedup exact+MinHash, language-ID, PII scrub)",
           "note": "stage 8 (decontamination) is a no-op: no eval set ships with this shard"}
    json.dump(man, open(os.path.join(HERE, "s9_owt_clean.manifest.json"), "w"), indent=2)
    print(f"\nwrote {out}")
    print(f"  {len(docs):,} docs · {words:,} whitespace words · "
          f"{os.path.getsize(out)/1e6:.1f} MB gz · sha256 {sha[:16]}")

if __name__ == "__main__":
    main()
