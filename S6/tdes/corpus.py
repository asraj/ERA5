"""Corpus intake.

Prefers the real local sources used earlier in the course:
  * OpenWebText (HuggingFace arrow shards)  -> general_web lane
  * India-Wikipedia markdown (hi/te/ta)     -> indic lane
and derives code / reasoning / agentic lanes plus a held-out eval set.

If those paths are absent (e.g. a grader's machine), it falls back to the small
bundled sample in corpus_sample/ so `python run_demo.py` always runs.
"""
from __future__ import annotations
import os, glob, json, re, random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(HERE, "corpus_sample")

# Optional real corpora. Point these at a local copy if you have one:
#   export TDES_OWT_DIR=/path/to/openwebtext        (HF arrow / parquet / jsonl)
#   export TDES_INDIC_DIR=/path/to/corpus_md        (hi.md, te.md, ta.md)
# If unset or absent, the bundled corpus_sample/ is used and the demo still runs
# end to end - no network, no large download, no machine-specific paths.
OWT_DIR = os.environ.get("TDES_OWT_DIR", os.path.join(HERE, "external", "openwebtext"))
INDIC_DIR = os.environ.get("TDES_INDIC_DIR", os.path.join(HERE, "external", "corpus_md"))


def _owt_docs(limit_docs: int, max_chars: int = 1400):
    out = []
    if not os.path.isdir(OWT_DIR):
        return out
    try:
        import pyarrow as pa
    except Exception:
        return out
    files = sorted(glob.glob(OWT_DIR + "/**/*.arrow", recursive=True))
    for f in files:
        if len(out) >= limit_docs:
            break
        if "/.cache/" in f:
            continue
        try:
            try:
                it = iter(pa.ipc.open_stream(pa.memory_map(f, "r")))
            except Exception:
                rd = pa.ipc.open_file(pa.memory_map(f, "r"))
                it = (rd.get_batch(i) for i in range(rd.num_record_batches))
            for b in it:
                names = b.schema.names
                col = "text" if "text" in names else names[0]
                for t in b.column(col).to_pylist():
                    t = re.sub(r"\s+", " ", (t or "")).strip()
                    if len(t) > 200:
                        out.append(t[:max_chars])
                        if len(out) >= limit_docs:
                            break
                if len(out) >= limit_docs:
                    break
        except Exception:
            continue
    return out


def _indic_docs(limit_docs: int, max_chars: int = 900):
    out = []
    if not os.path.isdir(INDIC_DIR):
        return out
    for lang in ("hi", "te", "ta"):
        p = os.path.join(INDIC_DIR, f"{lang}.md")
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        txt = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", txt)
        chunks = [re.sub(r"\s+", " ", c).strip() for c in re.split(r"\n\s*\n", txt)]
        for c in chunks:
            if len(c) > 120:
                out.append((lang, c[:max_chars]))
            if len(out) >= limit_docs:
                break
    return out[:limit_docs]


def _synth_code(n, rng):
    tmpl = ("def {name}(xs):\n    total = 0\n    for x in xs:\n        if x % {m} == 0:\n"
            "            total += x * {k}\n    return total\n\n"
            "class {cls}:\n    def __init__(self, n):\n        self.n = n\n"
            "    def scaled(self):\n        return [i * self.n for i in range({r})]\n")
    return [tmpl.format(name=f"agg_{i}", m=rng.randint(2, 9), k=rng.randint(2, 7),
                        cls=f"Box{i}", r=rng.randint(5, 40)) for i in range(n)]


def _synth_reasoning(n, rng):
    out = []
    for i in range(n):
        a, b, c = rng.randint(2, 40), rng.randint(2, 30), rng.randint(2, 12)
        out.append(f"Question: A train covers {a} km in {b} minutes, then {c} km more. "
                   f"<|think|> First find the rate: {a}/{b} km per minute. "
                   f"Then total distance is {a} + {c} = {a+c} km. "
                   f"Check the units are consistent. <|think|> "
                   f"Answer: the total distance is {a+c} km.")
    return out


def _synth_agentic(n, rng):
    out = []
    for i in range(n):
        q = rng.choice(["find recent grants", "locate the config bug", "summarise the outage"])
        out.append(f"<|user|> Task {i}: {q}. "
                   f"<|assistant|> Plan: search, then read results, then answer. "
                   f"call search(query='{q}', page={rng.randint(1,4)}) "
                   f"<|tool|> observation: {rng.randint(2,9)} results returned, one stale. "
                   f"<|assistant|> The stale entry failed; retry with a filter. "
                   f"call search(query='{q}', filter='recent') "
                   f"<|tool|> observation: 3 fresh records. "
                   f"<|assistant|> Final answer: {rng.randint(2,5)} relevant records found.")
    return out


def build_documents(seed: int = 11) -> dict:
    """Returns {lane: [(doc_id, text)]} plus an 'eval' pseudo-lane (never trainable)."""
    rng = random.Random(seed)
    owt = _owt_docs(140)
    indic = _indic_docs(90)
    source = "real:openwebtext+india-wikipedia"
    if not owt or not indic:                       # bundled fallback keeps the demo runnable
        s = json.load(open(os.path.join(SAMPLE, "sample.json"), encoding="utf-8"))
        owt = owt or s["general_web"]
        indic = indic or [tuple(x) for x in s["indic"]]
        source = "bundled_sample"

    docs = {
        "general_web": [(f"owt-{i:04d}", t) for i, t in enumerate(owt[:120])],
        "indic": [(f"indic-{lang}-{i:04d}", t) for i, (lang, t) in enumerate(indic[:80])],
        "code": [(f"code-{i:04d}", t) for i, t in enumerate(_synth_code(40, rng))],
        "reasoning": [(f"reason-{i:04d}", t) for i, t in enumerate(_synth_reasoning(40, rng))],
        "agentic": [(f"agent-{i:04d}", t) for i, t in enumerate(_synth_agentic(30, rng))],
    }
    # held-out benchmark material: registered, fingerprinted, NEVER trainable
    docs["_eval"] = [(f"eval-{i:04d}",
                      f"BENCHMARK ITEM {i}: What is the capital of country {i}? "
                      f"Answer: Capital-{i}. CANARY::tdes::never-train::{i:04d}")
                     for i in range(12)]
    docs["_meta"] = source
    return docs
