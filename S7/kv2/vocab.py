"""Real vocabulary for the audit.

Uses the India-Wikipedia corpora (hi/te/ta/en) when present, otherwise a bundled
sample, so the experiment runs anywhere. Tokens are tagged by script, because the
whole point of the audit is that the 32-byte window is not equally generous
across scripts.
"""
from __future__ import annotations
import os, re, json, unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.environ.get("KV2_CORPUS_DIR",
                            os.path.join(HERE, "external", "corpus_md"))
SAMPLE = os.path.join(HERE, "corpus_sample", "words.json")

RANGES = [("devanagari", 0x0900, 0x097F), ("telugu", 0x0C00, 0x0C7F),
          ("tamil", 0x0B80, 0x0BFF), ("bengali", 0x0980, 0x09FF)]


def script_of(word: str) -> str:
    for ch in word:
        o = ord(ch)
        for name, lo, hi in RANGES:
            if lo <= o <= hi:
                return name
        if ch.isascii() and ch.isalpha():
            return "latin"
    return "other"


# A word is letters PLUS combining marks. Python's \w drops Indic matras, which
# shatters "भारत" into "भ"+"रत" and would silently invalidate the whole audit,
# so we match Unicode letter+mark explicitly.
try:
    import regex as _re2
    WORD_RE = _re2.compile(r"[\p{L}\p{M}‌‍]+")
except ImportError:                       # stdlib fallback: letters + marks by range
    WORD_RE = re.compile(
        "[^\\W\\d_]"
        "[^\\W\\d_̀-ͯऀ-ःऺ-ॏ॑-ॗॢ-ॣ"
        "ঁ-ঃ়-্ஂா-்ఀ-ఄా-ౖ"
        "‌‍]*"
        "[̀-ͯऀ-ःऺ-ॏ॑-ॗॢ-ॣ"
        "ঁ-ঃ়-্ஂா-்ఀ-ఄా-ౖ"
        "‌‍]*", re.UNICODE)



def _from_corpus() -> dict:
    out = {}
    if not os.path.isdir(CORPUS_DIR):
        return out
    for lang in ("en", "hi", "te", "ta"):
        p = os.path.join(CORPUS_DIR, f"{lang}.md")
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        txt = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", txt)
        for w in WORD_RE.findall(txt):
            if 1 <= len(w) <= 60:
                out[w] = out.get(w, 0) + 1
    return out


def load_words(max_words: int = 20000) -> list:
    """Returns [(word, script, freq)] sorted by frequency (deterministic)."""
    counts = _from_corpus()
    source = "real:india-wikipedia"
    if len(counts) < 500:
        counts = Counter(json.load(open(SAMPLE, encoding="utf-8")))
        source = "bundled_sample"
    words = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]
    return [(w, script_of(w), c) for w, c in words], source


def colliding_pairs(words, codec) -> dict:
    """Group words by exact code; any group of size>1 is a permanent collision."""
    from .codec import code_key
    groups = {}
    for w, s, _ in words:
        groups.setdefault(code_key(codec.encode(w)), []).append((w, s))
    return {k: v for k, v in groups.items() if len(v) > 1}


def load_stream(vocab_words, max_tokens: int = 40000) -> list:
    """The corpus as an ordered id stream (real co-occurrence, OOV dropped).
    Needed so the LM experiment measures language modelling, not noise."""
    idx = {w: i for i, w in enumerate(vocab_words)}
    out = []
    if os.path.isdir(CORPUS_DIR):
        for lang in ("en", "hi", "te", "ta"):
            p = os.path.join(CORPUS_DIR, f"{lang}.md")
            if not os.path.exists(p):
                continue
            txt = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1",
                         open(p, encoding="utf-8").read())
            for w in WORD_RE.findall(txt):
                if w in idx:
                    out.append(idx[w])
                    if len(out) >= max_tokens:
                        return out
    if len(out) < 200:                      # bundled fallback: repeat the vocab
        out = [i for i in range(len(vocab_words))] * 40
    return out
