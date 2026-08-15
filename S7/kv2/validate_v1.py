"""Validation of this work against Kronecker Embeddings V1 (Shravan, 2026).

Two jobs:

  1. FIDELITY - prove our KroneckerV1 reimplementation matches the paper's spec,
     property by property (Eq. 1, coordinate layout, 1/sqrt(L), z-norm, d_c=256,
     d_p=32, UTF-8-safe truncation, byte-fallback tokens, special tokens).

  2. CLAIM TEST - the paper states twice, of tokens longer than d_p:

        "These truncated tokens still receive distinct embeddings based on their
         first 32 bytes; only the post-byte-32 byte structure is lost."   (§4.2)
        "The truncated tokens still receive unique embeddings from their first
         32 bytes."                                                       (§7)

     That is a universal claim. It is false whenever two tokens share a 32-byte
     prefix: they receive the SAME vector, not merely a degraded one. We test it
     on real vocabularies and report a counterexample if one exists.

Reported honestly: the paper's <=0.18% truncation rate is measured over TOKENIZER
VOCABULARIES, and we reproduce that order of magnitude. Our collision result does
not contradict the rate - it contradicts the claim about what happens to the
tokens inside it.
"""
from __future__ import annotations
import numpy as np
from .codec import KroneckerV1, ElasticKronecker, code_key, utf8_safe_len

PAPER = {
    "d_c": 256, "d_p": 32, "D": 8192,
    "truncation_coverage_at_32": 0.9982,          # Table §4.2, worst tokenizer
    "claim_truncated_tokens_stay_unique": True,   # §4.2 and §7
}


def fidelity() -> dict:
    """Property-by-property check of the reimplementation against the paper."""
    k = KroneckerV1()
    checks = {}

    # Eq.1 shape and coordinate layout: (b_p, p) -> b_p * d_p + p
    g = k.grid("A")                                    # 'A' = 0x41 at position 0
    flat = g.reshape(-1)
    checks["coordinate_layout_b*dp+p"] = bool(
        flat[0x41 * k.pos_dim + 0] != 0 and np.count_nonzero(flat) == 1)

    # at most L nonzeros, each equal to 1/sqrt(L) BEFORE z-norm
    t = "test"
    L = len(t.encode())
    gr = k.grid(t).reshape(-1)
    nz = gr[gr != 0]
    checks["at_most_L_nonzeros"] = bool(np.count_nonzero(gr) <= L)
    checks["nonzeros_equal_1_over_sqrt_L"] = bool(
        np.allclose(nz, 1 / np.sqrt(L), atol=1e-6))

    # expected squared L2 norm ~ 1 regardless of L (paper's variance argument)
    norms = [float((k.grid(w).reshape(-1) ** 2).sum())
             for w in ["a", "test", "internationalisation", "भारत"]]
    checks["unit_squared_norm_any_length"] = bool(
        all(abs(n - 1.0) < 1e-5 for n in norms))

    # per-token z-normalisation. NOTE: the paper's prose (§3.3) says "mean 0 and
    # standard deviation 1", but the reference CODE uses torch .std() (Bessel-
    # corrected, ddof=1) and ADDS eps=1e-6 to the denominator, so the realised
    # std is 0.99985, not 1. We follow the code, which is authoritative for an
    # implementation, and check bit-equality against a transcription of it.
    from . import reference as R
    e = k.encode("hello")
    ref = R.encode_single(R.utf8_safe_truncate(b"hello", k.pos_dim))
    checks["z_norm_matches_reference_code"] = bool(
        abs(e.mean()) < 1e-6 and abs(float(np.abs(e - ref).max())) < 1e-6)
    checks["prose_says_std1_code_gives"] = round(float(e.std()), 6)

    # bit-for-bit agreement with the reference implementation on a mixed sample
    worst = 0.0
    for w in ["a", "hello", "भारत", "இந்தியாவில்", "తెలుగు", "<s>", "x" * 40, "🎯"]:
        worst = max(worst, float(np.abs(
            k.encode(w) - R.encode_single(R.utf8_safe_truncate(w.encode(), k.pos_dim))).max()))
    checks["bit_identical_to_reference_impl"] = bool(worst < 1e-6)
    checks["max_abs_diff_vs_reference"] = worst

    # dimensions
    checks["d_c_256_d_p_32_D_8192"] = bool(
        k.char_dim == PAPER["d_c"] and k.pos_dim == PAPER["d_p"] and k.dim == PAPER["D"])

    # UTF-8-safe truncation: never split a multi-byte codepoint
    w = "अ" * 20                                        # 3 bytes each -> 60 bytes
    Ls = utf8_safe_len(w.encode(), 32)
    checks["utf8_safe_truncation"] = bool(Ls % 3 == 0 and Ls <= 32)

    # deterministic
    checks["deterministic"] = bool(np.array_equal(k.encode("भारत"), k.encode("भारत")))

    # byte-fallback / special tokens are encoded as their literal surface bytes
    checks["special_token_literal_bytes"] = bool(np.count_nonzero(k.grid("<s>")) == 3)

    checks["ALL_PASS"] = all(v for kk, v in checks.items()
                             if isinstance(v, bool))
    return checks


def truncation_rate(vocab_tokens) -> dict:
    """Reproduce the paper's Table §4.2 metric: % of a vocabulary within d_p bytes."""
    k32, k16, k64 = KroneckerV1(32), KroneckerV1(16), KroneckerV1(64)
    n = max(len(vocab_tokens), 1)
    out = {}
    for name, kk in (("d_p=16", k16), ("d_p=32", k32), ("d_p=64", k64)):
        within = sum(1 for t in vocab_tokens if len(t.encode()) <= kk.pos_dim)
        out[name] = round(100 * within / n, 2)
    out["paper_reference_at_32"] = round(100 * PAPER["truncation_coverage_at_32"], 2)
    out["vocab_size"] = len(vocab_tokens)
    return out


def test_uniqueness_claim(vocab_tokens, label: str) -> dict:
    """THE claim: do truncated tokens still receive distinct/unique embeddings?"""
    k = KroneckerV1()
    ek = ElasticKronecker()
    truncated = [t for t in vocab_tokens if k.dropped_bytes(t) > 0]
    groups = {}
    for t in vocab_tokens:
        groups.setdefault(code_key(k.encode(t)), []).append(t)
    colliding = [g for g in groups.values() if len(g) > 1]

    # elastic control on the same set
    egroups = {}
    for t in vocab_tokens:
        egroups.setdefault(code_key(ek.encode(t)), []).append(t)
    ecolliding = [g for g in egroups.values() if len(g) > 1]

    return {
        "vocabulary": label,
        "size": len(vocab_tokens),
        "tokens_truncated": len(truncated),
        "pct_truncated": round(100 * len(truncated) / max(len(vocab_tokens), 1), 3),
        "v1_collision_groups": len(colliding),
        "v1_tokens_sharing_a_vector": sum(len(g) for g in colliding),
        "claim_holds": len(colliding) == 0,
        "counterexample": (sorted(colliding, key=len, reverse=True)[0][:6]
                           if colliding else None),
        "elastic_collision_groups": len(ecolliding),
    }


def lesson_example() -> dict:
    """The pair named in the course lesson, checked end to end."""
    a, b = "अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता"
    k, ek = KroneckerV1(), ElasticKronecker()
    ka, kb = k.encode(a), k.encode(b)
    return {
        "pair": [a, b],
        "bytes": [len(a.encode()), len(b.encode())],
        "share_32_byte_prefix": a.encode()[:32] == b.encode()[:32],
        "v1_codes_identical": bool(np.array_equal(ka, kb)),
        "v1_cosine": 1.0 if np.array_equal(ka, kb) else None,
        "elastic_codes_identical": bool(np.array_equal(ek.encode(a), ek.encode(b))),
        "paper_claim": "truncated tokens still receive distinct/unique embeddings",
        "verdict": "REFUTED" if np.array_equal(ka, kb) else "holds",
    }
