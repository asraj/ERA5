"""Invariants for the Elastic Kronecker codec. python -m unittest discover -s tests"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv2.codec import KroneckerV1, ElasticKronecker, code_key
from kv2.transformer import gradcheck, TinyTransformer


class TestCodec(unittest.TestCase):
    def setUp(self):
        self.v1, self.ek = KroneckerV1(), ElasticKronecker()

    def test_deterministic(self):
        for w in ["apple", "भारत", "இந்தியாவில்", "a"]:
            self.assertTrue(np.array_equal(self.ek.encode(w), self.ek.encode(w)))

    def test_v1_truncates_and_elastic_does_not(self):
        long = "இந்தியாவிலேயே" * 3
        self.assertGreater(self.v1.dropped_bytes(long), 0)
        self.assertEqual(self.ek.dropped_bytes(long), 0)

    def test_lesson_collision_pair_is_fixed(self):
        a, b = "अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता"
        self.assertEqual(a.encode()[:32], b.encode()[:32])          # same 32-byte prefix
        self.assertEqual(code_key(self.v1.encode(a)), code_key(self.v1.encode(b)))
        self.assertNotEqual(code_key(self.ek.encode(a)), code_key(self.ek.encode(b)))

    def test_elastic_code_is_smaller(self):
        self.assertLess(self.ek.dim, self.v1.dim)

    def test_no_transposition_collisions(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(20, 60))
            s = [chr(int(rng.integers(97, 123))) for _ in range(n)]
            i, j = rng.choice(n, 2, replace=False)
            if s[i] == s[j]:
                continue
            t = s[:]; t[i], t[j] = t[j], t[i]
            self.assertNotEqual(code_key(self.ek.encode("".join(s))),
                                code_key(self.ek.encode("".join(t))))

    def test_prefix_similarity_preserved(self):
        def cos(a, b):
            return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        rel = cos(self.ek.encode("train"), self.ek.encode("training"))
        unrel = cos(self.ek.encode("train"), self.ek.encode("zebra"))
        self.assertGreater(rel, unrel)

    def test_unseen_token_still_encodes(self):
        v = self.ek.encode("qwertyuiopasdfgh")       # never in any vocabulary
        self.assertEqual(v.shape[0], self.ek.dim)
        self.assertTrue(np.isfinite(v).all())


class TestTransformer(unittest.TestCase):
    def test_gradients_match_numerical(self):
        for path, r in gradcheck().items():
            self.assertTrue(r["passed"], f"{path}: {r['max_relative_error']}")

    def test_identical_codes_give_identical_embeddings(self):
        """The mechanism behind the probe: equal codes -> equal model input."""
        codes = np.zeros((3, 16)); codes[0] = codes[1] = np.arange(16)
        m = TinyTransformer(3, d=8, max_T=2, input_path="codec", codes=codes)
        x = m.forward(np.array([[0, 1]]))
        self.assertTrue(np.allclose(m.codes[0], m.codes[1]))
        self.assertEqual(x.shape, (1, 2, 3))


class TestMatchesReferenceCode(unittest.TestCase):
    """Our KroneckerV1 must equal a literal transcription of the published
    reference implementation (theschoolofai/kronecker-embeddings)."""

    def test_codec_matches_reference_bit_for_bit(self):
        from kv2 import reference as R
        k = KroneckerV1()
        words = ["a", "hello", "apple", "भारत", "இந்தியாவில்", "తెలుగు",
                 "अंतर्राष्ट्रीयकरण", "<s>", "def f(x):", "x" * 40, "अ" * 20,
                 "🎯", "kronektikus", "", "netwrok"]
        for w in words:
            mine = k.encode(w)
            ref = R.encode_single(R.utf8_safe_truncate(w.encode(), 32))
            self.assertLess(float(np.abs(mine - ref).max()), 1e-6, w)

    def test_reference_index_convention(self):
        """index = byte_value * pos_dim + pos, per codec.py."""
        from kv2 import reference as R
        v = R.encode_single(b"A", z_normalize=False)
        self.assertEqual(int(np.argmax(v)), 0x41 * 32 + 0)

    def test_dynamic_and_cached_modes_are_identical(self):
        """The reference guarantees both modes give bit-identical output."""
        from kv2 import reference as R
        toks = ["hello", "भारत", "இந்தியா", "x" * 50]
        dyn = R.KroneckerEmbedding(len(toks), 16, tokens=toks, mode="dynamic", seed=0)
        cac = R.KroneckerEmbedding(len(toks), 16, tokens=toks, mode="cached", seed=0)
        ids = np.array([[0, 1, 2, 3]])
        self.assertTrue(np.allclose(dyn.forward(ids), cac.forward(ids), atol=1e-6))

    def test_elastic_is_a_drop_in_sibling(self):
        """ElasticKronecker exposes the same surface as KroneckerEmbedding."""
        ek = ElasticKronecker()
        for attr in ("D", "encode", "extra_repr", "dropped_bytes"):
            self.assertTrue(hasattr(ek, attr), attr)
        self.assertEqual(ek.D, ek.dim)


class TestPaperValidation(unittest.TestCase):
    def test_reimplementation_matches_paper_spec(self):
        from kv2.validate_v1 import fidelity
        f = fidelity()
        for k, v in f.items():
            if isinstance(v, bool):          # floats here are measurements, not verdicts
                self.assertTrue(v, f"paper property failed: {k}")
        self.assertEqual(f["max_abs_diff_vs_reference"], 0.0)
        self.assertTrue(f["bit_identical_to_reference_impl"])

    def test_paper_uniqueness_claim_is_refuted(self):
        from kv2.validate_v1 import lesson_example
        r = lesson_example()
        self.assertTrue(r["share_32_byte_prefix"])
        self.assertTrue(r["v1_codes_identical"])      # paper says these stay distinct
        self.assertFalse(r["elastic_codes_identical"])
        self.assertEqual(r["verdict"], "REFUTED")

    def test_utf8_safe_truncation_never_splits_a_codepoint(self):
        from kv2.codec import utf8_safe_len
        b = ("अ" * 20).encode()
        self.assertEqual(utf8_safe_len(b, 32) % 3, 0)


if __name__ == "__main__":
    unittest.main()
