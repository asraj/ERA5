"""Automated tests for the invariants that matter.

Runs on stdlib unittest (no pytest needed):
    python -m unittest discover -s tests -v
These tests build their own small system in a temp dir - they do not read the
demo's artifacts - so they prove the behaviour, not the output files.
"""
from __future__ import annotations
import os, sys, json, tempfile, unittest, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdes.tokenizer import FrozenTokenizer, sha256_str
from tdes.shards import ShardWriter, ShardStore, admission_check
from tdes.firewall import EvalFirewall
from tdes.mixture import default_schedule
from tdes.opus import Opus
from tdes.ledger import ConsumptionLedger, LearningLedger
from tdes.engine import Engine, CrashSignal
from tdes.packing import Packer


def tiny_world(tmp, steps=12):
    docs = {"general_web": [(f"w{i}", f"the system trained on document number {i} of plain web text " * 3)
                            for i in range(12)],
            "indic": [(f"i{i}", f"भारत एक देश है संख्या {i}। यहाँ अनेक भाषाएँ हैं। " * 3) for i in range(8)],
            "code": [(f"c{i}", f"def f{i}(x):\n    return x * {i}\n" * 3) for i in range(8)],
            "reasoning": [(f"r{i}", f"Question {i}. <|think|> step one then step two. <|think|> Answer: {i}.")
                          for i in range(8)],
            "agentic": [(f"a{i}", f"<|user|> task {i} <|assistant|> call tool(x={i}) <|tool|> obs {i} "
                                  f"<|assistant|> done {i}") for i in range(8)]}
    evals = [(f"e{i}", f"BENCHMARK {i}: capital question. Answer: X{i}. CANARY::t::{i}") for i in range(5)]
    tok = FrozenTokenizer.train([t for v in docs.values() for _, t in v] + [t for _, t in evals], 1024)
    tok.save(os.path.join(tmp, "tokenizer.json"))
    w = ShardWriter(tmp, tok, sha256_str("clean/1"))
    for lane, items in docs.items():
        w.write(f"{lane}-00", items, lane=lane, language="x", license="MIT", tier="tier-A")
    ev_man = w.write("eval-00", evals, lane="eval", language="en", license="MIT",
                     tier="tier-A", split="test", eval_overlap=True)
    store = ShardStore(tmp, tok.hash)
    fw = EvalFirewall()
    fw.register("eval-00", split="test", content_hash=ev_man.content_hash,
                benchmark_id="B1", tokens=store.tokens("eval-00"))
    sched = default_schedule(steps, seq_len=128)
    eng = Engine(tmp, store, sched, fw, tok, Opus(),
                 ConsumptionLedger(os.path.join(tmp, "ledgers", "c.jsonl")),
                 LearningLedger(os.path.join(tmp, "ledgers", "l.json")),
                 microbatch=2, samples_per_seq=3, log=lambda *_: None)
    return tok, store, fw, sched, eng, ev_man


class TestTokenizerAndShards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tokenizer_is_deterministic_and_faithful(self):
        tok, *_ = tiny_world(self.tmp)
        for s in ["भारत एक देश है।", "def f(x): return x", "plain english text 123!"]:
            self.assertEqual(tok.encode(s), tok.encode(s))
            self.assertEqual("".join(tok.decode(tok.encode(s)).split()), "".join(s.split()))

    def test_tokenizer_hash_changes_when_vocab_changes(self):
        tok, *_ = tiny_world(self.tmp)
        other = FrozenTokenizer(tok.vocab + ["<extra>"])
        self.assertNotEqual(tok.hash, other.hash)

    def test_shard_content_hash_detects_mutation(self):
        tok, store, *_ = tiny_world(self.tmp)
        sid = sorted(store.manifests)[0]
        self.assertTrue(store.verify(sid)[0])
        p = os.path.join(self.tmp, "shards", f"{sid}.bin")
        os.chmod(p, 0o666)
        with open(p, "ab") as f:
            f.write(b"\x01\x00\x00\x00")
        store._cache.pop(sid, None)
        self.assertFalse(store.verify(sid)[0])


class TestFirewall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_eval_shard_never_admitted(self):
        tok, store, fw, sched, eng, ev_man = tiny_world(self.tmp)
        self.assertFalse(fw.check_shard("eval-00")[0])
        self.assertFalse(admission_check(ev_man, tok.hash)[0])

    def test_eval_tokens_never_reach_a_batch(self):
        tok, store, fw, sched, eng, _ = tiny_world(self.tmp)
        self.assertFalse(fw.check_tokens(store.tokens("eval-00")[:100], "probe")[0])
        b = eng.builder.build("t", 0, "p0")
        for sq in b.sequences:
            self.assertTrue(fw.check_tokens(sq.tokens, "batch")[0])
        self.assertNotIn("eval-00", b.shard_ids)


class TestPackingMasks(unittest.TestCase):
    def test_pad_never_bears_loss_and_positions_reset(self):
        p = Packer(32, pad_id=0, eos_id=1)
        samples = [{"tokens": [5, 6, 7, 8], "sample_id": "a", "lane": "x", "ctx_len": 2},
                   {"tokens": [9, 10, 11], "sample_id": "b", "lane": "x", "ctx_len": 1}]
        sq = p.pack(samples, "structure_preserving")
        self.assertTrue(sq.validate(32, 0))
        self.assertEqual(sq.loss_mask[0], 0)          # context tokens masked out
        self.assertEqual(sq.loss_mask[2], 1)
        for t, m in zip(sq.tokens, sq.loss_mask):
            if t == 0:
                self.assertEqual(m, 0)

    def test_all_policies_produce_valid_sequences(self):
        p = Packer(24, pad_id=0, eos_id=1)
        s = [{"tokens": list(range(2, 12)), "sample_id": f"s{i}", "lane": "x"} for i in range(4)]
        for pol in ["pad_only", "concat_chop", "greedy", "best_fit",
                    "structure_preserving", "long_context"]:
            sq = p.pack(s, pol)
            self.assertTrue(sq.validate(24, 0), pol)


class TestMixture(unittest.TestCase):
    def test_floors_never_exceed_share_and_warmup_blends(self):
        sched = default_schedule(40, 128)
        for st in sched.stages:
            for lane, fl in st.protected_floors.items():
                self.assertGreaterEqual(st.mixture.get(lane, 0), fl)
        s2 = sched.stages[1]
        w_first, _ = sched.weights_for(s2.step_start)          # inside warmup
        w_late, _ = sched.weights_for(s2.step_end - 1)         # after warmup
        self.assertNotEqual(round(w_first["code"], 6), round(w_late["code"], 6))
        self.assertAlmostEqual(sum(w_first.values()), 1.0, places=6)


class TestDeterminismAndRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_batch_is_pure_function_of_coordinates(self):
        _, _, _, _, eng, _ = tiny_world(self.tmp)
        a = eng.builder.build("main", 7, "pv1")
        b = eng.builder.build("main", 7, "pv1")
        self.assertEqual(a.hash(), b.hash())
        self.assertEqual(a.token_span_ids, b.token_span_ids)
        self.assertNotEqual(a.hash(), eng.builder.build("main", 8, "pv1").hash())
        self.assertNotEqual(a.hash(), eng.builder.build("other", 7, "pv1").hash())

    def test_crash_resume_has_no_gap_or_duplicate(self):
        _, _, _, _, eng, _ = tiny_world(self.tmp, steps=12)
        with self.assertRaises(CrashSignal):
            eng.run("main", 0, 12, checkpoint_every=4, crash_at=6)
        pre = {r["global_step"]: r for r in eng.cl.read("main")}
        meta = eng.load_checkpoint("main-step0003")
        expected_hash = pre[4]["batch_hash"]
        eng.cl.rollback_to(meta["ledger_offset"])
        rebuilt = eng.builder.build("main", 4, pre[4]["proxy_version"])
        self.assertEqual(rebuilt.batch_id, meta["next_batch_id"])
        self.assertEqual(rebuilt.hash(), expected_hash)
        eng.run("main", 4, 12, checkpoint_every=4)
        ok, why = eng.cl.contiguous("main")
        self.assertTrue(ok, why)
        self.assertEqual([r["global_step"] for r in eng.cl.read("main")], list(range(12)))

    def test_replay_reproduces_hashes(self):
        _, _, _, _, eng, _ = tiny_world(self.tmp, steps=10)
        eng.run("main", 0, 10, checkpoint_every=5)
        rep = eng.replay("main", 1, 6)
        self.assertTrue(rep["all_match"], rep["mismatches"])
        self.assertTrue(all(c["token_spans_match"] for c in rep["checked"]))

    def test_fork_diverges_and_records_origin(self):
        _, _, _, _, eng, _ = tiny_world(self.tmp, steps=10)
        eng.run("main", 0, 10, checkpoint_every=5)
        fk = eng.fork("main-step0004", "branch-x")
        self.assertTrue(fk["streams_differ"])
        self.assertEqual(fk["origin_checkpoint"], "main-step0004")

    def test_checkpoint_binds_ledger_offset(self):
        _, _, _, _, eng, _ = tiny_world(self.tmp, steps=8)
        eng.run("main", 0, 8, checkpoint_every=4)
        with open(os.path.join(self.tmp, "checkpoints", "main-step0003.json")) as f:
            meta = json.load(f)
        # the offset must point exactly at the end of the 4th record: truncating
        # there leaves steps 0..3 and nothing else.
        eng.cl.rollback_to(meta["ledger_offset"])
        self.assertEqual([r["global_step"] for r in eng.cl.read("main")], [0, 1, 2, 3])
        self.assertEqual(meta["next_batch_id"], "main:4")


class TestOpus(unittest.TestCase):
    def test_decisions_are_deterministic_and_floor_overrides(self):
        o1, o2 = Opus(), Opus()
        self.assertEqual(o1.score("c1", "indic", "pv"), o2.score("c1", "indic", "pv"))
        d = o1.evaluate("cX", lane="indic", stage="s", shard_ids=["s1"], proxy_version="pv",
                        effective_tokens=10, floor_deficit=True)
        self.assertIn(d.status, ("accepted", "protected_override"))
        low = Opus(accept_rate=0.0)
        d2 = low.evaluate("cY", lane="indic", stage="s", shard_ids=["s1"], proxy_version="pv",
                          effective_tokens=10, floor_deficit=True)
        self.assertEqual(d2.status, "protected_override")
        self.assertTrue(d2.protected_floor_override)
        d3 = low.evaluate("cZ", lane="code", stage="s", shard_ids=["s1"], proxy_version="pv",
                          effective_tokens=10, floor_deficit=False)
        self.assertEqual(d3.status, "rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
