#!/usr/bin/env python3
"""One command that reproduces every number in the paper.

    python run_all.py            # ~2 minutes, writes results/

Problem 3 from Session 7: the 32-position window wastes space on short tokens and
hard-crops long ones. This runs the audits and the training proofs for the
Elastic Kronecker codec proposed as the fix.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv2.vocab import load_words
from kv2.codec import KroneckerV1, ElasticKronecker, code_key
from kv2 import audit
from kv2.transformer import gradcheck
from kv2.experiments import discrimination_probe, lm_experiment
from kv2 import validate_v1 as V1CHECK

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
V5_VOCAB, V5_WIDTH = 131072, 8096          # the V5 reference shape from the lesson


def scale_table(v1, ek):
    dense = V5_VOCAB * V5_WIDTH
    return {
        "reference_shape": {"vocab": V5_VOCAB, "d_model": V5_WIDTH},
        "dense_table_params": dense,
        "kronecker_v1_params": v1.dim * V5_WIDTH,
        "elastic_params": ek.dim * V5_WIDTH,
        "v1_code_dim": v1.dim, "elastic_code_dim": ek.dim,
        "elastic_vs_v1_saving_pct": round(100 * (1 - ek.dim / v1.dim), 2),
        "elastic_vs_dense_saving_pct": round(100 * (1 - (ek.dim * V5_WIDTH) / dense), 3),
        "v1_vs_dense_saving_pct": round(100 * (1 - (v1.dim * V5_WIDTH) / dense), 3),
    }


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    v1, ek = KroneckerV1(), ElasticKronecker()
    words, source = load_words(60000)
    log = []

    def say(m):
        print(m); log.append(m)

    say("=" * 72)
    say("Kronecker V2 - Problem 3: the 32-position budget")
    say(f"corpus: {source} | distinct words: {len(words):,}")
    say("=" * 72)

    # ---- V0: validation against the published V1 paper ----
    fid = V1CHECK.fidelity()
    say("\n[V0] FIDELITY vs Kronecker Embeddings V1 (Shravan, 2026) spec")
    say("  " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in list(fid.items())[:5]))
    say(f"  all spec properties reproduced: {fid['ALL_PASS']}")
    lex = V1CHECK.lesson_example()
    say(f"\n[V1] PAPER CLAIM TEST - 'truncated tokens still receive distinct embeddings'")
    say(f"  pair {lex['pair'][0]} / {lex['pair'][1]}  bytes={lex['bytes']}")
    say(f"  share 32-byte prefix: {lex['share_32_byte_prefix']}  ->  v1 codes identical: "
        f"{lex['v1_codes_identical']}  (elastic identical: {lex['elastic_codes_identical']})")
    say(f"  VERDICT: claim {lex['verdict']}")
    claim_word = V1CHECK.test_uniqueness_claim([w for w, _, _ in words], "wikipedia-words")
    say(f"  on {claim_word['size']} wikipedia words: {claim_word['v1_collision_groups']} collision groups, "
        f"{claim_word['v1_tokens_sharing_a_vector']} tokens share a vector (elastic: "
        f"{claim_word['elastic_collision_groups']})")
    trunc = V1CHECK.truncation_rate([w for w, _, _ in words])
    say(f"  truncation coverage (paper Table 4.2 metric): {trunc}")

    # ---- A1 byte budget ----
    bb = audit.byte_budget(words)
    say("\n[A1] BYTE BUDGET - what 32 bytes actually buys, per script")
    say(f"  {'script':12}{'words':>7}{'B/char':>8}{'chars fit':>11}{'>32B':>7}{'%':>7}")
    for s, r in bb.items():
        say(f"  {s:12}{r['words']:>7}{r['bytes_per_char']:>8}{r['chars_that_fit_in_32B']:>11}"
            f"{r['words_over_32B']:>7}{r['pct_over_32B']:>7}")

    # ---- A2 collisions ----
    say("\n[A2] COLLISION AUDIT - words the codec maps to ONE vector, permanently")
    col = {}
    for c in (v1, ek):
        r = audit.collisions(words, c); col[c.name] = r
        say(f"  {c.name:20} groups={r['collision_groups']:5}  words_affected={r['words_affected']:5}"
            f"  ({r['pct_words_affected']}%)  per_script={r['per_script']}")
    for ex in col[v1.name]["examples"][:4]:
        say("      v1 collides: " + "  ==  ".join(ex))

    # adversarial transposition sweep (elastic's own worst case)
    rng = np.random.default_rng(0); adv = {"v1": 0, "elastic": 0, "n": 0}
    for _ in range(3000):
        n = int(rng.integers(20, 60)); s = [chr(int(rng.integers(97, 123))) for _ in range(n)]
        i, j = rng.choice(n, 2, replace=False)
        if s[i] == s[j]:
            continue
        adv["n"] += 1
        t = s[:]; t[i], t[j] = t[j], t[i]
        a, b = "".join(s), "".join(t)
        adv["v1"] += code_key(v1.encode(a)) == code_key(v1.encode(b))
        adv["elastic"] += code_key(ek.encode(a)) == code_key(ek.encode(b))
    say(f"  adversarial transpositions ({adv['n']} genuine): v1={adv['v1']}  elastic={adv['elastic']}")

    # ---- A3 occupancy, A4 geometry ----
    occ = {c.name: audit.occupancy(words, c) for c in (v1, ek)}
    geo = {c.name: audit.geometry(c) for c in (v1, ek)}
    say("\n[A3] OCCUPANCY / TRUNCATION")
    for k, r in occ.items():
        say(f"  {k:20} dim={r['dim']:5} cells_used={r['mean_cells_used_pct']}%  "
            f"bytes_dropped={r['total_bytes_dropped']}  words_truncated={r['words_truncated']}")
    say("\n[A4] GEOMETRY - is 'similar spelling => similar vector' preserved?")
    for k, r in geo.items():
        say(f"  {k:20} related={r['mean_cos_related']}  unrelated={r['mean_cos_unrelated']}"
            f"  separation={r['separation']}")

    # ---- scale ----
    sc = scale_table(v1, ek)
    say(f"\n[A5] PARAMETERS at the V5 reference shape ({V5_VOCAB}x{V5_WIDTH})")
    say(f"  dense table       {sc['dense_table_params']:>15,}")
    say(f"  kronecker-v1      {sc['kronecker_v1_params']:>15,}   ({sc['v1_vs_dense_saving_pct']}% vs dense)")
    say(f"  elastic (ours)    {sc['elastic_params']:>15,}   ({sc['elastic_vs_dense_saving_pct']}% vs dense,"
        f" {sc['elastic_vs_v1_saving_pct']}% vs v1)")

    # ---- gradcheck ----
    gc = gradcheck()
    say(f"\n[P0] GRADCHECK (backprop correctness): "
        + "  ".join(f"{k}: {v['max_relative_error']:.2e} {'PASS' if v['passed'] else 'FAIL'}"
                    for k, v in gc.items()))

    # ---- E1 discrimination probe ----
    pairs = audit.find_collision_pairs(words, v1, want=3)
    say("\n[E1] DISCRIMINATION PROBE - train a transformer to tell colliding words apart")
    for a, b, s in pairs:
        say(f"  pair ({s}): {a}  vs  {b}")
    probe = discrimination_probe(pairs, {"dense": None, "kronecker_v1": v1, "elastic": ek},
                                 steps=400)
    say(f"  {'input path':16}{'mean val acc':>14}   (chance = 0.500)")
    for k, v in probe["summary"].items():
        say(f"  {k:16}{v['mean_val_acc']:>14.3f}   per_pair={v['per_pair']}")

    # ---- E2 language modelling ----
    say("\n[E2] LANGUAGE MODEL on real multilingual text (no-regression check)")
    lm = lm_experiment(words, {"dense": None, "kronecker_v1": v1, "elastic": ek}, steps=300)
    say(f"  {'input path':16}{'val_loss':>10}{'val_acc':>9}")
    for k, v in lm["results"].items():
        say(f"  {k:16}{v['final_val_loss']:>10.4f}{v['final_val_acc']:>9.3f}")

    # ---- verdict ----
    v1c, ekc = col[v1.name], col[ek.name]
    verdict = {
        "problem": "3 - dynamic length / 32-position waste",
        "v1_collisions_removed": v1c["words_affected"] - ekc["words_affected"],
        "elastic_collisions": ekc["words_affected"],
        "elastic_truncation": occ[ek.name]["total_bytes_dropped"],
        "code_dim_saving_pct": sc["elastic_vs_v1_saving_pct"],
        "probe_v1_acc": probe["summary"]["kronecker_v1"]["mean_val_acc"],
        "probe_elastic_acc": probe["summary"]["elastic"]["mean_val_acc"],
        "geometry_preserved": geo[ek.name]["separation"] > 0,
        "lm_no_regression": lm["results"]["elastic"]["final_val_loss"]
                            <= lm["results"]["kronecker_v1"]["final_val_loss"] + 0.1,
        "gradcheck_passed": all(v["passed"] for v in gc.values()),
        "v1_reimplementation_faithful": fid["ALL_PASS"],
        "v1_uniqueness_claim_refuted": lex["verdict"] == "REFUTED",
    }
    verdict["all_claims_hold"] = bool(
        verdict["v1_reimplementation_faithful"] and verdict["v1_uniqueness_claim_refuted"]
        and verdict["elastic_collisions"] == 0 and verdict["elastic_truncation"] == 0
        and verdict["probe_v1_acc"] < 0.6 and verdict["probe_elastic_acc"] > 0.9
        and verdict["geometry_preserved"] and verdict["lm_no_regression"]
        and verdict["gradcheck_passed"] and verdict["code_dim_saving_pct"] > 0)

    say("\n" + "=" * 72)
    for k, v in verdict.items():
        say(f"  {k:28} {v}")
    say("=" * 72)
    say(f"  completed in {time.time() - t0:.0f}s")

    json.dump({"source": source, "n_words": len(words),
               "v1_fidelity": fid, "v1_claim_lesson_example": lex,
               "v1_claim_on_words": claim_word, "v1_truncation_rate": trunc,
               "byte_budget": bb,
               "collisions": col, "adversarial_transpositions": adv, "occupancy": occ,
               "geometry": geo, "scale": sc, "gradcheck": gc,
               "probe": probe, "lm": lm, "verdict": verdict},
              open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=2)
    open(os.path.join(OUT, "run.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    return 0 if verdict["all_claims_hold"] else 1


if __name__ == "__main__":
    sys.exit(main())
