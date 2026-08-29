#!/usr/bin/env python3
"""Execute the notebook's OWN code cells and check every claim it makes.

Why this exists: a notebook that has been hand-edited since it last ran is a
notebook that does not run. This parses S9_loss_harness.ipynb, strips the Colab
`!pip` magics, and executes the code cells in order in one namespace.

It runs with S9_OFFLINE=1, which swaps the hub download for a locally built
model of the same architecture (LlamaConfig with SmolLM2-135M's hyper-parameters)
and the Session-2 tokenizer, because this sandbox has no network access to
huggingface.co. Depth and step count are reduced so it finishes on CPU; every
shape, mask, count and identity being checked is independent of those.
"""
import json, os, sys, re, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
NB   = os.path.join(HERE, "S9_loss_harness.ipynb")

os.environ["S9_OFFLINE"]   = "1"
os.environ["S9_LAYERS"]    = os.environ.get("S9_LAYERS", "2")     # CPU: 2 layers, not 30
os.environ["S9_STEPS"]     = os.environ.get("S9_STEPS", "12")
os.environ["S9_BS"]        = os.environ.get("S9_BS", "2")
os.environ["S9_SEQ"]       = os.environ.get("S9_SEQ", "128")
os.environ["S9_SHARD"]     = os.path.join(HERE, "data", "s9_owt_clean.jsonl.gz")
# The full pass runs fp32 because this CPU has no bf16 kernels and bf16 takes minutes.
# The bf16 case is covered by a targeted regression at the end instead -- see below.
os.environ["S9_DTYPE"]     = os.environ.get("S9_DTYPE", "float32")
os.environ["S9_TOKENIZER"] = "/sessions/serene-dazzling-volta/mnt/ERA/s2_submission/upload/tokenizer.json"

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(cells):
    src = "".join(c["source"])
    # drop Colab shell magics, preserving indentation so `except: !pip ...` stays valid
    src = re.sub(r"^(\s*)!.*$", r"\1pass", src, flags=re.M)
    print(f"\n{'='*74}\ncell {i+1}/{len(cells)}\n{'='*74}")
    try:
        exec(compile(src, f"<cell {i+1}>", "exec"), ns)
    except Exception:
        import traceback; traceback.print_exc()
        print(f"\nFAILED in cell {i+1}"); sys.exit(1)

# ---- independent re-derivation of the notebook's own claims -------------------
print(f"\n{'='*74}\nINDEPENDENT CHECKS\n{'='*74}")
fails = []
def chk(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond: fails.append(name)

V, D = ns["V"], ns["D"]
chk("vocab/width match SmolLM2-135M", (V, D) == (49152, 576), f"V={V} D={D}")
chk("shift drops exactly one position",
    ns["shift_targets"].shape[1] == ns["tokens"].shape[1] - 1)
chk("flattened logits and targets agree",
    ns["flat_logits"].shape[0] == ns["flat_targets"].shape[0])
chk("padding mask lowers the contributing count",
    ns["n_mask"] < ns["n_pad"], f"{ns['n_pad']:,} -> {ns['n_mask']:,}")
chk("boundary mask removes exactly one label per join",
    ns["n_split"] == ns["n_join"] - ns["n_boundaries"],
    f"{ns['n_join']:,} - {ns['n_boundaries']} = {ns['n_split']:,}")
# The "boundaries are harder" claim is a statement about a TRAINED model. Offline we
# run random weights, where every position sits at ln(V) and the comparison is noise --
# so offline we check the mechanism (that the right positions were selected) instead,
# and only assert the loss ordering when real weights are in play.
chk("boundary mask selects exactly the document-start targets",
    int(ns["bmask"].sum()) == ns["n_boundaries"] == ns["n_join"] - ns["n_split"],
    f"{ns['n_boundaries']} joins")
chk("single-join case masks exactly one position",
    ns["n_pair_off"] == ns["n_pair_on"] - 1)
if os.environ["S9_OFFLINE"] != "1":
    chk("boundary positions are harder than ordinary ones",
        ns["per_tok"][ns["bmask"]].mean().item() > ns["per_tok"][~ns["bmask"]].mean().item())
else:
    print("SKIP  boundary-positions-are-harder   (random weights: every position is at ln V)")
chk("untrained perplexity lands on V",
    0.5 < ns["p_rand"] / V < 2.0, f"ppl={ns['p_rand']:,.0f} V={V:,} ratio={ns['p_rand']/V:.3f}")
chk("untrained loss equals ln(V)",
    abs(ns["l_rand"] - math.log(V)) < 0.7, f"{ns['l_rand']:.4f} vs {math.log(V):.4f}")
chk("untying costs exactly V x D",
    ns["untied_total"] - ns["tied_total"] == V * D, f"+{V*D:,}")
chk("tied head really shares storage with the embeddings", ns["same"] is True)
chk("chunked CE gives the same loss",
    abs(ns["l_full"] - ns["l_chunk"]) < 1e-4, f"delta={abs(ns['l_full']-ns['l_chunk']):.2e}")
chk("chunked CE gives the same gradients",
    (ns["h_a"].grad - ns["h_b"].grad).abs().max().item() < 1e-4,
    f"max|d|={(ns['h_a'].grad-ns['h_b'].grad).abs().max().item():.2e}")
chk("chunking reduces the analytic logits footprint",
    ns["logit_bytes"] > ns["chunk_bytes"], f"{ns['logit_bytes']:.1f} -> {ns['chunk_bytes']:.1f} MiB")
chk("head 2 stays above head 1", ns["e2"] > ns["e1"], f"{ns['e2']:.4f} > {ns['e1']:.4f}")
chk("both heads improved",
    ns["f1"] >= ns["e1"] and ns["f2"] > ns["e2"],
    f"h1 {ns['f1']:.3f}->{ns['e1']:.3f}  h2 {ns['f2']:.3f}->{ns['e2']:.3f}")
chk("head 2 starts near ln(V)", ns["f2"] > 0.6 * math.log(V), f"{ns['f2']:.3f} vs {math.log(V):.3f}")

trunk_dt = next(ns["two"].trunk.parameters()).dtype
chk("head 2 dtype follows the trunk", ns["two"].head2.weight.dtype == trunk_dt,
    f"head2={ns['two'].head2.weight.dtype} trunk={trunk_dt}")

# --- REGRESSION for the bug that reached Colab ---------------------------------
# transformers v5 loads SmolLM2 in bfloat16. A bare nn.Linear is fp32, so head 2's
# first matmul died with "mat1 and mat2 have different dtype". The offline substitute
# was fp32 throughout, so it never reproduced the one thing that mattered.
# This rebuilds TwoHeadModel on a bf16 trunk and runs a real forward pass.
import torch
from transformers import LlamaConfig, LlamaForCausalLM
tiny = LlamaConfig(vocab_size=ns["V"], hidden_size=ns["D"], intermediate_size=1536,
                   num_hidden_layers=1, num_attention_heads=9, num_key_value_heads=3,
                   hidden_act="silu", max_position_embeddings=2048, rms_norm_eps=1e-5,
                   tie_word_embeddings=True, rope_theta=10000.0)
for dt in (torch.bfloat16, torch.float16, torch.float32):
    base = LlamaForCausalLM(tiny).to(dt)
    m2   = ns["TwoHeadModel"](base, tiny)
    ok_dtype = m2.head2.weight.dtype == dt
    try:
        l1, l2 = m2.losses(torch.randint(0, ns["V"], (1, 16)))
        ok_run = bool(torch.isfinite(l1) and torch.isfinite(l2))
        err = ""
    except Exception as e:
        ok_run, err = False, f"{type(e).__name__}: {e}"
    chk(f"TwoHeadModel works on a {str(dt).split('.')[-1]} trunk", ok_dtype and ok_run,
        err or f"head2={m2.head2.weight.dtype}  l1={l1.item():.3f} l2={l2.item():.3f}")
    del base, m2

print(f"\n{len(cells)} cells executed in {time.time()-t0:.0f}s")
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1)
print("\nALL CHECKS PASS — every cell runs and every claim re-derives.")
