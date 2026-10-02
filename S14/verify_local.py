#!/usr/bin/env python3
"""Execute the notebook's OWN code cells offline and re-derive every checkable claim.

Same discipline as S9-S13: parses S14_upcycle.ipynb and runs the actual cells, so a notebook
edited since it last ran cannot pass. S14_OFFLINE=1 swaps in a miniature model and a synthetic
bigram language so both phases and all three branches run on a CPU in well under a minute.
"""
import json, os, sys, re, math, time, shutil, tempfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "S14_upcycle.ipynb")
RUN = tempfile.mkdtemp(prefix="s14_verify_")
os.environ.update(S14_OFFLINE="1", S14_RUN_DIR=RUN)

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(cells):
    src = re.sub(r"^(\s*)!.*$", r"\1pass", "".join(c["source"]), flags=re.M)
    print(f"\n{'='*72}\ncell {i+1}/{len(cells)}\n{'='*72}", flush=True)
    try:
        exec(compile(src, f"<cell {i+1}>", "exec"), ns)
    except Exception:
        import traceback; traceback.print_exc(); print(f"\nFAILED in cell {i+1}"); sys.exit(1)
WALL = time.time() - t0

print(f"\n{'='*72}\nINDEPENDENT CHECKS\n{'='*72}")
fails = []
def chk(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond: fails.append(name)

import numpy as np, torch
CFG, T = ns["CFG"], ns["T"]
L, d, E, k, V = CFG["layers"], CFG["d"], CFG["experts"], CFG["topk"], CFG["vocab"]

# --- data ----------------------------------------------------------------------------------
man = ns["MANIFEST"]
path = [os.path.join(ns["DATA_DIR"], f) for f in os.listdir(ns["DATA_DIR"]) if f.endswith(".bin")][0]
chk("data file matches its manifest sha256", hashlib.sha256(open(path, "rb").read()).hexdigest() == man["sha256"])
perm = ns["PERM"]
chk("one epoch: the training order is a permutation", len(set(perm.tolist())) == len(perm) == ns["N_CHUNKS"])
B = CFG["batch"]; p1 = set(perm[:ns["STEPS_DENSE"] * B].tolist()); p2 = set(perm[ns["STEPS_DENSE"] * B:ns["STEPS_TOTAL"] * B].tolist())
chk("phase 2 trains on tokens phase 1 never saw", not (p1 & p2) and len(p2) > 0)

# --- parameter counts, from the shapes -------------------------------------------------------
def counts(V, d, L, T_, E, k):
    dense = V * d + T_ * d + L * (12 * d * d + 4 * d) + 2 * d
    return dense, dense + L * ((E - 1) * 8 * d * d + d * E), dense + L * ((k - 1) * 8 * d * d + d * E)
cd, cm, ca = counts(V, d, L, T, E, k)
chk("dense / MoE total / MoE active parameter counts match the shapes",
    (ns["N_DENSE"], ns["N_MOE"], ns["N_ACTIVE"]) == (cd, cm, ca), f"{cd:,} / {cm:,} / {ca:,}")
rd, rm, ra = counts(50304, 256, 10, 512, 8, 2)
chk("the Colab configuration: 20.88M dense -> 57.60M total, 26.15M active",
    (rd, rm, ra) == (20_883_968, 57_604_608, 26_147_328), f"{rd:,} -> {rm:,} / {ra:,}")

# --- conversion ----------------------------------------------------------------------------
ck = ns["CHECK"]
chk("r = 0 conversion reproduces the dense loss", ck["exact_loss_gap"] < 1e-5, f"{ck['exact_loss_gap']:.1e}")
chk("r = 0 conversion reproduces one layer's output", ck["exact_ffn_gap"] < 1e-5, f"{ck['exact_ffn_gap']:.1e}")
chk("drop-upcycling redraws exactly r*H neurons per expert", ck["drop_sizes"] == [int(round(CFG["drop"] * 4 * d))])
chk("kept neurons are bit-identical to the dense FFN", ck["kept_exact"] is True)
chk("two experts' redrawn sets overlap ~r^2", abs(ck["overlap"] - CFG["drop"] ** 2) < 0.06, f"{ck['overlap']:.3f}")
chk("bias forces selection", ck["bias_forces"] == 1.0)
chk("bias leaves router probabilities and weights untouched",
    ck["probs_unchanged"] == 0.0 and ck["bias_weights_gap"] < 1e-6)
chk("aux term: 1 when even, N at top-1 collapse, N/2 at top-2 collapse",
    abs(ck["aux_even"] - 1) < 1e-6 and abs(ck["aux_collapse_k1"] - E) < 1e-6 and abs(ck["aux_collapse_k2"] - E / 2) < 1e-6)

# independent: re-do an r=0 upcycle and compare every expert with the dense FFN directly
LM, upcycle = ns["LM"], ns["upcycle"]
torch.manual_seed(3); dn = LM(); sd = {k_: v.detach() for k_, v in dn.state_dict().items()}
m0, _ = upcycle(sd, "bias", r=0.0)
same = all(torch.equal(m0.blocks[l].ffn.experts[e].fc.weight, dn.blocks[l].ffn.fc.weight) and
           torch.equal(m0.blocks[l].ffn.experts[e].fc2.weight, dn.blocks[l].ffn.fc2.weight)
           for l in range(L) for e in range(E))
chk("independent: with r = 0 every expert IS the dense FFN, in every layer", same)
m5, masks = upcycle(sd, "bias", r=0.5)
w_old = dn.blocks[0].ffn.fc.weight.detach()[masks[(0, 0)]]
w_new = m5.blocks[0].ffn.experts[0].fc.weight.detach()[masks[(0, 0)]]
chk("independent: redrawn weights match the replaced ones' mean and std",
    abs(w_new.std() / w_old.std() - 1) < 0.1 and abs(w_new.mean() - w_old.mean()) < 3 * w_old.std() / math.sqrt(w_old.numel()),
    f"std ratio {(w_new.std()/w_old.std()).item():.3f}")
chk("independent: non-FFN weights are copied unchanged",
    torch.equal(m5.blocks[0].attn.qkv.weight, dn.blocks[0].attn.qkv.weight) and torch.equal(m5.emb.weight, dn.emb.weight))

# --- runs ----------------------------------------------------------------------------------
P1, A, B_, C_ = ns["P1"], ns["A"], ns["B_"], ns["C_"]
chk("phase 2 branches cover identical steps", (A["s0"], A["s1"]) == (B_["s0"], B_["s1"]) == (C_["s0"], C_["s1"]))
chk("phase 2 starts where phase 1 ended", A["s0"] == P1["s1"] == ns["STEPS_DENSE"])
chk("both MoE branches start from the identical upcycled model", B_["start_val"] == C_["start_val"])
for r in (P1, A, B_, C_):
    chk(f"{r['name']}: finite loss, lower at the end than at the start",
        math.isfinite(r["final_val"]) and r["final_val"] < r["start_val"], f"{r['start_val']:.3f} -> {r['final_val']:.3f}")
chk("reported conversion jump re-derives", abs(ns["JUMP_B"] - (B_["start_val"] - A["start_val"])) < 1e-12)
chk("reported same-token gain re-derives", abs(ns["GAIN_B"] - (A["final_val"] - B_["final_val"])) < 1e-12)
load_state = ns["load_state"]
sb, sc = load_state("phase2_B_moe_bias"), load_state("phase2_C_moe_aux")
bb = torch.stack([v for k_, v in sb.items() if k_.endswith("ffn.bias")])
cb = torch.stack([v for k_, v in sc.items() if k_.endswith("ffn.bias")])
chk("bias run: the biases moved; aux run: they never did", bb.abs().sum() > 0 and cb.abs().sum() == 0,
    f"|bias| B {bb.abs().mean():.4f}, C {cb.abs().mean():.4f}")
chk("bias updates are whole multiples of gamma",
    torch.allclose(bb / CFG["gamma"], (bb / CFG["gamma"]).round(), atol=1e-3))
fc_ = torch.tensor(B_["final_counts"])
chk("final MaxVio and dead count re-derive from the saved counts", tuple(ns["load_stats"](fc_)) == tuple(B_["final_load"]))

# --- a disconnect, simulated on an MoE run: the biases must survive the checkpoint ----------
tp, Interrupted = ns["train_phase"], ns["Interrupted"]
build = lambda: upcycle(ns["DENSE_STATE"], "bias")[0]
s0, s1 = ns["STEPS_DENSE"], ns["STEPS_TOTAL"]
ref = tp("verify_ref", build, s0, s1, CFG["lr"], rewarm=CFG["rewarm"], ckpt_every_steps=10)
cut = s0 + (s1 - s0) // 2 + 3
try:
    tp("verify_cut", build, s0, s1, CFG["lr"], rewarm=CFG["rewarm"], ckpt_every_steps=10, _interrupt_at=cut)
    cutoff = False
except Interrupted:
    cutoff = True
chk("simulated disconnect left a checkpoint", cutoff and os.path.exists(os.path.join(RUN, "verify_cut.ckpt.pt")))
res = tp("verify_cut", build, s0, s1, CFG["lr"], rewarm=CFG["rewarm"], ckpt_every_steps=10)
chk("resumed MoE run is bit-identical: final loss", res["resumes"] == 1 and res["final_val"] == ref["final_val"],
    f"{res['final_val']!r}")
chk("resumed MoE run is bit-identical: every training loss and load point",
    res["curve"] == ref["curve"] and res["load"] == ref["load"])
chk("resumed MoE run ends with bit-identical biases",
    all(torch.equal(v, load_state("verify_ref")[k_]) for k_, v in load_state("verify_cut").items() if k_.endswith("ffn.bias")))

# --- stale-literal guard on the findings cell ------------------------------------------------
LESSON = {str(i) for i in range(0, 13)}
summary = "".join(cells[-1]["source"])
lits = set()
for m in re.finditer(r"print\(\s*(f?)(\"|')(.*?)\2", summary, re.S):
    if m.group(1): continue
    lits |= set(re.findall(r"\d[\d,]*\.?\d*", m.group(3)))
stray = sorted(t for t in lits if t.replace(",", "") not in LESSON)
chk("no run-dependent number is typed into the findings cell", not stray, f"stray: {stray}" if stray else "")
chk("the findings cell states no fixed verdict",
    "beats the dense model" not in summary.replace('"beats"', "") and "experts diverge" not in summary)

shutil.rmtree(RUN, ignore_errors=True)
print(f"\n{len(cells)} cells executed in {WALL:.0f}s")
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1)
print("\nALL CHECKS PASS - every cell runs and every claim re-derives.")
