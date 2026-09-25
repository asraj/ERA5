#!/usr/bin/env python3
"""Execute the notebook's OWN code cells offline and re-derive every checkable claim.

Same discipline as S9-S12: this parses S13_reversible.ipynb and runs the actual cells, so a
notebook edited since it last ran cannot pass. S13_OFFLINE=1 swaps in a miniature model and a
synthetic bigram language so the whole pipeline (data, checks, screen, three runs, probe, results)
runs on a CPU in about a minute. The GPU numbers come from the Colab run; what this proves is that
the code is correct and that every printed claim is derived, not typed.
"""
import json, os, sys, re, math, time, shutil, tempfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "S13_reversible.ipynb")
RUN = tempfile.mkdtemp(prefix="s13_verify_")
os.environ.update(S13_OFFLINE="1", S13_RUN_DIR=RUN)

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

# --- data ----------------------------------------------------------------------------------
man = ns["MANIFEST"]
path = [os.path.join(ns["DATA_DIR"], f) for f in os.listdir(ns["DATA_DIR"]) if f.endswith(".bin")][0]
chk("data file matches its manifest sha256",
    hashlib.sha256(open(path, "rb").read()).hexdigest() == man["sha256"])
perm = ns["PERM"]
chk("training order is a permutation: every window exactly once",
    len(perm) == ns["N_CHUNKS"] and len(set(perm.tolist())) == ns["N_CHUNKS"])
chk("validation and training are disjoint slices",
    len(ns["VAL"]) + len(ns["TRAIN"]) == man["tokens"])

# --- parameter count, from the shapes, for the miniature and for the real config ------------
def count(V, d, L, T_):
    return V * d + T_ * d + L * (12 * d * d + 4 * d) + 2 * d
chk("parameter count matches V*d + T*d + L*(12d^2+4d) + 2d",
    ns["N_PARAMS"] == count(CFG["vocab"], CFG["d"], CFG["layers"], T), f"{ns['N_PARAMS']:,}")
real = count(50304, 256, 10, 512)
chk("the Colab configuration is ~20.9M parameters", abs(real / 1e6 - 20.88) < 0.01, f"{real:,}")

# --- fused head ----------------------------------------------------------------------------
ce = ns["CE_CHECK"]
chk("fused head: loss equals F.cross_entropy", ce["loss"] < 1e-5, f"{ce['loss']:.1e}")
chk("fused head: both gradients match", ce["gh"] < 1e-5 and ce["gW"] < 1e-5,
    f"{ce['gh']:.1e}, {ce['gW']:.1e}")

# --- the reversible backward ---------------------------------------------------------------
gc_ = ns["GRAD_CHECK"]
orc = max(v["oracle"] for (i, s, a), v in gc_.items() if not a)
reb = max(v["rebuilt"] for (i, s, a), v in gc_.items() if not a)
chk("with true states, RevStack gradients equal autograd to fp32 rounding", orc < 5e-6, f"{orc:.1e}")
chk("with rebuilt states the gap is small and larger than the oracle gap",
    reb < 1e-2 and reb >= orc, f"{reb:.1e}")
chk("every reversible variant was checked at both weight scales",
    {(i, s) for (i, s, a) in gc_} == {(i, s) for i in ns["REV_VARIANTS"] for s in (1.0, 4.0)})

# independent: RevStack keeps exactly two states, whatever the depth; plain autograd grows
LM, SavedBytes, RevStack, run_plain = ns["LM"], ns["SavedBytes"], ns["RevStack"], ns["run_plain"]
def saved_stack(layers, reversible, B=4):
    old = CFG["layers"]; CFG["layers"] = layers
    try:
        torch.manual_seed(0); m = LM("midpoint_a")
        x = torch.randint(0, CFG["vocab"], (B, T))
        p0 = m.embed(x)
        with SavedBytes(m) as sb:
            if reversible: out = RevStack.apply(p0, m, *m.rev_params)
            else:          out = run_plain(p0, m.blocks, m.integ)
        return sb.bytes
    finally:
        CFG["layers"] = old
B = 4; state = B * T * CFG["d"] * 4
r2, r8 = saved_stack(2, True, B), saved_stack(8, True, B)
p2, p8 = saved_stack(2, False, B), saved_stack(8, False, B)
chk("RevStack saves exactly two fp32 states", r8 == 2 * state, f"{r8:,} = 2 x {state:,}")
chk("RevStack memory does not depend on depth (2 vs 8 layers)", r2 == r8)
chk("plain autograd memory grows with depth", p8 > 3 * p2, f"{p2:,} -> {p8:,}")

# --- the linear-stability table, re-derived with numpy -------------------------------------
bg = ns["backward_growth"]; I = ns["INTEGRATORS"]
chk("midpoint: backward growth 1 (marginal)", abs(bg(I["midpoint"])[0] - 1) < 1e-9)
chk("midpoint(a) at a=0.5: backward growth 2", abs(bg(I["midpoint_a"])[0] - 2) < 1e-9)
chk("leapfrog: a double root at 1", bg(I["leapfrog"])[1] and abs(bg(I["leapfrog"])[0] - 1) < 1e-6)
rc = ns["RECON"]
chk("reconstruction was measured for every variant",
    all((k, "fp32", 1.0) in rc for k in ns["REV_VARIANTS"]))
chk("measured midpoint(a) growth at init is near its predicted x2",
    1.4 < rc[("midpoint_a", "fp32", 1.0)]["growth"] < 2.8,
    f"x{rc[('midpoint_a','fp32',1.0)]['growth']:.2f}")

# --- memory anatomy ------------------------------------------------------------------------
M, B0 = ns["MEM"], ns["B0"]
chk("same rule: RevStack saves less than plain autograd",
    M["midpoint(a), RevStack"][B0]["saved"] < M["midpoint(a), plain autograd"][B0]["saved"],
    f"{ns['SAVED_RATIO']:.1f}x")
chk("the rule alone changes nothing: baseline and midpoint(a) plain save the same",
    abs(M["baseline, fused head"][B0]["saved"] - M["midpoint(a), plain autograd"][B0]["saved"])
    / M["baseline, fused head"][B0]["saved"] < 0.05)
chk("the naive head saves more than the fused head",
    M["baseline, naive head"][B0]["saved"] > M["baseline, fused head"][B0]["saved"])

# --- runs ------------------------------------------------------------------------------------
S, W = ns["SCREEN"], ns["WINNER"]
ok = {k: v for k, v in S.items() if k != "baseline" and not v["diverged"]}
chk("the winner is the lowest non-diverged screen loss", W == min(ok, key=lambda k: ok[k]["final_val"]), W)
R1, R2, R3 = ns["RUN1"], ns["RUN2"], ns["RUN3"]
chk("runs 1 and 2 used identical batch, steps and tokens",
    (R1["batch"], R1["steps"], R1["tokens"]) == (R2["batch"], R2["steps"], R2["tokens"]))
chk("run 3 used a larger batch than run 2", R3["batch"] > R2["batch"])
chk("run 3 learning rate follows sqrt scaling capped at 3x",
    abs(R3["lr"] - CFG["lr"] * min(math.sqrt(R3["batch"] / CFG["batch"]), 3.0)) < 1e-12)
for r in (R1, R2, R3):
    chk(f"{r['name']}: finite final loss and a result file on disk",
        math.isfinite(r["final_val"]) and os.path.exists(os.path.join(RUN, r["name"] + ".json")))
chk("reported overhead re-derives from the saved results",
    abs(ns["OVERHEAD"] - (R1["tok_s"] / R2["tok_s"] - 1)) < 1e-12)
chk("reported max-batch throughput ratio re-derives", abs(ns["THRU"] - R3["tok_s"] / R2["tok_s"]) < 1e-12)
chk("resume works: re-running a finished run loads it instead of training",
    ns["train_run"](R1["name"], "baseline", R1["batch"], CFG["train_tokens"], CFG["lr"])["tok_s"] == R1["tok_s"])

# --- a disconnect, simulated: interrupt mid-run, resume, compare with an uninterrupted run ---
tr, Interrupted = ns["train_run"], ns["Interrupted"]
kw = dict(integ="midpoint_a", batch=CFG["batch"], tokens=CFG["train_tokens"], lr=CFG["lr"],
          ckpt_every_steps=10)
ref = tr("verify_uninterrupted", **kw)
cut = ref["steps"] // 2 + 3                       # three steps past a checkpoint: those are redone
try:
    tr("verify_interrupted", **kw, _interrupt_at=cut)
    interrupted = False
except Interrupted:
    interrupted = True
ck = os.path.join(RUN, "verify_interrupted.ckpt.pt")
chk("the simulated disconnect happened and left a checkpoint on disk", interrupted and os.path.exists(ck))
import torch as _t
chk("the checkpoint is from before the cut, so some steps are redone",
    _t.load(ck, weights_only=False)["state"]["step"] < cut)
res = tr("verify_interrupted", **kw)
chk("the second call resumed rather than restarting", res["resumes"] == 1)
chk("resumed final loss is bit-identical to the uninterrupted run",
    res["final_val"] == ref["final_val"], f"{res['final_val']!r} vs {ref['final_val']!r}")
chk("every logged training loss is bit-identical",
    [c[1] for c in res["curve"]] == [c[1] for c in ref["curve"]], f"{len(ref['curve'])} points")
chk("every validation point is bit-identical", res["vcurve"] == ref["vcurve"])
chk("the checkpoint is removed once the result file exists", not os.path.exists(ck))

# --- regression guards for the three defects the first GPU runs exposed ---------------------
# (CUDA-only paths, so they are checked in the source; the offline run cannot reach them)
src_all = "\n".join("".join(c["source"]) for c in cells)
chk("GPU run defect 1: the fused head follows the caller's autocast state",
    "torch.is_autocast_enabled()" in src_all and "cd = AMP_DTYPE if (USE_AMP and h.is_cuda)" not in src_all)
chk("GPU run defect 2: the fp16 check frees its 810.6 MiB of chunk temporaries",
    re.search(r"del h, W, t, hr, Wr, hf, Wf, gh_ref, gW_ref, gh16, gW16, gW_early, lg, pr, Wc", src_all) is not None)
chk("GPU run defect 3: the memory slope uses the two largest batches measured",
    "sorted(MEM[label])[-2:]" in src_all)
chk("the 'at rest' line is compared with the 12 bytes/param it should be", "N_PARAMS * 12" in src_all)

# --- stale-literal guard on the findings cell ------------------------------------------------
LESSON = {str(k) for k in range(0, 13)} | {"34", "86", "30", "50", "10", "2"}
summary = "".join(cells[-1]["source"])
lits = set()
for m in re.finditer(r"print\(\s*(f?)(\"|')(.*?)\2", summary, re.S):
    if m.group(1): continue
    lits |= set(re.findall(r"\d[\d,]*\.?\d*", m.group(3)))
stray = sorted(t for t in lits if t.replace(",", "") not in LESSON)
chk("no run-dependent number is typed into the findings cell", not stray,
    f"stray: {stray}" if stray else "only labels and lesson constants")

shutil.rmtree(RUN, ignore_errors=True)
print(f"\n{len(cells)} cells executed in {WALL:.0f}s")
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1)
print("\nALL CHECKS PASS - every cell runs and every claim re-derives.")
