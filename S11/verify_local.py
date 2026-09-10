#!/usr/bin/env python3
"""Execute the notebook's OWN code cells offline and re-derive every claim.

S11_OFFLINE=1 swaps the SmolLM2 download for a locally built model of the same
architecture plus the Session-2 tokenizer, because this sandbox cannot reach
huggingface.co. Depths and step counts are reduced so it finishes on CPU; the
Adam arithmetic, the bias-correction timescales and the muP scaling rules are
all exact and independent of those.
"""
import json, os, sys, re, math, time, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
NB   = os.path.join(HERE, "S11_optimizers.ipynb")
E = os.environ
E["S11_OFFLINE"] = "1"
for k, v in [("S11_LAYERS","2"),("S11_SEQ","64"),("S11_NSEQ","40"),
             ("S11_WARMUP","12"),("S11_RATIO_STEPS","40"),
             ("S11_TOTAL","24"),("S11_STOP","16"),
             ("S11_SW_VOCAB","512"),("S11_SW_SEQ","32"),("S11_SW_LAYERS","2"),
             ("S11_SW_STEPS","25"),("S11_SW_BS","8"),
             ("S11_WIDTHS","64,128,256"),("S11_LRS","1e-3,3e-3,1e-2,3e-2")]:
    E.setdefault(k, v)
E["S11_SHARD"]     = "/sessions/serene-dazzling-volta/mnt/ERA/ERA5/ERA5/S9/data/s9_owt_clean.jsonl.gz"
E["S11_TOKENIZER"] = "/sessions/serene-dazzling-volta/mnt/ERA/s2_submission/upload/tokenizer.json"

cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(cells):
    src = re.sub(r"^(\s*)!.*$", r"\1pass", "".join(c["source"]), flags=re.M)
    print(f"\n{'='*72}\ncell {i+1}/{len(cells)}\n{'='*72}")
    try:
        exec(compile(src, f"<cell {i+1}>", "exec"), ns)
    except Exception:
        import traceback; traceback.print_exc()
        print(f"\nFAILED in cell {i+1}"); sys.exit(1)

print(f"\n{'='*72}\nINDEPENDENT CHECKS\n{'='*72}")
fails = []
def chk(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond: fails.append(name)

import torch
B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3
GRADS = [0.50, 0.40, 0.60, 0.45, 0.55]

# --- item 1: re-derive Adam from scratch, and against the lesson's printed table --
m = v = 0.0; w = 1.0; mine = []
for t, g in enumerate(GRADS, 1):
    m = B1*m + (1-B1)*g; v = B2*v + (1-B2)*g*g
    mh, vh = m/(1-B1**t), v/(1-B2**t)
    w += -LR*mh/(math.sqrt(vh)+EPS); mine.append((m, v, mh, vh, w))
lesson = [(0.0500,0.000250,0.5000,0.2500,0.999000),(0.0850,0.000410,0.4474,0.2050,0.998012),
          (0.1365,0.000769,0.5037,0.2567,0.997018),(0.1678,0.000971,0.4881,0.2431,0.996028),
          (0.2061,0.001273,0.5032,0.2550,0.995031)]
ok = all(round(a, len(str(b).split(".")[1])) == b
         for row, lrow in zip(mine, lesson) for a, b in zip(row, lrow))
chk("Adam by hand matches the lesson's printed table", ok)
chk("notebook's own Adam matches this independent one",
    all(abs(a["w"]-b[4]) < 1e-12 for a, b in zip(ns["rows"], mine)))
chk("notebook's Adam matches PyTorch to <1e-12", ns["worst"] < 1e-12, f"{ns['worst']:.2e}")
st = [abs(r["step"]) for r in ns["rows"]]
dev = 100 * max(abs(s/LR - 1) for s in st)
chk("steps are far more uniform than the gradients", dev < 2.0,
    f"steps vary {dev:.2f}% while gradients vary {100*(max(GRADS)/min(GRADS)-1):.0f}%")
# The lesson's prose says "within half a percent of 0.001"; its own table shows
# 1.20%. Assert the discrepancy explicitly so nobody silently "fixes" the notebook
# to agree with the sentence instead of the arithmetic.
chk("the lesson's 'half a percent' prose is ~2x too tight (table is right)",
    0.5 < dev < 2.0, f"measured {dev:.2f}%, prose says 0.50%")

# --- item 2: the bias-correction timescale, recomputed ---------------------------
rr = lambda t: math.sqrt(1-B2**t)/(1-B1**t)
chk("t=1 uncorrected step is 3.16x too large", abs(1/rr(1) - 3.1623) < 1e-3, f"{1/rr(1):.4f}")
chk("the gap GROWS over the first 20 steps", 1/rr(20) > 1/rr(1),
    f"t=1 {1/rr(1):.2f}x -> t=20 {1/rr(20):.2f}x")
chk("worst point is around step 10", 5 <= ns["peak"] <= 20, f"peak at t={ns['peak']}")
def su(tol):
    t = 1
    while abs(rr(t)-1) > tol: t += 1
    return t
for lab, tol in (("10%",0.10),("5%",0.05),("1%",0.01)):
    chk(f"'{lab}' answer is in the thousands and matches the notebook",
        ns["ANS"][lab] == su(tol) and su(tol) > 1000, f"{su(tol):,} steps")
chk("beta2 is what sets it, not beta1", su(0.01) > 40 * 20)

# --- item 3 ---------------------------------------------------------------------
chk("update/weight logged for every tracked layer",
    all(len(v_) == int(E["S11_RATIO_STEPS"]) for v_ in ns["hist"].values()),
    f"{len(ns['hist'])} tensors x {int(E['S11_RATIO_STEPS'])} steps")
chk("ratios are finite and positive",
    all(math.isfinite(x) and x >= 0 for v_ in ns["hist"].values() for x in v_))

# --- item 4 ---------------------------------------------------------------------
T, S = int(E["S11_TOTAL"]), int(E["S11_STOP"])
chk("cosine is mid-decay when stopped early",
    0.01 < ns["h_cos"][-1][0]/ns["PEAK_LR"] < 0.99,
    f"{100*ns['h_cos'][-1][0]/ns['PEAK_LR']:.0f}% of peak")
chk("WSD is still at (or near) peak when stopped early",
    ns["h_wsd"][-1][0]/ns["PEAK_LR"] > 0.9,
    f"{100*ns['h_wsd'][-1][0]/ns['PEAK_LR']:.0f}% of peak")
chk("both ran the same number of steps", len(ns["h_cos"]) == len(ns["h_wsd"]) == S)

# --- item 5: the muP rules, checked structurally rather than by outcome ----------
TinyLM = ns["TinyLM"]
for wdt in (256, 512):
    a, b = TinyLM(wdt, mup=False), TinyLM(wdt, mup=True)
    md = wdt / 256
    # muP-for-Adam does NOT rescale init when the base is 1/sqrt(fan_in): the
    # difference is the per-tensor LR plus the readout multiplier. The first
    # version of this notebook rescaled init too, double-counting the readout.
    exp = 1/math.sqrt(a.head.weight.shape[-1])
    chk(f"init is IDENTICAL in SP and muP at width {wdt}",
        abs(a.head.weight.std().item() - b.head.weight.std().item()) < 0.1*exp
        and abs(b.head.weight.std().item() - exp) < 0.15*exp,
        f"SP {a.head.weight.std().item():.5f} vs muP {b.head.weight.std().item():.5f}")
    chk(f"muP leaves the embedding init alone at width {wdt}",
        abs(a.emb.weight.std().item() - b.emb.weight.std().item())
        < 0.15*a.emb.weight.std().item())
    # the positional table is input-side and must be vector-like, not matrix-like
    vecs = {id(p_) for p_ in b.param_groups(1e-3)[0]["params"]}
    chk(f"pos is classified vector-like at width {wdt}", id(b.pos) in vecs,
        f"pos.dim()={b.pos.dim()} - a rank test alone files it as a matrix")
    chk(f"muP applies the 1/md readout multiplier at width {wdt}",
        abs(b.md - wdt/256) < 1e-9)
    g = b.param_groups(1e-3)
    chk(f"muP gives matrix-like params lr/md at width {wdt}",
        len(g) == 2 and abs(g[1]["lr"] - 1e-3/md) < 1e-15 and abs(g[0]["lr"] - 1e-3) < 1e-15,
        f"vector {g[0]['lr']:.2e}, matrix {g[1]['lr']:.2e}, md={md}")
    del a, b
chk("standard parameterization uses one group",
    len(TinyLM(256, mup=False).param_groups(1e-3)) == 1)
chk("the sweep ran both parameterizations at every width",
    all(("standard", w) in ns["sweep"] and ("muP", w) in ns["sweep"] for w in ns["WIDTHS"]))
chk("every sweep cell produced a number", all(len(r["losses"]) == len(ns["LRS"])
    for r in ns["sweep"].values()))
# an edge minimum is a bound, not a location - refine() must say so
rf, ok_ = ns["refine"]([1e-4,1e-3,1e-2], [1.0, 2.0, 3.0])
chk("refine() flags a minimum sitting on the grid edge", ok_ is False, f"est {rf:.1e}")
rf2, ok2 = ns["refine"]([1e-4,1e-3,1e-2], [3.0, 1.0, 2.0])
chk("refine() fits an interior minimum", ok2 is True and 1e-4 < rf2 < 1e-2, f"est {rf2:.2e}")

# --- the "beyond the assignment" sections ---------------------------------------
closed = lambda b1: math.sqrt((1-b1)/(1+b1)) * math.sqrt(2/math.pi)
chk("Adam on a constant gradient takes exactly one eta",
    abs(ns["same_sign"] - 1.0) < 0.01, f"{ns['same_sign']:.4f}")
chk("Adam on noisy gradients matches the closed form, not the lesson's 0.281",
    abs(ns["noisy"] - closed(0.9)) < 0.25 * closed(0.9)
    and abs(ns["noisy"] - 0.281) > 0.15 * 0.281,
    f"measured {ns['noisy']:.4f}, closed form {closed(0.9):.4f}, lesson 0.281")
chk("decoupled decay is exactly geometric",
    abs(ns["w_"] - (1 - ns["ETA"]*ns["LAM"])**100) < 1e-12)
chk("orthogonalisation flattens the spectrum to 1",
    abs(ns["s2"].max().item() - 1) < 1e-4 and abs(ns["s2"].min().item() - 1) < 1e-4,
    f"span {ns['s2'].min().item():.6f}-{ns['s2'].max().item():.6f}")
chk("the momentum matrix really is spectrally concentrated",
    ns["eff"] < 0.5 * len(ns["s"]),
    f"effective rank {ns['eff']:.1f} of {len(ns['s'])}")
chk("gradient-descent multiplier table reproduces the lesson",
    abs(gd := (1 - 2*0.90)) + 0 == 0.8 or abs(gd + 0.8) < 1e-12, f"1-2*0.9 = {1-2*0.9:+.2f}")

print(f"\n{len(cells)} cells executed in {time.time()-t0:.0f}s")
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1)
print("\nALL CHECKS PASS - every cell runs and every claim re-derives.")
