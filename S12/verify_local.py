#!/usr/bin/env python3
"""Execute the notebook's OWN code cells and re-derive every claim independently.

Same discipline as S9/S10/S11: this parses S12_zero.ipynb and runs the actual cells,
so a notebook edited since it last ran cannot pass. It then recomputes each claim from
scratch rather than reading the notebook's variables where that is possible.

Step counts are reduced by default so it finishes quickly on CPU; the byte accounting,
the collectives and every closed-form claim are exact and independent of that.
"""
import json, os, sys, re, math, time

HERE = os.path.dirname(os.path.abspath(__file__))
NB   = os.path.join(HERE, "S12_zero.ipynb")
E    = os.environ
for k, v in [("S12_STEPS", "8"), ("S12_WORLD", "32"), ("S12_MICRO", "2"),
             ("S12_GLOO_W", "4"), ("S12_THREADS", "4")]:
    E.setdefault(k, v)

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
WALL = time.time() - t0

print(f"\n{'='*72}\nINDEPENDENT CHECKS\n{'='*72}")
fails = []
def chk(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond: fails.append(name)

import torch
W    = int(E["S12_WORLD"])
GiB  = 2 ** 30
runs, curves = ns["runs"], ns["curves"]
NPAD, NP, P_SIM = ns["NPAD"], ns["NP"], ns["P_SIM"]

# --- the arithmetic of section 0 ----------------------------------------------------
chk("16 bytes per weight is 2+2+4+4+4", sum(ns["BYTES"].values()) == 16)
chk("30B x 16 bytes = 447.0 GiB", abs(30e9 * 16 / GiB - 447.0) < 0.05,
    f"{30e9*16/GiB:.2f} GiB")
chk("P for a 30B bf16 model is 60 GB", abs(ns["P_BYTES"] - 60e9) < 1e6)
chk("an 80 GB card is 74.5 GiB", abs(ns["CARD"] - 74.5) < 0.05, f"{ns['CARD']:.3f}")

# --- collectives: recomputed here, not read from the notebook -----------------------
Cluster = ns["Cluster"]
torch.manual_seed(7)
xs = [torch.randn(1024) for _ in range(W)]
truth = torch.stack(xs).mean(0)
cl = Cluster(W)
ar = cl.all_reduce_mean([x.clone() for x in xs])
chk("all_reduce_mean returns the true mean on every rank",
    max((a - truth).abs().max().item() for a in ar) < 1e-6)
rs = Cluster(W).reduce_scatter_mean([x.clone() for x in xs])
chk("reduce_scatter gives rank k the k-th slice of that mean",
    all((rs[k] - torch.chunk(truth, W)[k]).abs().max().item() < 1e-6 for k in range(W)))
ag = Cluster(W).all_gather([s.clone() for s in rs])
chk("reduce-scatter then all-gather IS all-reduce (the section-4 identity)",
    (ag - truth).abs().max().item() < 1e-6, f"{(ag-truth).abs().max().item():.2e}")
ring, per_rank = Cluster(W).ring_all_reduce_mean([x.clone() for x in xs], count=False)
chk("the ring implementation agrees with the direct mean",
    max((r - truth).abs().max().item() for r in ring) < 1e-5)
chk("ring cost is exactly 2(W-1)/W per rank",
    abs(per_rank / (1024 * 4) - 2 * (W - 1) / W) < 1e-9,
    f"{per_rank/(1024*4):.4f} vs {2*(W-1)/W:.4f}")
chk("gloo cross-check ran and agreed", ns["GLOO_OK"] is True,
    ns["gloo_note"] or "all four collectives identical")

# --- memory: re-derive bytes/weight from first principles ---------------------------
def bpw(stage, w):
    return ((2 if stage in ("single","dp","zero1","zero2") else 2/w)
          + (2 if stage in ("single","dp","zero1") else 2/w)
          + (12 if stage in ("single","dp") else 12/w))
for stage in ("dp", "zero1", "zero2", "zero3"):
    chk(f"{stage}: measured bytes/weight matches first principles at W={W}",
        abs(runs[stage]["bpw"] - bpw(stage, W)) < 1e-9,
        f"{runs[stage]['bpw']:.4f}")
for stage, want in (("dp",16.00), ("zero1",5.50), ("zero2",3.75), ("zero3",2.00)):
    chk(f"{stage}: the lesson's W=8 figure of {want}", abs(bpw(stage, 8) - want) < 5e-3)
chk("ZeRO-3 shards rather than compresses: bpw x W = 16",
    abs(runs["zero3"]["bpw"] * W - 16) < 1e-6, f"{runs['zero3']['bpw']*W:.3f}")

# --- communication ------------------------------------------------------------------
for stage, mult in (("dp",2), ("zero1",2), ("zero2",2), ("zero3",3)):
    got = runs[stage]["wire"] / P_SIM
    chk(f"{stage}: wire is {mult}P, exactly {mult}(W-1)/W",
        abs(got - mult * (W - 1) / W) < 1e-6, f"{got:.4f}P")
chk("ZeRO-1 and ZeRO-2 cost data parallelism nothing extra",
    abs(runs["zero1"]["wire"] - runs["dp"]["wire"]) < 1
    and abs(runs["zero2"]["wire"] - runs["dp"]["wire"]) < 1)
chk("ZeRO-3 costs exactly half as much again",
    abs(runs["zero3"]["wire"] / runs["dp"]["wire"] - 1.5) < 1e-6)

# --- equivalence --------------------------------------------------------------------
for stage in ("zero1", "zero2", "zero3"):
    chk(f"{stage} reproduces data parallelism exactly",
        max(abs(a-b) for a, b in zip(curves[stage], curves["dp"])) == 0.0)
gap = max(abs(a-b) for a, b in zip(curves["dp"], curves["single"]))
chk("vs single-GPU it is close but NOT exact (and the notebook says so)",
    0 < gap < 1e-3, f"{gap:.2e}")

# --- compute ------------------------------------------------------------------------
fl = {s: runs[s]["flops"] for s in ("dp","zero1","zero2","zero3")}
chk("every arrangement does identical arithmetic", len(set(fl.values())) == 1,
    f"{list(fl.values())[0]:,} FLOPs")

# --- the closed-form claims of the beyond sections -----------------------------------
chk("the 4-byte floor is world-size invariant",
    all(abs((bpw("zero1", w) - 12/w) - 4.0) < 1e-12 for w in (8, 64, 512, 10**6))
    and all(abs(bpw("dp", w) - 16.0) < 1e-12 for w in (8, 64, 512, 10**6)),
    "weights+gradients stay 4 B/w for every W")
chk("4 bytes/weight fills a 74.5 GiB card at 20.0B params",
    abs(ns["CARD"] * GiB / 4 / 1e9 - 20.0) < 0.05, f"{ns['CARD']*GiB/4/1e9:.2f}B")
chk("DP and ZeRO-1 never fit at any world size",
    all(bpw(s, w) * 30e9 / GiB > 74.5 for s in ("dp","zero1") for w in (8,64,4096)))
for card, peak, t in (("H100", 989.4e12, 7.10), ("B200", 2.25e15, 3.12)):
    mfu = (6 * 30e9 * 1e6) / (64 * peak) / t
    chk(f"the lesson's {card} step time encodes MFU = 40%", abs(mfu - 0.40) < 0.005,
        f"{100*mfu:.1f}%")
chk("2P over InfiniBand is 2.40 s", abs(2 * 60e9 / 50e9 - 2.40) < 1e-9)
chk("2P over NVLink is 0.27 s", abs(2 * 60e9 / 450e9 - 0.2667) < 1e-3)
sim = ns["simulate"]
chk("the 83% overlap figure is the (N-1)/N ceiling at 6 buckets",
    abs(sim(2, 7.10, 0.05)[1] - 5/6) < 1e-9, f"{100*sim(2,7.10,0.05)[1]:.2f}% vs 83.33%")
chk("H100 prefers the smallest bucket, B200 prefers two",
    min(range(1,13), key=lambda k: sim(k, 7.10, 0.05)[0]) == 1
    and min(range(1,13), key=lambda k: sim(k, 3.12, 0.05)[0]) == 2)
fp8 = 2 * (1 + 1/32) + 12
chk("MXFP8 is 14.0625 bytes per weight", abs(fp8 - 14.0625) < 1e-9)
chk("MXFP8 saves 12.1% of the stored state", abs(100*(16-fp8)/16 - 12.1) < 0.05,
    f"{100*(16-fp8)/16:.2f}%")
chk("MXFP8 halves the wire, which is the larger effect",
    abs((2*30e9/50e9) / (2*60e9/50e9) - 0.5) < 1e-12)

# --- regression guard: the summary cell must not retype a measured number ---------
# An earlier build hardcoded a factor in the recap while the cell above measured a
# different one. Any run-dependent figure in the recap must be interpolated, not typed.
chk("the measured bf16 factor is exposed, not retyped", "BF16_FACTOR" in ns,
    f"{ns.get('BF16_FACTOR', float('nan')):,.0f}x")
indep = ((ns["final_w"]("dp") - ns["final_w"]("single")).abs().max().item()
         / (ns["fp32_w"]["dp"] - ns["fp32_w"]["single"]).abs().max().item())
chk("and it re-derives independently", abs(indep / ns["BF16_FACTOR"] - 1) < 1e-9,
    f"{indep:,.0f}x")

# every number left in a non-f-string print() of the recap must be a lesson constant
# lesson constants (inputs being checked against) and structural claims that the
# checks above already assert independently - everything else must be interpolated
LESSON_CONSTS = {"0","1","2","3","4","5","6","7","8","9","10","11","12","16","20",
                 "32","40","50","78","83","12.1","16.00","5.50","3.75","2.00",
                 "7.10","3.12",          # the lesson's two step times
                 "0.00","00"}            # from "0.00e+00", asserted exactly above
summary = "".join(cells[-1]["source"])
PRINT_STR = re.compile(r"print\(\s*(f?)([\"'])(.*?)\2", re.S)
literals = set()
for m in PRINT_STR.finditer(summary):
    if m.group(1):
        continue                                  # f-strings interpolate; fine
    literals |= set(re.findall(r"\d[\d,]*\.?\d*", m.group(3)))
stale = sorted(t for t in literals if t.replace(",", "") not in LESSON_CONSTS)
chk("no un-interpolated run-dependent number survives in the recap", not stale,
    f"stray literals: {stale}" if stale else "only lesson constants")

print(f"\n{len(cells)} cells executed in {WALL:.0f}s")
if fails:
    print(f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1)
print("\nALL CHECKS PASS - every cell runs and every claim re-derives.")
