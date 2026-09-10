md("## Summary — the five answers")

code(r"""
print("=" * 78)
print(f"1  Adam by hand      5 steps, worst |hand - PyTorch| = {worst:.1e} (float64)")
print(f"                     gradients ranged {max(GRADS)/min(GRADS):.2f}x, steps ranged "
      f"{max(steps)/min(steps):.4f}x -> eta sets the distance")
print(f"2  bias correction   t=1 {1/ratio(1):.2f}x too large, WORST at t={peak} ({1/ratio(peak):.2f}x), "
      f"t=20 still {1/ratio(20):.2f}x")
print(f"                     stops mattering at ~{ANS['10%']:,} / {ANS['5%']:,} / {ANS['1%']:,} "
      f"steps (10% / 5% / 1%) - beta2 sets this")
print(f"3  update/weight      warmup ends {WARMUP}; ratio plateaus "
      f"{sorted(v for v in plats.values() if v is not None)}")
print(f"                     final ratios " +
      ", ".join(f"{k.split('.')[0]}={hist[k][-1]:.1e}" for k in list(watch)[:3]))
print(f"4  cosine vs WSD      stopped at {STOP}/{TOTAL}: cosine {lc:.4f} (lr "
      f"{100*h_cos[-1][0]/PEAK_LR:.0f}% of peak), WSD {lw_:.4f} (lr "
      f"{100*h_wsd[-1][0]/PEAK_LR:.0f}%)")
print(f"                     keep WSD - the cosine checkpoint is an unfinished {TOTAL}-step model")
print(f"5  lr sweep           standard minima ratio per doubling "
      f"{math.exp(statistics.mean(math.log(r) for r in std_r)):.2f}x (expect 0.50x)")
print(f"                     muP minima ratio per doubling "
      f"{math.exp(statistics.mean(math.log(r) for r in mup_r)):.2f}x (expect 1.00x)")
print(f"                     width 4096: muP transfer {mup_pred:.2e}; "
      f"standard 1/width extrapolation {std_pred:.2e}")
print("=" * 78)
""")
nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.10"},
      "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "S11_optimizers.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote %s  (%d cells: %d code, %d markdown)" % (
    out, len(C), sum(1 for c in C if c["cell_type"]=="code"),
    sum(1 for c in C if c["cell_type"]=="markdown")))

