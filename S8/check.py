#!/usr/bin/env python3
"""Verifies the chronology. Exits non-zero if anything the assignment grades is wrong."""
import re, sys
d = open("data.js", encoding="utf-8").read()
blocks = d.split("{\n  id:")[1:]
entries = []
for b in blocks:
    g = lambda f: (re.search(rf'\b{f}:\s*"((?:[^"\\]|\\.)*)"', b) or [None, None])[1]
    entries.append({"id": re.search(r'^\s*"([^"]+)"', b).group(1), "name": g("name"),
                    "date": g("date"), "url": g("url"), "note": g("dateNote"),
                    "raw": b})
fail = []
# 1. chronological
dates = [e["date"] for e in entries]
if dates != sorted(dates): fail.append("NOT in chronological order")
# 2. required coverage (assignment's minimum list)
REQUIRED = ["Scaled dot-product", "Absolute learned", "Sinusoidal", "RoPE", "ALiBi", "Multi-Query",
            "Grouped-Query", "Sliding-window", "Attention sinks", "NTK-aware", "YaRN",
            "Linear attention", "Delta rule", "Gated DeltaNet", "Latent Attention", "Top-k",
            "DeepSeek Sparse", "DroPE"]
names = " | ".join(e["name"] for e in entries)
for r in REQUIRED:
    if r not in names: fail.append(f"MISSING required mechanism: {r}")
# 3. every entry has honest trade-offs + a date source
for e in entries:
    for f in ("bill", "how", "buys", "costs", "pick", "dateNote", "url", "paper"):
        if f + ":" not in e["raw"]: fail.append(f"{e['id']}: missing {f}")
    if not re.match(r"^https?://", e["url"] or ""): fail.append(f"{e['id']}: bad source URL")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", e["date"] or ""): fail.append(f"{e['id']}: bad date")
# 4. arXiv self-check: if the source is an arXiv ID, its YYMM must match the stated date
for e in entries:
    m = re.search(r"arxiv\.org/abs/(\d{2})(\d{2})\.", e["url"] or "")
    if not m: continue
    yy, mm = "20" + m.group(1), m.group(2)
    dy, dm = e["date"][:4], e["date"][5:7]
    # allow a 1-month lag: v1 submitted end of month, listed the next
    agrees = (yy == dy and abs(int(mm) - int(dm)) <= 1)
    if not agrees:
        # Legitimate when the thing SHIPPED before it was written up - but then the
        # entry must say so explicitly. Silence is the error, not the mismatch.
        explained = re.search(r"release|Reddit|community|not a paper", e["note"] or "", re.I)
        if not explained:
            fail.append(f"{e['id']}: arXiv ID {m.group(1)}{m.group(2)} disagrees with date "
                        f"{e['date']} and dateNote does not explain why")
        else:
            print(f"  note: {e['id']} dated {e['date']} by release, paper is "
                  f"{m.group(1)}{m.group(2)} - explained in dateNote (OK)")
print(f"entries: {len(entries)}  span: {dates[0]} -> {dates[-1]}")
print(f"required mechanisms covered: {len(REQUIRED)}/{len(REQUIRED)}" if not any('MISSING' in f for f in fail) else "")
bonus_n = len(re.findall(r"bonus:\s*true", d))
print(f"bonus entries: {bonus_n}")
if fail:
    print("\nFAILURES:"); [print("  -", f) for f in fail]; sys.exit(1)
print("\nALL CHECKS PASS: chronological, fully covered, every date sourced, arXiv IDs agree with dates.")

def check_deployable(folder):
    """Every src/href the page requests must exist in the folder being published.

    This exists because the first Netlify deploy was a single-file drop: index.html
    went up, data.js did not, and the page fell straight into its own "chronology
    could not load" fallback. Validating data.js locally proved nothing about what
    was actually served."""
    import os, re
    idx = os.path.join(folder, "index.html")
    assert os.path.exists(idx), f"no index.html in {folder}"
    html = open(idx, encoding="utf-8").read()
    refs = re.findall(r'<(?:script|link|img)[^>]+(?:src|href)="([^"]+)"', html)
    local = [r for r in refs if not r.startswith(("http://", "https://", "//", "#", "data:", "mailto:"))]
    missing = [r for r in local if not os.path.exists(os.path.join(folder, r.split("?")[0]))]
    print(f"\n  publish folder: {folder}")
    print(f"  files present : {sorted(os.listdir(folder))}")
    print(f"  local assets index.html requests: {local or 'none'}")
    if missing:
        raise SystemExit(f"  MISSING FROM PUBLISH FOLDER: {missing}\n"
                         f"  -> deploying this folder would 404 on {missing[0]}")
    print("  all referenced assets present")
    return local


if __name__ == "__main__" and "--dist" in __import__("sys").argv:
    import sys
    folder = sys.argv[sys.argv.index("--dist") + 1]
    check_deployable(folder)
