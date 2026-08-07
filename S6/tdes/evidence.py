"""Evidence bundle generation.

Every row is produced by inspecting artifacts the run actually wrote (ledgers,
manifests, checkpoints, reports). Nothing here is asserted by hand: each check is
a function that recomputes its verdict from the generated files.
"""
from __future__ import annotations
import json, os


class Evidence:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, requirement: str, passed: bool, evidence_ref: str, detail: dict | None = None):
        self.rows.append({"requirement": requirement, "result": "PASS" if passed else "FAIL",
                          "evidence": evidence_ref, "detail": detail or {}})
        return passed

    @property
    def all_passed(self) -> bool:
        return all(r["result"] == "PASS" for r in self.rows)

    def write(self, out_dir: str, extra: dict):
        os.makedirs(out_dir, exist_ok=True)
        bundle = {"generated_by": "tdes/run_demo.py",
                  "all_passed": self.all_passed,
                  "summary": {r["requirement"]: r["result"] for r in self.rows},
                  "checks": self.rows, **extra}
        json.dump(bundle, open(os.path.join(out_dir, "evidence.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)

        lines = ["# Evidence Bundle — V5 Training Data Execution System", "",
                 f"**Overall: {'PASS' if self.all_passed else 'FAIL'}** "
                 f"({sum(r['result']=='PASS' for r in self.rows)}/{len(self.rows)} checks passed)", "",
                 "| Requirement | Result | Evidence |", "|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r['requirement']} | **{r['result']}** | `{r['evidence']}` |")
        lines += ["", "## Key numbers", ""]
        for k, v in extra.get("headline", {}).items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## Detail", ""]
        for r in self.rows:
            if r["detail"]:
                lines.append(f"**{r['requirement']}** — " +
                             ", ".join(f"`{k}`={v}" for k, v in r["detail"].items()))
        open(os.path.join(out_dir, "evidence.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        return bundle
