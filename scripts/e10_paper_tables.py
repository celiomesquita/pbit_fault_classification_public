"""E10 paper presentation from the independently verified frozen-run summary."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/e10_v1/summary.csv"
LABELS = {
    "B0_historical": "Historical control", "B0_qpromotion": "Matched promotion",
    "B1_complete": "Complete", "B2_CMR_fixed": "Tuned complete, CMR fixed",
    "B2_CMR_optimized": "Tuned complete, CMR optimised", "B2_macro_F1_fixed": "Tuned complete, F1 fixed",
    "B2_macro_F1_optimized": "Tuned complete, F1 optimised", "B3_joint": "Joint selection",
}


def main():
    verified = json.loads((ROOT / "data/e10_verification.json").read_text())
    assert verified["status"] == "passed"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == verified["summary_sha256"]
    data = pd.read_csv(SOURCE)
    rows = data[np.isclose(data.target, .90) & np.isclose(data.budget, .40)]
    lines = ["% Source: verified data/e10_v1/summary.csv, target .90, budget .40.",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Policy & FA0 & F01 & F02 & F03 & F04 & F05 & F06 (expl.) \\", r"\midrule"]
    for arm, label in LABELS.items():
        row = rows[rows.arm == arm].iloc[0]
        values = [row.FA0] + [row[f"detect_F0{i}"] for i in range(1, 7)]
        lines.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/manuscript/tables/tab_e10_primary.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # The selected fallback does not depend on an unattainable recall target.
    for (arm, budget), group in data.groupby(["arm", "budget"]):
        assert len(group) == 3
        for metric in ["FA0", "min_recall", "detect_F02", "detect_F05", "detect_F06", "exact_F06"]:
            assert np.ptp(group[metric]) < 1e-12
    lines = ["% Same observed values at all three recall targets; all were internally infeasible.",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Policy & Budget & FA0 & Min. recall & F02 & F05 & F06 detect & F06 exact \\", r"\midrule"]
    for budget in (.30, .40, .50):
        for arm, label in LABELS.items():
            row = data[(data.arm == arm) & np.isclose(data.target, .90) & np.isclose(data.budget, budget)].iloc[0]
            lines.append(label + f" & {budget:.2f} & " + " & ".join(f"{row[m]:.3f}" for m in (
                "FA0", "min_recall", "detect_F02", "detect_F05", "detect_F06", "exact_F06")) + r" \\")
        lines.append(r"\midrule")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/supplement/tables/tab_e10_sensitivity.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
