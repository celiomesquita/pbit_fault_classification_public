"""Paper tables from verified E12 outputs, without changing the frozen analysis."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e12_diagnostic"


def main():
    assert json.loads((OUT / "verification.json").read_text())["status"] == "passed"
    completed = json.loads((OUT / "completion.json").read_text())
    for name in ("error_concentration.csv", "unit_errors.csv", "geometry_summary.csv"):
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == completed["output_sha256"][name]
    data = pd.read_csv(OUT / "error_concentration.csv")
    lines = ["% Source: data/e12_diagnostic/error_concentration.csv; means over reused seeds.",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Class & Policy & Cycles/SN & Cycle recall & Unit recall & SN $<.90$ & Top-SN misses & Exposure \\", r"\midrule"]
    for cls in ("F02", "F05"):
        for arm, label in (("B1_complete", "Complete"), ("B3_joint", "Joint selection")):
            row = data[(data.arm == arm) & (data.Fault_Mode == cls)].iloc[0]
            lines.append(f"{cls} & {label} & {row.n_cycles}/{row.n_units} & {row.cycle_weighted_recall:.3f} & "
                 f"{row.equal_unit_recall:.3f} & {row.units_below_90}/{row.n_units} & {row.top1_miss_share:.3f} & {row.top1_exposure_share:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/manuscript/tables/tab_e12_unit_summary.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    geometry = pd.read_csv(OUT / "geometry_summary.csv")
    labels = dict(raw_absolute="Raw level", raw_shape="Within-cycle shape", F18_IQR="F18 / training IQR", F246_IQR="F246 / training IQR")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Class & Representation & Normal nearer & Normal $q_{.95}$ & Nearer: missed & Nearer: detected \\", r"\midrule"]
    for cls in ("F02", "F05"):
        for rep, label in labels.items():
            row = geometry[(geometry.arm == "B1_complete") & (geometry.Fault_Mode == cls) & (geometry.representation == rep)].iloc[0]
            values = [row[c] for c in ("normal_closer_cycle_fraction", "within_normal_q95_cycle_fraction", "normal_closer_missed_fraction", "normal_closer_detected_fraction")]
            lines.append(f"{cls} & {label} & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/supplement/tables/tab_e12_geometry.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
