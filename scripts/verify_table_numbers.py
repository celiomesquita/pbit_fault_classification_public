from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIRS = (ROOT / "paper" / "manuscript" / "tables", ROOT / "paper" / "supplement" / "tables")
DATA = ROOT / "data"
MAP = {
    "tab_classifier_comparison": ["families_summary.md", "paper_extra_accuracy.md"],
    "tab_ablation": ["ablation_summary.md", "paper_extra_accuracy.md"],
    "tab_hpo_nested": ["nested_cv_summary.md"],
    "tab_test_set_results": ["final_test_reanalysis.md", "final_test_uncertainty.md", "final_safety_eval_results.csv"],
    "tab_operating_point": ["risk_control_summary.md"],
    "tab_risk_control": ["risk_control_summary.md"],
    "tab_loso_f06": ["loso_f06_summary.md"],
    "tab_stratification": ["leakage_summary.md"],
    "tab_leakage_comparison": ["leakage_summary.md"],
    "tab_imbalance": ["imbalance_summary.md"],
    "tab_cost_sensitive": ["cost_sensitive_summary.md"],
    "tab_margin_sensitivity": ["margin_sensitivity_summary.md"],
    "tab_hpsens": ["hpsens_summary.md"],
}
NUM = re.compile(r"[-+]?\d+\.\d+")


def main() -> int:
    bad = 0
    for name, files in MAP.items():
        text = "\n".join((DATA / f).read_text(encoding="utf-8") for f in files)
        pool = [float(x) for x in NUM.findall(text)]
        body = "\n".join(l for l in next(d / f"{name}.tex" for d in TABLE_DIRS if (d / f"{name}.tex").exists()).read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("%"))
        body = body.replace("$-$", "-").replace("$+$", "+")
        missing = []
        for tok in NUM.findall(body):
            d = len(tok.split(".")[1])
            val = abs(float(tok))
            if not any(abs(round(abs(x), d) - val) < 1e-9 for x in pool):
                missing.append(tok)
        print(f"{name}: {'OK' if not missing else 'NOT FOUND: ' + ', '.join(missing)}")
        bad += len(missing)
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
