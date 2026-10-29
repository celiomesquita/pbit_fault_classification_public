"""Regenerate closure tables from existing summaries, without changing experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["untuned", "macro_F1_fixed", "macro_F1_optimized", "CMR_fixed", "CMR_optimized",
        "minimum_cost", "metacost"]


def policy_tables():
    values = pd.read_csv(ROOT / "data/e1_external/metrics_and_contrasts.csv")
    availability = pd.read_csv(ROOT / "data/e1_external/availability.csv")
    tables = []
    for all_budgets in (False, True):
        lines = ["% R1.1: existing E1 metrics and availability; no interval-based labels.",
                 r"\begin{tabular}{" + ("llcccccccc" if all_budgets else "lccccccc") + "}",
                 r"\toprule"]
        lines.append(("Budget & " if all_budgets else "") +
                     r"Policy & \makecell{Fallback\\/50 folds} & \makecell{FA0 $>$ budget\\/10 seeds} & CMR & $\bar L$ & FA0 & $\Delta$CMR & " +
                     (r"$\Delta\bar L$ & " if all_budgets else "") + r"$\Delta$FA0 \\")
        lines.append(r"\midrule")
        for budget in ([.40, .30, .50] if all_budgets else [.40]):
            for arm in ARMS:
                r = values[(values.arm == arm) & np.isclose(values.budget, budget)].set_index("metric")
                a = availability[(availability.arm == arm) & np.isclose(availability.budget, budget)].iloc[0]
                def diff(metric):
                    if arm == "untuned":
                        return "---"
                    v = r.loc[metric]
                    result = f"{v['diff']:.3f}"
                    if all_budgets:
                        result += f" [{v.diff_lo:.3f}; {v.diff_hi:.3f}]"
                    return result.replace("-", "$-$")
                cells = ([f"{budget:.2f}"] if all_budgets else []) + [
                    arm.replace("_", r"\_"), str(a.failed_folds), str(a.seeds_FA0_above_budget),
                    *[f"{r.loc[m, 'value']:.3f}" for m in ("CMR", "Lbar", "FA0")], diff("CMR")]
                if all_budgets:
                    cells.append(diff("Lbar"))
                cells.append(diff("FA0"))
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\midrule")
        lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
        tables.append(("tab_e1_all_budgets" if all_budgets else "tab_independent_policies", lines))
    for name, lines in tables:
        (ROOT / f"paper/{'manuscript' if name == 'tab_independent_policies' else 'supplement'}/tables/{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def e8_table():
    summary = (ROOT / "data/e8_smotewb_summary.md").read_text(encoding="utf-8")
    wanted = ["lgbm_freq_sev_gaussian", "li_fixed", "li_tuned_grouped", "li_fixed_random", "li_tuned_random"]
    rows = {}
    for line in summary.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in wanted:
            rows[cells[0]] = cells
    lines = ["% Source: data/e8_smotewb_summary.md; protocol docs/protocolos/E8.md.",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Pipeline & Partition & CMR & FA0 & Macro-F1 & Micro-F1 \\", r"\midrule"]
    for arm in wanted:
        c = rows[arm]
        partition = "Random" if arm.endswith("random") else "Grouped by SN"
        label = "Gaussian reference" if "gaussian" in arm else ("Li fixed" if "fixed" in arm else "Li retuned")
        lines.append(" & ".join([label, partition, c[1], c[3], c[6], c[8]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/supplement/tables/tab_e8_reproduction.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    policy_tables()
    e8_table()
