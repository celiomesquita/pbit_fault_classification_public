from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
ARMS = ["rf_F246", "lgbm_F246", "xgb_F246", "gb_F246", "lgbm_F115", "lgbm_F18", "rf_F115", "rf_F18",
        "B1_always_N01", "B2_prior_random", "B3_threshold_std", "B4_threshold_safety", "B5_always_F03"]


def main() -> None:
    a = pd.concat([pd.read_csv(DATA / "ablation" / f"preds_seed{s}.csv") for s in SEEDS])
    f = pd.concat([pd.read_csv(DATA / "families" / f"preds_seed{s}.csv") for s in SEEDS])
    d = a.merge(f[["seed", "Experiment_Index", "xgb_F246", "gb_F246"]], on=["seed", "Experiment_Index"])
    lines = ["# Acurácia agregada por braço (média das 10 sementes; ciclos reais, 54 SN)", "",
             "| braco | acuracia |", "|---|---|"]
    for arm in ARMS:
        acc = np.mean([(g["Fault_Mode"] == g[arm]).mean() for _, g in d.groupby("seed")])
        lines.append(f"| {arm} | {acc:.3f} |")
    (DATA / "paper_extra_accuracy.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
