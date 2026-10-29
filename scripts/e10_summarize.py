"""Descriptive E10 summaries; never select using outer labels."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.triage_policy import CLASSES, evaluate

OUT = ROOT / "data/e10_v1"


def summarize():
    completion = json.loads((OUT / "completion.json").read_text())
    assert len(completion["completed"]) == 50
    folders = [OUT / f"seed_{r['seed']}_outer_{r['outer']}" for r in completion["completed"]]
    for folder in folders:
        manifest = json.loads((folder / "complete.json").read_text())
        for name in ("external_predictions.csv", "fold_metrics.csv", "unit_metrics.csv", "inner_selections.json"):
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == manifest["output_sha256"][name]
    predictions = pd.concat([pd.read_csv(f / "external_predictions.csv") for f in folders], ignore_index=True)
    folds = pd.concat([pd.read_csv(f / "fold_metrics.csv") for f in folders], ignore_index=True)
    unit_metrics = pd.concat([pd.read_csv(f / "unit_metrics.csv") for f in folders], ignore_index=True)
    seed_rows = []
    for (seed, arm, target, budget), data in predictions.groupby(["seed", "arm", "target", "budget"]):
        assert len(data) == 1195 and data.Experiment_Index.is_unique
        assert data.Serial_Number.nunique() == 54
        assert data.groupby("Serial_Number").outer.nunique().eq(1).all()
        metrics = evaluate(data.Fault_Mode.to_numpy(), data.prediction.to_numpy())
        critical_units = unit_metrics[(unit_metrics.seed == seed) & (unit_metrics.arm == arm) &
                          np.isclose(unit_metrics.target, target) & np.isclose(unit_metrics.budget, budget) &
                          (unit_metrics.critical_n > 0)]
        seed_rows.append(dict(seed=seed, arm=arm, target=target, budget=budget,
                         Lbar=critical_units.CMR.mean(),
                         external_meets=bool(metrics["min_recall"] >= target-1e-12 and metrics["FA0"] <= budget+1e-12),
                         external_FA0_exceeds=bool(metrics["FA0"] > budget+1e-12), **metrics))
    by_seed = pd.DataFrame(seed_rows)
    by_seed.to_csv(OUT / "metrics_by_seed.csv", index=False)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    unit_metrics.to_csv(OUT / "unit_metrics.csv", index=False)
    columns = ["FA0", "FA0_N01", "FA0_N02", "CMR", "R3", "Lbar", "macro_F1", "min_recall", "mean_recall"] + [
        f"{kind}_{cls}" for cls in CLASSES[2:] for kind in ("detect", "exact")]
    rows = []
    for (arm, target, budget), data in by_seed.groupby(["arm", "target", "budget"]):
        assert len(data) == 10
        inner = folds[(folds.arm == arm) & np.isclose(folds.target, target) & np.isclose(folds.budget, budget)]
        assert len(inner) == 50
        rows.append(dict(arm=arm, target=target, budget=budget, inner_feasible_folds=int(inner.inner_feasible.sum()),
                     external_meets_seeds=int(data.external_meets.sum()),
                     external_FA0_exceeds_seeds=int(data.external_FA0_exceeds.sum()),
                     **{c: data[c].mean() for c in columns}))
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUT / "summary.csv", index=False)
    contrasts = []
    pairs = [("B1_complete", "B0_qpromotion")] + [
        (arm, "B1_complete") for arm in aggregate.arm.unique() if arm.startswith("B2_") or arm == "B3_joint"]
    for a, b in pairs:
        for target in (.85, .90, .95):
            for budget in (.30, .40, .50):
                aa = by_seed[(by_seed.arm == a) & np.isclose(by_seed.target, target) & np.isclose(by_seed.budget, budget)].set_index("seed")
                bb = by_seed[(by_seed.arm == b) & np.isclose(by_seed.target, target) & np.isclose(by_seed.budget, budget)].set_index("seed")
                difference = aa[columns] - bb[columns]
                contrasts.append(dict(A=a, B=b, target=target, budget=budget,
                                 **{c: difference[c].mean() for c in columns}))
    pd.DataFrame(contrasts).to_csv(OUT / "paired_differences.csv", index=False)
    lines = ["# E10 — Comparação descritiva de políticas de triagem", "",
             "F01–F05: detecção primária; F06: exploratória. Normalidade: N01/N02. "
             "Resultados não demonstram aceitação operacional ou generalização externa.", "",
             "Métricas externas agrupadas dentro de cada semente e depois médias sobre dez sementes. "
             "Mínimo recall é a média dos mínimos por semente, não garantia por classe. "
             "Todos os folds e fallbacks são preservados; sementes reutilizam as mesmas unidades.", ""]
    for target, budget in [(.90, .40)] + [(r, b) for r in (.85, .90, .95) for b in (.30, .40, .50) if (r, b) != (.90, .40)]:
        lines += [f"## Recall mínimo {target:.0%}; orçamento FA0 {budget:.2f}", "",
                  "| Braço | Admissível interno /50 | Critério externo /10 | FA0 | Mín. recall | F01 | F02 | F03 | F04 | F05 | F06 (expl.) | CMR |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in aggregate[np.isclose(aggregate.target, target) & np.isclose(aggregate.budget, budget)].itertuples():
            values = [row.arm, str(row.inner_feasible_folds), str(row.external_meets_seeds)] + [
                f"{getattr(row, c):.3f}" for c in ["FA0", "min_recall"] + [f"detect_{x}" for x in CLASSES[2:]] + ["CMR"]]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines += ["B0_historical preserva a seleção histórica e não isola a nova regra de admissibilidade. "
              "B0_qpromotion e B1_complete compartilham score e critério interno; diferem em permitir retirar alarmes. "
              "B2 tem quatro sub-braços, sem escolher o melhor com dados externos. "
              "B3 usa seleção conjunta interna sob orçamento fixo.", "",
              "Detalhes: summary.csv, metrics_by_seed.csv, fold_metrics.csv, unit_metrics.csv, "
              "paired_differences.csv e os scores/configurações por partição. "
              "Nenhum intervalo ou teste confirmatório é derivado desta reutilização de unidades.", ""]
    (ROOT / "data/e10_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"E10 summary written: {len(by_seed)} seed/scenario/arm records.", flush=True)


if __name__ == "__main__":
    summarize()
