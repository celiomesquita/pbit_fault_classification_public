"""Retrospective inference correction from saved predictions, without model refits."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.inference import confusion_metrics, paired_comparison, holm_adjust
from pbit.modeling import FAULT_ORDER

SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
MARGINS = {"CMR": 0.02, "FA0": 0.03, "DR": 0.02, "macro_F1": 0.03}
OUT = ROOT / "data/restart/inference"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}
    frames = {}
    for source in ["ablation", "families", "imbalance", "hpsens"]:
        parts = []
        for seed in SEEDS:
            path = ROOT / "data" / source / f"preds_seed{seed}.csv"
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            part = pd.read_csv(path)
            assert part.seed.eq(seed).all() and part.Experiment_Index.is_unique
            parts.append(part)
        frames[source] = pd.concat(parts, ignore_index=True).sort_values(["seed", "Experiment_Index"]).reset_index(drop=True)
    metadata = ["seed", "Experiment_Index", "Serial_Number", "Fault_Mode"]
    for frame in frames.values():
        pd.testing.assert_frame_equal(frame[metadata], frames["ablation"][metadata], check_dtype=False)
    for column in ["rf_F246", "lgbm_F246"]:
        frames["families"][column] = frames["ablation"][column]
    units = sorted(frames["ablation"].Serial_Number.unique())
    uidx = {s: i for i, s in enumerate(units)}
    cidx = {c: i for i, c in enumerate(FAULT_ORDER)}
    rng = np.random.default_rng(20261010)
    weights = rng.multinomial(len(units), np.ones(len(units)) / len(units), size=10000)
    cache = {}

    def arm(source, name):
        key = (source, name)
        if key not in cache:
            frame = frames[source]
            cm = np.zeros((len(SEEDS), len(units), 8, 8))
            for si, seed in enumerate(SEEDS):
                rows = frame[frame.seed.eq(seed)]
                np.add.at(cm[si], (rows.Serial_Number.map(uidx).to_numpy(), rows.Fault_Mode.map(cidx).to_numpy(),
                                  rows[name].map(cidx).to_numpy()), 1)
            point = {k: v.mean(axis=0) for k, v in confusion_metrics(cm.sum(axis=1)).items()}
            sample = {k: v.mean(axis=1) for k, v in confusion_metrics(np.einsum("bu,suij->bsij", weights, cm)).items()}
            cache[key] = point, sample
        return cache[key]

    comparisons = []
    def add(family, source, a, b, metric, size, scope):
        pa, ba = arm(source, a)
        pb, bb = arm(source, b)
        result = paired_comparison(ba[metric] - bb[metric], pa[metric] - pb[metric], MARGINS[metric], size)
        comparisons.append({"family": family, "source": source, "A": a, "B": b, "metric": metric, "scope": scope, **result})

    for other in ["lgbm_F246", "xgb_F246"]:
        for metric in ["CMR", "FA0"]:
            add("families", "families", "rf_F246", other, metric, 4, "retrospective_family")
    for model in ["lgbm", "rf"]:
        for metric in MARGINS:
            add("feature_reduction", "ablation", f"{model}_F246", f"{model}_F115", metric, 8, "retrospective_family")
        for metric in MARGINS:
            add(f"features18_{model}", "ablation", f"{model}_F246", f"{model}_F18", metric, 4, "exploratory")
    for a, b, metric in [
        ("lgbm_freq_sev_gaussian", "lgbm_freq_gaussian", "CMR"),
        ("lgbm_freq_sev_gaussian", "lgbm_freq_sev_none", "macro_F1"),
        ("lgbm_freq_sev_gaussian", "lgbm_freq_sev_smote", "macro_F1"),
    ]:
        add("imbalance", "imbalance", a, b, metric, 3, "retrospective_family")
    for model, prefix in [("lgbm", "L"), ("rf", "R")]:
        variants = [c for c in frames["hpsens"] if c.startswith(prefix)]
        for variant in variants:
            add(f"hpsens_{model}", "hpsens", variant, f"{model}_base", "CMR", len(variants), "retrospective_family" if model == "lgbm" else "exploratory")
    results = pd.DataFrame(comparisons)
    results["p_Holm"] = np.nan
    for _, group in results.groupby("family", sort=False):
        results.loc[group.index, "p_Holm"] = holm_adjust(group.p_centered.to_numpy())
    results.to_csv(OUT / "comparisons.csv", index=False)
    summary = {"scope": "retrospective_correction; families_and_FA0_margin_defined_after_existing_results",
               "bootstrap_resamples": 10000, "bootstrap_seed": 20261010, "units": len(units), "seeds": SEEDS,
               "seed_sampling": "same_cluster_weights_then_mean_metrics_over_fixed_seeds",
               "source_sha256": inputs, "max_discard_fraction": float(results.discard_fraction.max()),
               "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), ROOT / "src/pbit/inference.py", ROOT / "src/pbit/modeling.py"]},
               "equivalent_by_family": results.groupby("family").equivalent.sum().to_dict()}
    (OUT / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paper_rows = results[results.family.isin(["families", "feature_reduction", "imbalance"])]
    lines = ["% Source: data/restart/inference/comparisons.csv; scripts/reanalyze_inference.py.", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Comparison & Metric & Interval level & Corrected interval & $\delta_{\min}$ & Equivalent \\", r"\midrule"]
    for row in paper_rows.itertuples():
        if row.family == "families":
            name = "RF $-$ " + ("LightGBM" if row.B.startswith("lgbm") else "XGBoost")
        elif row.family == "feature_reduction":
            name = ("LightGBM" if row.A.startswith("lgbm") else "RF") + ": F246 $-$ F115"
        else:
            name = "P1: weights" if row.metric == "CMR" else ("P2: Gaussian $-$ none" if row.B.endswith("none") else "P3: Gaussian $-$ SMOTE")
        metric = "Macro-F1" if row.metric == "macro_F1" else row.metric
        lines.append(f"{name} & {metric} & {100*row.equivalence_level:.2f}\\% & [{row.equivalence_low:+.3f}; {row.equivalence_high:+.3f}] & {row.delta_min:.3f} & {'yes' if row.equivalent else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/supplement/tables/tab_corrected_inference.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results[["family", "A", "B", "metric", "estimate", "ci95_low", "ci95_high", "p_Holm", "equivalent", "delta_min"]].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ["source_sha256", "code_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
