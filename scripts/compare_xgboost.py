"""Post-result XGBoost extension of the conditional, spectral binary experiment.

Run: py scripts/compare_xgboost.py [--rapido]. Existing four-family predictions
are preserved. Reference logistic predictions are refitted to verify cache compatibility.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance

import train_binary_model as model


METRICS = ("auc", "recall_20", "false_alarm_20", "critical_miss_20")
SOURCE = "binary_no_f02_spectral"


def paired_intervals(first, second, label, draws):
    keys = ["repetition", "experimento_idx", "numero_serie", "condicao", "y"]
    first = first.sort_values(keys).reset_index(drop=True)
    second = second.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(first[keys], second[keys])
    units = sorted(first.numero_serie.unique())
    positions = first.numero_serie.map(dict(zip(units, range(len(units))))).values
    point_a, point_b = model.summarise(first), model.summarise(second)
    differences = []
    for draw in draws:
        weights = draw[positions].astype(float)
        if len(np.unique(first.y.values[weights > 0])) != 2:
            continue
        a, b = model.summarise(first, weights), model.summarise(second, weights)
        differences.append({metric: a[metric] - b[metric] for metric in METRICS})
    samples = pd.DataFrame(differences)
    return [{"comparison": label, "metric": metric,
             "estimate": point_a[metric] - point_b[metric],
             "ci_low": samples[metric].quantile(0.025),
             "ci_high": samples[metric].quantile(0.975)} for metric in METRICS]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root, output = model.base.RAIZ, model.base.SAIDA
    output.mkdir(exist_ok=True)
    model.USE_SPECTRAL = True
    model.GRIDS["xgboost"] = model.XGBOOST_GRID[:1] if model.base.RAPIDO else model.XGBOOST_GRID
    df = model.load_data(True)
    sets = model.build_feature_sets(df)
    conditions, groups = df.condicao.values, df.numero_serie.values
    y = model.make_target(conditions)
    paths = [root / "dados" / name for name in
             ("pbit_features_v3.csv", "pbit_spectral_features_v3.csv")]
    paths += [root / "resultados" / f"{SOURCE}_{suffix}.csv"
              for suffix in ("predictions", "inner_selection")]
    manifest = {
        "design": "Post-result conditional comparison; F02 excluded; same original outer/inner folds",
        "grid": model.GRIDS["xgboost"], "n_estimators": 200, "learning_rate": 0.05,
        "tree_method": "hist", "class_weight": "balanced within each fit",
        "early_stopping": False, "bootstrap_seed": 2026, "bootstrap_draws": model.N_BOOT,
        "simplicity_order": model.SIMPLICITY_ORDER, "simplicity_margin": model.SIMPLICITY_MARGIN,
        "python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
        "inputs_sha256": {str(path.relative_to(root)): sha256(path) for path in paths},
        "code_sha256": {str(path.relative_to(root)): sha256(path) for path in
                         (Path(__file__), root / "scripts" / "train_binary_model.py")},
    }
    (output / "xgboost_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cached = pd.read_csv(paths[-2])
    inner = pd.read_csv(paths[-1])
    cached = cached[cached.repetition < model.base.N_REP].copy()
    inner = inner[inner.repetition < model.base.N_REP].copy()
    if model.base.RAPIDO:
        # The compatibility check uses the original grid, even in a smoke run.
        model.GRIDS["logistic"] = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
    reference, old_choices = model.procedure_predictions(cached, inner)
    parts, selections, importances = [], [], []
    for repetition in range(model.base.N_REP):
        fold_ids = model.base.dobras(df, repetition)["E3b"]
        for fold in range(model.base.N_FOLD):
            seed = 1000 * repetition + fold
            train, test = fold_ids != fold, fold_ids == fold
            assert not set(groups[train]) & set(groups[test])
            for set_name, frame in sets.items():
                old = cached[(cached.repetition == repetition) & (cached.fold == fold)
                             & (cached.feature_set == set_name) & (cached.model == "logistic")]
                np.testing.assert_array_equal(old.experimento_idx, df.experimento_idx.values[test])
                np.testing.assert_array_equal(old.y, y[test])
                np.testing.assert_array_equal(old.numero_serie, groups[test])
                np.testing.assert_array_equal(old.condicao, conditions[test])
                X = frame.values
                params, auc, scores = model.inner_fit("logistic", X[train], y[train], conditions[train], groups[train], seed)
                check = model.make_model("logistic", params, seed).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
                np.testing.assert_allclose(check, old.score, rtol=1e-10, atol=1e-12)
                for alpha in model.ALPHAS:
                    np.testing.assert_allclose(old[f"threshold_{int(alpha * 100):02d}"],
                                               np.quantile(scores[y[train] == 0], 1 - alpha), rtol=1e-10, atol=1e-12)
                params, auc, scores = model.inner_fit("xgboost", X[train], y[train], conditions[train], groups[train], seed)
                fitted = model.make_model("xgboost", params, seed).fit(X[train], y[train])
                row = old.copy()
                row["model"] = "xgboost"
                row["score"] = fitted.predict_proba(X[test])[:, 1]
                for alpha in model.ALPHAS:
                    row[f"threshold_{int(alpha * 100):02d}"] = np.quantile(scores[y[train] == 0], 1 - alpha)
                parts.append(row)
                selections.append({"repetition": repetition, "fold": fold, "feature_set": set_name,
                                   "model": "xgboost", "params": str(params), "inner_auc": auc})
                if set_name == "E":
                    importance = permutation_importance(fitted, X[test], y[test], scoring="roc_auc",
                                                        n_repeats=10, random_state=seed, n_jobs=1)
                    importances.extend({"repetition": repetition, "fold": fold, "feature": feature,
                                        "auc_drop": value} for feature, value in zip(frame.columns, importance.importances_mean))
            pd.concat(parts).to_csv(output / "xgboost_predictions.csv", index=False, lineterminator="\n")
            pd.DataFrame(selections).to_csv(output / "xgboost_inner_selection.csv", index=False, lineterminator="\n")
            print(f"repetition {repetition}, fold {fold}: reference verified; XGBoost complete", flush=True)
    extra = pd.concat(parts, ignore_index=True)
    expanded, choices = model.procedure_predictions(pd.concat([cached, extra]), pd.concat([inner, pd.DataFrame(selections)]))
    choices.to_csv(output / "xgboost_expanded_procedure_choices.csv", index=False, lineterminator="\n")
    expanded.to_csv(output / "xgboost_expanded_procedure_predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(importances).to_csv(output / "xgboost_importance_by_fold.csv", index=False, lineterminator="\n")
    pd.DataFrame(importances).groupby("feature").auc_drop.agg(["mean", "std"]).sort_values("mean", ascending=False).to_csv(
        output / "xgboost_importance.csv", lineterminator="\n")
    frames = {"original_procedure": reference, "expanded_procedure": expanded}
    for family in ("logistic", "forest", "boosting"):
        frames[f"E_{family}"] = cached[(cached.feature_set == "E") & (cached.model == family)]
    for set_name in sets:
        frames[f"{set_name}_xgboost"] = extra[extra.feature_set == set_name]
    results = []
    for label, frame in frames.items():
        results += model.with_intervals(frame, label, np.random.default_rng(2026))
    pd.DataFrame(results).to_csv(output / "xgboost_results.csv", index=False, lineterminator="\n")
    units = sorted(df.numero_serie.unique())
    draws = np.random.default_rng(2026).multinomial(len(units), np.full(len(units), 1 / len(units)), size=model.N_BOOT)
    differences = []
    for comparator in ("E_logistic", "E_forest", "E_boosting", "original_procedure"):
        differences += paired_intervals(frames["E_xgboost"], frames[comparator], f"E_xgboost_minus_{comparator}", draws)
    differences += paired_intervals(expanded, reference, "expanded_minus_original_procedure", draws)
    pd.DataFrame(differences).to_csv(output / "xgboost_paired_differences.csv", index=False, lineterminator="\n")
    print(choices.groupby(["feature_set", "model"]).size().to_string(), flush=True)
    print(pd.DataFrame(differences).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
