"""Protocol P-C3: robustness of the headline procedure (family chosen inside each outer fold) rather than a fixed logistic model.

Selection is restricted to feature set E (chosen in all 25 outer folds of the main analysis); families, grids and the 0.01
simplicity margin are unchanged. (i) unseen failure modes, (ii) failure-to-normal training-label errors, (iii) F02 escaping inspection.
Usage, from scripts/: py evaluate_robustness_procedure.py [--rapido].
Outputs: resultados/procedure_unseen_mode.csv, procedure_label_noise.csv, procedure_f02_escape.csv, procedure_robustness_choices.csv.
"""
import ast
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import evaluate_generalization as generalization
import evaluate_label_noise as noise
import experimentos_base as base
import train_binary_model as model

MODES = generalization.MODES
RATES = noise.RATES
ALPHAS = model.ALPHAS
INPUT = base.RAIZ / "resultados"  # main-analysis artefacts; outputs follow base.SAIDA


def select_procedure(X, y, conditions, groups, seed):
    """Within-fold selection on set E: inner grouped validation per family, then the simplest family within the margin."""
    candidates, scores = [], {}
    for name in model.GRIDS:
        params, inner_auc, oof = model.inner_fit(name, X, y, conditions, groups, seed)
        candidates.append({"feature_set": "E", "model": name, "params": str(params), "inner_auc": inner_auc})
        scores[name] = oof
    pick = model.select_simplest(pd.DataFrame(candidates))
    oof = scores[pick.model]
    thresholds = {alpha: np.quantile(oof[y == 0], 1 - alpha) for alpha in ALPHAS}
    fitted = model.make_model(pick.model, ast.literal_eval(pick.params), seed).fit(X, y)
    return fitted, thresholds, pick


def inner_scores(name, params, X, y, conditions, groups, seed):
    """Out-of-fold inner scores of a fixed configuration, with the same folds and seeds as train_binary_model.inner_fit."""
    scores = np.zeros(len(y))
    for train, test in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, conditions, groups):
        scores[test] = model.make_model(name, params, seed).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return scores


def frame(fitted, X, thresholds, groups, conditions, y, **extra):
    out = pd.DataFrame({**extra, "numero_serie": groups, "condicao": conditions, "y": y, "score": fitted.predict_proba(X)[:, 1]})
    return generalization.threshold_columns(out, thresholds)


def run_unseen(df, X, conditions, groups, choices):
    y = model.make_target(conditions)
    parts = []
    for repetition, mode, fold in product(range(base.N_REP), MODES, range(base.N_FOLD)):
        fold_ids = base.dobras(df, repetition)["E3b"]
        test_units = fold_ids == fold
        train = ~test_units & (conditions != mode)
        evaluated = test_units & (np.isin(conditions, model.NORMAL) | (conditions == mode))
        assert not set(groups[train]) & set(groups[evaluated])
        seed = 1000 * repetition + fold
        fitted, thresholds, pick = select_procedure(X[train], y[train], conditions[train], groups[train], seed)
        choices.append({"experiment": "unseen", "condition": mode, "repetition": repetition, "fold": fold, "model": pick.model, "params": pick.params})
        parts.append(frame(fitted, X[evaluated], thresholds, groups[evaluated], conditions[evaluated], y[evaluated], mode=mode, repetition=repetition, fold=fold))
    return pd.concat(parts, ignore_index=True)


def run_noise(df, X, conditions, groups, choices):
    y = model.make_target(conditions)
    parts = []
    for repetition, rate, fold in product(range(base.N_REP), RATES, range(base.N_FOLD)):
        fold_ids = base.dobras(df, repetition)["E3b"]
        test, train = fold_ids == fold, fold_ids != fold
        seed = 1000 * repetition + fold
        rng = np.random.default_rng([seed, int(rate * 100)])
        noise.corrupt_n01_rows(conditions, train, rate, rng)  # consumes the generator exactly as in evaluate_label_noise
        noisy = noise.flip_failure_labels(y, train, rate, rng)
        fitted, thresholds, pick = select_procedure(X[train], noisy[train], conditions[train], groups[train], seed)
        choices.append({"experiment": "label_noise", "condition": f"{rate:.2f}", "repetition": repetition, "fold": fold, "model": pick.model, "params": pick.params})
        parts.append(frame(fitted, X[test], thresholds, groups[test], conditions[test], y[test], rate=rate, repetition=repetition, fold=fold))
    return pd.concat(parts, ignore_index=True)


def run_escape(df_all, keep, df, X_all, saved):
    """F02 cycles of units outside training, scored by the configuration chosen in each outer fold of the main analysis."""
    choices = pd.read_csv(INPUT / "binary_no_f02_spectral_procedure_choices.csv")
    is_f02 = (df_all["condicao"] == "F02").values
    conditions, groups = df["condicao"].values, df["numero_serie"].values
    y = model.make_target(conditions)
    X = X_all[keep]
    rows, checks = [], []
    for row in choices.itertuples():
        assert row.feature_set == "E"
        repetition, fold = row.repetition, row.fold
        train = base.dobras(df, repetition)["E3b"] != fold
        seed = 1000 * repetition + fold
        params = ast.literal_eval(row.params)
        oof = inner_scores(row.model, params, X[train], y[train], conditions[train], groups[train], seed)
        threshold = np.quantile(oof[y[train] == 0], 1 - 0.20)
        stored = saved[(saved.repetition == repetition) & (saved.fold == fold) & (saved.feature_set == "E") & (saved.model == row.model)]["threshold_20"].iloc[0]
        checks.append(abs(threshold - stored))
        fitted = model.make_model(row.model, params, seed).fit(X[train], y[train])
        outside = is_f02 & ~df_all["numero_serie"].isin(set(groups[train])).values
        rows.append({"repetition": repetition, "fold": fold, "model": row.model, "n_f02": int(outside.sum()),
                     "alarms": int((fitted.predict_proba(X_all[outside])[:, 1] > threshold).sum())})
    assert max(checks) < 1e-9, f"recomputed thresholds differ from saved ones: {max(checks)}"
    out = pd.DataFrame(rows)
    return out, max(checks)


def intervals(predictions, key, rng):
    rows = []
    for value, part in predictions.groupby(key):
        rows += [row | {key: value} for row in model.with_intervals(part, str(value), rng)]
    return pd.DataFrame(rows).drop(columns="configuration")


def main():
    pd.set_option("display.width", 220, "display.max_rows", 200)
    output = base.SAIDA
    output.mkdir(exist_ok=True)
    df_all = generalization.load_all()
    keep = (df_all["condicao"] != "F02").values
    df = df_all[keep].reset_index(drop=True)
    conditions, groups = df["condicao"].values, df["numero_serie"].values
    X = model.build_feature_sets(df)["E"].values
    rng = np.random.default_rng(2026)
    choices = []

    saved = pd.read_csv(INPUT / "binary_no_f02_spectral_predictions.csv")
    escape, gap = run_escape(df_all, keep, df, model.build_feature_sets(df_all)["E"].values, saved)
    escape.to_csv(output / "procedure_f02_escape.csv", index=False, lineterminator="\n")
    print(f"threshold check max gap {gap:.2e}; F02 alarm rate {escape.alarms.sum() / escape.n_f02.sum():.3f}", flush=True)

    unseen = run_unseen(df, X, conditions, groups, choices)
    table = intervals(unseen, "mode", rng).assign(training="mode_not_seen")
    chosen = pd.read_csv(INPUT / "binary_no_f02_spectral_procedure_choices.csv")
    procedure = pd.concat([saved[(saved.repetition == r.repetition) & (saved.fold == r.fold) & (saved.feature_set == r.feature_set) & (saved.model == r.model)]
                           for r in chosen.itertuples()])
    seen = []
    for mode in MODES:
        subset = procedure[procedure["condicao"].isin(model.NORMAL + (mode,))]
        seen += [row | {"mode": mode, "training": "mode_seen"} for row in model.with_intervals(subset, mode, rng)]
    table = pd.concat([table, pd.DataFrame(seen).drop(columns="configuration")], ignore_index=True)
    table.to_csv(output / "procedure_unseen_mode.csv", index=False, lineterminator="\n")
    view = table[table.metric.isin(["auc", "recall_20", "false_alarm_20"])]
    print(view.pivot_table(index=["mode", "training"], columns="metric", values="estimate").round(3).to_string(), flush=True)

    noisy = run_noise(df, X, conditions, groups, choices)
    table = intervals(noisy, "rate", rng)
    table.to_csv(output / "procedure_label_noise.csv", index=False, lineterminator="\n")
    view = table[table.metric.isin(["auc", "recall_20", "false_alarm_20", "critical_miss_20"])]
    print(view.pivot_table(index="rate", columns="metric", values="estimate").round(3).to_string())

    choices = pd.DataFrame(choices)
    choices.to_csv(output / "procedure_robustness_choices.csv", index=False, lineterminator="\n")
    print(choices.groupby(["experiment", "model"]).size().to_string())


if __name__ == "__main__":
    main()
