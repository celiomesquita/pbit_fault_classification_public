"""Plan action P20: how much labelling error can the models tolerate? Errors are injected only in training data.

N01-only detector: failure cycles enter the N01 training set as if mislabelled. Supervised model: failure labels are flipped to normal.
Thresholds come from the same corrupted labels, as they would in real use. Test labels are never changed.
Usage, from scripts/: py evaluate_label_noise.py. Output: resultados/label_noise.csv.
"""
from itertools import product

import numpy as np
import pandas as pd

import evaluate_generalization as generalization
import experimentos_base as base
import train_binary_model as model

RATES = (0.0, 0.01, 0.05, 0.10)
ALPHAS = model.ALPHAS


def corrupt_n01_rows(conditions, train, rate, rng):
    """Boolean mask of the rows used as N01: true N01 plus failure rows amounting to `rate` of the true N01 count."""
    n01 = train & (conditions == "N01")
    failures = np.flatnonzero(train & ~np.isin(conditions, model.NORMAL))
    chosen = rng.choice(failures, size=min(int(round(rate * n01.sum())), len(failures)), replace=False)
    mask = n01.copy()
    mask[chosen] = True
    return mask


def flip_failure_labels(y, train, rate, rng):
    """Copy of y with a share `rate` of the training failures labelled as normal."""
    noisy = y.copy()
    failures = np.flatnonzero(train & (y == 1))
    noisy[rng.choice(failures, size=int(round(rate * len(failures))), replace=False)] = 0
    return noisy


def run(df, X, conditions, groups):
    y = model.make_target(conditions)
    parts = []
    for repetition, rate, fold in product(range(base.N_REP), RATES, range(base.N_FOLD)):
        fold_ids = base.dobras(df, repetition)["E3b"]
        test, train = fold_ids == fold, fold_ids != fold
        seed = 1000 * repetition + fold
        rng = np.random.default_rng([seed, int(rate * 100)])
        common = {"rate": rate, "repetition": repetition, "fold": fold, "numero_serie": groups[test], "condicao": conditions[test], "y": y[test]}

        used = corrupt_n01_rows(conditions, train, rate, rng)
        detector = generalization.MahalanobisDetector().fit(X[used])
        thresholds = generalization.detector_thresholds(generalization.MahalanobisDetector, X[used], groups[used])
        parts.append(generalization.threshold_columns(pd.DataFrame({**common, "model": "n01_only", "score": detector.score(X[test])}), thresholds))

        noisy = flip_failure_labels(y, train, rate, rng)
        fitted = model.make_model("logistic", generalization.LOGISTIC, seed).fit(X[train], noisy[train])
        thresholds = generalization.supervised_thresholds(X[train], noisy[train], conditions[train], groups[train], seed)
        parts.append(generalization.threshold_columns(pd.DataFrame({**common, "model": "supervised", "score": fitted.predict_proba(X[test])[:, 1]}), thresholds))
    return pd.concat(parts, ignore_index=True)


def main():
    df_all = generalization.load_all()
    df = df_all[df_all["condicao"] != "F02"].reset_index(drop=True)
    X = model.build_feature_sets(df)["E"].values
    results = run(df, X, df["condicao"].values, df["numero_serie"].values)
    rng = np.random.default_rng(2026)
    rows = []
    for (name, rate), frame in results.groupby(["model", "rate"]):
        rows += [row | {"model": name, "rate": rate} for row in model.with_intervals(frame, name, rng)]
    table = pd.DataFrame(rows).drop(columns="configuration")
    table.to_csv(base.RAIZ / "resultados" / "label_noise.csv", index=False, lineterminator="\n")
    view = table[table.metric.isin(["auc", "recall_20", "false_alarm_20", "critical_miss_20"])]
    pd.set_option("display.width", 200)
    print(view.assign(text=view.estimate.round(3).astype(str) + " [" + view.ci_low.round(3).astype(str) + "; " + view.ci_high.round(3).astype(str) + "]").pivot(index=["model", "rate"], columns="metric", values="text").to_string())


if __name__ == "__main__":
    main()
