"""Plan actions P18 and P19: detection of a failure mode never seen in training, and a detector trained only on N01.
Usage, from scripts/: py evaluate_generalization.py. Outputs: resultados/unseen_mode.csv, resultados/n01_only.csv, resultados/f02_escape.csv."""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import experimentos_base as base
import train_binary_model as model

ALPHAS = model.ALPHAS
MODES = ["F01", "F03", "F04", "F05", "F06"]
LOGISTIC = {"C": 0.01}
METRICS = ["auc", "recall_20", "false_alarm_20", "critical_miss_20", "recall_10", "false_alarm_10", "critical_miss_10"]


def load_all():
    df = pd.read_csv(base.RAIZ / "dados" / "pbit_features_v3.csv")
    return df.merge(pd.read_csv(base.RAIZ / "dados" / "pbit_spectral_features_v3.csv"), on="experimento_idx", validate="one_to_one")


class MahalanobisDetector:
    """Distance of a cycle from the N01 cloud, with a shrunk covariance so that it stays stable with many correlated features."""

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.covariance = LedoitWolf().fit(self.scaler.transform(X))
        return self

    def score(self, X):
        return self.covariance.mahalanobis(self.scaler.transform(X))


class IsolationDetector:
    def fit(self, X):
        self.forest = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        return -self.forest.score_samples(X)


def threshold_columns(frame, thresholds):
    for alpha in ALPHAS:
        frame[f"threshold_{int(alpha * 100):02d}"] = thresholds[alpha]
    return frame


def supervised_thresholds(X, y, conditions, groups, seed):
    scores = np.zeros(len(y))
    for train, test in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, conditions, groups):
        scores[test] = model.make_model("logistic", LOGISTIC, seed).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return {alpha: np.quantile(scores[y == 0], 1 - alpha) for alpha in ALPHAS}


def run_unseen_mode(df, X, conditions, groups):
    """Train without one mode in each fold; test on that mode and on normal cycles of units outside the training set."""
    y = model.make_target(conditions)
    parts = []
    for repetition, mode, fold in product(range(base.N_REP), MODES, range(base.N_FOLD)):
        fold_ids = base.dobras(df, repetition)["E3b"]
        test_units = fold_ids == fold
        train = ~test_units & (conditions != mode)
        evaluated = test_units & (np.isin(conditions, model.NORMAL) | (conditions == mode))
        assert not set(groups[train]) & set(groups[evaluated])
        seed = 1000 * repetition + fold
        fitted = model.make_model("logistic", LOGISTIC, seed).fit(X[train], y[train])
        thresholds = supervised_thresholds(X[train], y[train], conditions[train], groups[train], seed)
        parts.append(threshold_columns(pd.DataFrame({"mode": mode, "repetition": repetition, "fold": fold, "numero_serie": groups[evaluated], "condicao": conditions[evaluated],
                                                     "y": y[evaluated], "score": fitted.predict_proba(X[evaluated])[:, 1]}), thresholds))
    return pd.concat(parts, ignore_index=True)


def detector_thresholds(factory, X_n01, groups_n01):
    scores = np.zeros(len(X_n01))
    for train, test in GroupKFold(3).split(X_n01, groups=groups_n01):
        scores[test] = factory().fit(X_n01[train]).score(X_n01[test])
    return {alpha: np.quantile(scores, 1 - alpha) for alpha in ALPHAS}


def run_n01_only(df, sets, conditions, groups):
    y = model.make_target(conditions)
    detectors = {"mahalanobis": MahalanobisDetector, "isolation_forest": IsolationDetector}
    parts = []
    for repetition, (set_name, X), (name, factory), fold in product(range(base.N_REP), sets.items(), detectors.items(), range(base.N_FOLD)):
        fold_ids = base.dobras(df, repetition)["E3b"]
        test = fold_ids == fold
        train = ~test & (conditions == "N01")
        detector = factory().fit(X[train])
        thresholds = detector_thresholds(factory, X[train], groups[train])
        parts.append(threshold_columns(pd.DataFrame({"configuration": f"{name}_{set_name}", "repetition": repetition, "fold": fold, "numero_serie": groups[test], "condicao": conditions[test],
                                                     "y": y[test], "score": detector.score(X[test])}), thresholds))
    return pd.concat(parts, ignore_index=True)


def f02_escape(df_all, keep, sets_all, df):
    """Alarm rate on F02 cycles from units that never entered training, for the two main models at the adopted point."""
    is_f02 = (df_all["condicao"] == "F02").values
    conditions, groups = df["condicao"].values, df["numero_serie"].values
    y = model.make_target(conditions)
    X_all = sets_all["E"].values
    X = X_all[keep]
    rows = []
    for repetition, fold in product(range(base.N_REP), range(base.N_FOLD)):
        fold_ids = base.dobras(df, repetition)["E3b"]
        train = fold_ids != fold
        outside = is_f02 & ~df_all["numero_serie"].isin(set(groups[train])).values
        seed = 1000 * repetition + fold
        logistic = model.make_model("logistic", LOGISTIC, seed).fit(X[train], y[train])
        threshold = supervised_thresholds(X[train], y[train], conditions[train], groups[train], seed)[0.20]
        rows.append({"model": "logistic_E", "alarm_rate": (logistic.predict_proba(X_all[outside])[:, 1] > threshold).mean()})
        n01 = train & (conditions == "N01")
        detector = MahalanobisDetector().fit(X[n01])
        threshold = detector_thresholds(MahalanobisDetector, X[n01], groups[n01])[0.20]
        rows.append({"model": "mahalanobis_E", "alarm_rate": (detector.score(X_all[outside]) > threshold).mean()})
    return pd.DataFrame(rows).groupby("model")["alarm_rate"].agg(["mean", "min", "max"]).reset_index()


def main():
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_columns", 30)
    output = base.RAIZ / "resultados"
    df_all = load_all()
    keep = (df_all["condicao"] != "F02").values
    df = df_all[keep].reset_index(drop=True)
    conditions, groups = df["condicao"].values, df["numero_serie"].values
    sets = model.build_feature_sets(df)
    rng = np.random.default_rng(2026)

    unseen = run_unseen_mode(df, sets["E"].values, conditions, groups)
    seen = pd.read_csv(output / "binary_no_f02_spectral_predictions.csv")
    seen = seen[(seen.feature_set == "E") & (seen.model == "logistic")]
    rows = []
    for mode in MODES:
        rows += [row | {"mode": mode, "training": "mode_not_seen"} for row in model.with_intervals(unseen[unseen["mode"] == mode], mode, rng)]
        subset = seen[seen["condicao"].isin(model.NORMAL + (mode,))]
        rows += [row | {"mode": mode, "training": "mode_seen"} for row in model.with_intervals(subset, mode, rng)]
    unseen_table = pd.DataFrame(rows).drop(columns="configuration")
    unseen_table.to_csv(output / "unseen_mode.csv", index=False, lineterminator="\n")
    view = unseen_table[unseen_table.metric.isin(["auc", "recall_20", "false_alarm_20"])]
    print(view.assign(text=view.estimate.round(3).astype(str) + " [" + view.ci_low.round(3).astype(str) + "; " + view.ci_high.round(3).astype(str) + "]").pivot(index=["mode", "training"], columns="metric", values="text").to_string())

    selected = {"E": sets["E"].values, "D": sets["D"].values, "S": sets["E"][[c for c in sets["E"].columns if c.startswith(("spectral_", "lag"))]].values}
    n01_only = run_n01_only(df, selected, conditions, groups)
    rows = []
    for configuration, frame in n01_only.groupby("configuration"):
        rows += model.with_intervals(frame, configuration, rng)
    table = pd.DataFrame(rows)
    table.to_csv(output / "n01_only.csv", index=False, lineterminator="\n")
    view = table[table.metric.isin(METRICS)]
    print(view.assign(text=view.estimate.round(3).astype(str) + " [" + view.ci_low.round(3).astype(str) + "; " + view.ci_high.round(3).astype(str) + "]").pivot(index="configuration", columns="metric", values="text")[METRICS].to_string())
    rates = n01_only.assign(alarm=n01_only["score"] > n01_only["threshold_20"]).groupby(["configuration", "condicao"])["alarm"].mean().unstack().round(3)
    rates.to_csv(output / "n01_only_alarm_rate_by_condition_at_20.csv", lineterminator="\n")
    print(rates.to_string())
    primary = n01_only[n01_only.configuration == "mahalanobis_E"]
    critical = primary.assign(alarm=primary["score"] > primary["threshold_20"])
    print(critical[critical.condicao.isin(["F03", "F05"])].groupby(["condicao", "numero_serie"]).alarm.mean().round(2).to_dict())

    escape = f02_escape(df_all, keep, model.build_feature_sets(df_all), df)
    escape.to_csv(output / "f02_escape.csv", index=False, lineterminator="\n")
    print(escape.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
