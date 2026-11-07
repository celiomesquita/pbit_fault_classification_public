"""Protocol P-R1: can F04 be told apart from F05? Exact test over every assignment of the eight units to the two classes.

Statistic: AUC of pooled leave-one-unit-out predictions (F05 positive). Null: all 70 assignments of four units to each class
(35 partitions and their complements). Also exports the level shift relative to N01 in the same unit and the file position of each class.
Usage, from scripts/: py analyze_f04_f05.py. Outputs: resultados/f04_f05_separability.csv, f04_f05_level_shift.csv, f04_f05_position.csv.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experimentos_base as base
import train_binary_model as model

DATA = "pbit_features_v3.csv"
MODELS = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "forest": lambda: RandomForestClassifier(200, min_samples_leaf=3, random_state=0, n_jobs=1),
}


def louo_auc(factory, X, y, groups):
    scores = np.zeros(len(y))
    for unit in np.unique(groups):
        test = groups == unit
        scores[test] = factory().fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    return roc_auc_score(y, scores)


def main():
    output = base.SAIDA
    output.mkdir(exist_ok=True)
    df_all = pd.read_csv(base.RAIZ / "dados" / DATA)
    sets = model.build_feature_sets(df_all)
    keep = df_all["condicao"].isin(["F04", "F05"]).values
    df = df_all[keep].reset_index(drop=True)
    groups = df["numero_serie"].values
    units = sorted(np.unique(groups))
    observed_f05 = set(df.loc[df.condicao == "F05", "numero_serie"])
    assert len(units) == 8 and len(observed_f05) == 4

    rows = []
    for set_name in ("A", "D"):
        X = sets[set_name].values[keep]
        for name, factory in MODELS.items():
            null = []
            for chosen in combinations(units, 4):
                y = np.isin(groups, chosen).astype(int)
                null.append(louo_auc(factory, X, y, groups))
            observed = louo_auc(factory, X, np.isin(groups, list(observed_f05)).astype(int), groups)
            null = np.array(null)
            rows.append({"feature_set": set_name, "model": name, "observed_auc": observed, "n_assignments": len(null),
                         "p_value": (null >= observed - 1e-12).mean(), "null_median": np.median(null)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(output / "f04_f05_separability.csv", index=False, lineterminator="\n")

    medians = df_all.groupby(["numero_serie", "condicao"])["media"].median().unstack()
    shift = []
    for condition in ("F04", "F05"):
        for unit in medians.index[medians[condition].notna()]:
            n01 = medians.loc[unit, "N01"]
            shift.append({"condicao": condition, "numero_serie": unit, "level_ratio_to_n01": medians.loc[unit, condition] / n01 if pd.notna(n01) else np.nan})
    shift = pd.DataFrame(shift)
    shift.to_csv(output / "f04_f05_level_shift.csv", index=False, lineterminator="\n")
    print(shift.groupby("condicao").level_ratio_to_n01.agg(["median", "min", "max", "count"]).round(3).to_string())

    position = df.groupby("condicao").agg(first_idx=("experimento_idx", "min"), last_idx=("experimento_idx", "max"),
                                          units=("numero_serie", lambda s: " ".join(sorted(set(s)))))
    position["shared_units"] = len(set(df.loc[df.condicao == "F04", "numero_serie"]) & observed_f05)
    position.to_csv(output / "f04_f05_position.csv", lineterminator="\n")
    print(position.to_string())


if __name__ == "__main__":
    main()
