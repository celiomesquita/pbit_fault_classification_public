"""Test whether F02 can be told from N01 in units never seen, using cycle-relative and frequency descriptors (criterion C7).
Usage, from scripts/: py analyze_f02_vs_n01.py. Output: resultados/f02_vs_n01.csv."""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import experimentos_base as base
import train_binary_model as model

N_REPETITIONS = 5
N_PERMUTATIONS = 200
N_BOOT = 2000
SETTINGS = {"logistic": {"C": 0.1}, "forest": {"min_samples_leaf": 3}}


def out_of_fold(name, X, y, groups, seed):
    scores = np.zeros(len(y))
    for fold, (train, test) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)):
        scores[test] = model.make_model(name, SETTINGS[name], 1000 * seed + fold).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return scores


def main():
    df = pd.read_csv(base.RAIZ / "dados" / "pbit_features_v3.csv")
    spectral = pd.read_csv(base.RAIZ / "dados" / "pbit_spectral_features_v3.csv")
    relative = model.build_feature_sets(df)["D"]
    frequency = spectral.drop(columns="experimento_idx")
    sets = {"D_relative": relative, "S_frequency": frequency, "E_relative_and_frequency": pd.concat([relative, frequency], axis=1)}
    mask = df["condicao"].isin(["N01", "F02"]).values
    y = (df["condicao"].values[mask] == "F02").astype(int)
    groups = df["numero_serie"].values[mask]
    units = sorted(set(groups))
    position = pd.Series(groups).map({unit: i for i, unit in enumerate(units)}).values
    rng = np.random.default_rng(2026)

    rows = []
    for (set_name, frame), name in product(sets.items(), SETTINGS):
        X = frame.values[mask]
        scores = np.column_stack([out_of_fold(name, X, y, groups, seed) for seed in range(N_REPETITIONS)]).mean(axis=1)
        samples = []
        for draw in rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=N_BOOT):
            weights = draw[position].astype(float)
            if y[weights > 0].min() != y[weights > 0].max():
                samples.append(roc_auc_score(y, scores, sample_weight=weights))
        real = np.mean([roc_auc_score(y, out_of_fold(name, X, y, groups, 100 + seed)) for seed in range(N_REPETITIONS)])
        null = []
        blocks = pd.Series(groups + "|" + y.astype(str))
        unique_blocks = blocks.drop_duplicates()
        n_permutations = N_PERMUTATIONS if name == "logistic" else 50
        for permutation in range(n_permutations):
            labels = rng.permutation(unique_blocks.str.split("|").str[1].values)
            shuffled = blocks.map(dict(zip(unique_blocks.values, labels))).astype(int).values
            null.append(roc_auc_score(shuffled, out_of_fold(name, X, shuffled, groups, permutation)))
        rows.append({"feature_set": set_name, "model": name, "auc": real, "ci_low": np.quantile(samples, 0.025), "ci_high": np.quantile(samples, 0.975),
                     "null_mean": np.mean(null), "null_p97_5": np.quantile(null, 0.975), "p_value": (np.sum(np.array(null) >= real) + 1) / (len(null) + 1), "n_permutations": n_permutations})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(base.RAIZ / "resultados" / "f02_vs_n01.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
