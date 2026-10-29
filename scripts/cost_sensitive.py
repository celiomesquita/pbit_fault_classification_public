from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nested_cv_tuning as nt  # noqa: E402
from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, SEVERITY_MAP, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = DATA / "cost_sensitive"
KAPPAS = [2, 3, 5, 10, 20]
THREADS = 8
N_BAGS = 10
CLASSES = ["F01", "F02", "F03", "F04", "F05", "F06", "N01", "N02"]  # sorted order of LightGBM classes_


def cost_matrix(kappa: float) -> np.ndarray:
    """C[j, i]: cost of predicting class i when the true class is j."""
    sev = np.array([SEVERITY_MAP[c] for c in CLASSES])
    C = np.zeros((len(CLASSES), len(CLASSES)))
    for j in range(len(CLASSES)):
        for i in range(len(CLASSES)):
            if i == j:
                continue
            if sev[i] < sev[j]:
                C[j, i] = kappa * (sev[j] - sev[i])
            elif sev[i] > sev[j]:
                C[j, i] = sev[i] - sev[j]
            else:
                C[j, i] = 1.0
    return C


def proba_full(model, X):
    P = np.zeros((len(X), len(CLASSES)))
    p = model.predict_proba(X)
    for ci, c in enumerate(model.classes_):
        P[:, CLASSES.index(c)] = p[:, ci]
    return P


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in train.columns if c not in META_COLS]
    rng = np.random.default_rng(42)
    parts = []
    for k, (otr, ote) in enumerate(nt.folds_for(train, 5)):
        otrain, otest = train.iloc[otr].reset_index(drop=True), train.iloc[ote].reset_index(drop=True)
        aug = extract_features(augment_training_set(long_df, set(otrain["Experiment_Index"]), target_per_class=160,
                                                    noise_scale=0.01, random_state=42)).reset_index(drop=True)
        X = np.vstack([otrain[cols].values, aug[cols].values]).astype(float)
        y = np.concatenate([otrain["Fault_Mode"].values, aug["Fault_Mode"].values])
        Xt = otest[cols].values.astype(float)
        sw = compute_sample_weights(y, "freq_sev")

        base = lgb.LGBMClassifier(**{**nt.UNTUNED, "n_jobs": THREADS}).fit(X, y, sample_weight=sw)
        P_test = proba_full(base, Xt)

        P_bag = np.zeros((len(X), len(CLASSES)))
        for b in range(N_BAGS):
            idx = rng.integers(0, len(X), len(X))
            m = lgb.LGBMClassifier(**{**nt.UNTUNED, "n_estimators": 200, "random_state": 100 + b,
                                      "n_jobs": THREADS}).fit(X[idx], y[idx], sample_weight=sw[idx])
            P_bag += proba_full(m, X)
        P_bag /= N_BAGS

        out = otest[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        for kappa in KAPPAS:
            C = cost_matrix(kappa)
            out[f"mec_k{kappa}"] = np.array(CLASSES)[np.argmin(P_test @ C, axis=1)]
            y_new = np.array(CLASSES)[np.argmin(P_bag @ C, axis=1)]
            m2 = lgb.LGBMClassifier(**{**nt.UNTUNED, "n_jobs": THREADS}).fit(
                X, y_new, sample_weight=compute_sample_weights(y_new, "freq_sev"))
            out[f"mc_k{kappa}"] = m2.predict(Xt)
            out[f"mc_relabelled_k{kappa}"] = (y_new != y).mean()
        parts.append(out)
        print(f"fold {k} done", flush=True)
    pd.concat(parts).to_csv(OUT / "preds.csv", index=False)


if __name__ == "__main__":
    main()
