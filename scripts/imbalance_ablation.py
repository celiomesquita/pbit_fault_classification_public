from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
from population import load_train, out_dir  # noqa: E402
OUT = out_dir(DATA / "imbalance")
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
WEIGHTS = ["none", "freq", "sev", "freq_sev"]
AUGS = ["none", "gaussian", "smote"]
MINORITY = ["F04", "F05", "F06"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    seed = args.seed
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train(DATA / "train_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in train.columns if c not in META_COLS]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
        np.arange(len(train)), train["Fault_Mode"].values, train["Serial_Number"].values))
    parts = []
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr, te = train.iloc[tr_idx], train.iloc[te_idx]
        y_real = tr["Fault_Mode"].values
        X_real = tr[cols].values.astype(float)
        Xt = te[cols].values.astype(float)

        gaus = extract_features(augment_training_set(
            long_df, set(tr["Experiment_Index"]), target_per_class=160, noise_scale=0.01,
            random_state=seed + k)).reset_index(drop=True)
        data = {
            "none": (X_real, y_real),
            "gaussian": (np.vstack([X_real, gaus[cols].values.astype(float)]),
                         np.concatenate([y_real, gaus["Fault_Mode"].values])),
        }
        imp_s = SimpleImputer(strategy="median")
        Xs = imp_s.fit_transform(X_real)
        counts = pd.Series(y_real).value_counts()
        strategy = {c: 160 for c in MINORITY if counts.get(c, 0) < 160}
        k_nn = max(1, min(5, int(min(counts[c] for c in strategy)) - 1))
        Xr, yr = SMOTE(sampling_strategy=strategy, k_neighbors=k_nn, random_state=seed + k).fit_resample(Xs, y_real)
        data["smote"] = (Xr, yr)

        out = te[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        out.insert(0, "fold", k)
        out.insert(0, "seed", seed)
        for aug in AUGS:
            X, y = data[aug]
            imp = SimpleImputer(strategy="median")
            X_i = imp.fit_transform(X)
            Xt_i = imp.transform(Xt)
            for w in WEIGHTS:
                sw = compute_sample_weights(y, w)
                m = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": args.threads}).fit(X_i, y, sample_weight=sw)
                out[f"lgbm_{w}_{aug}"] = m.predict(Xt_i)
                rf = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                            random_state=42, n_jobs=args.threads)
                rf.fit(X_i, y, sample_weight=sw)
                out[f"rf_{w}_{aug}"] = rf.predict(Xt_i)
        parts.append(out)
        print(f"seed {seed} fold {k} done", flush=True)
    pd.concat(parts).to_csv(OUT / f"preds_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main()
