from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = DATA / "hpsens"

LGB_BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
LGB_VARIANTS = {
    "L01_trees100": dict(n_estimators=100),
    "L02_trees200": dict(n_estimators=200),
    "L03_trees800": dict(n_estimators=800),
    "L04_lr0.02": dict(learning_rate=0.02),
    "L05_lr0.10": dict(learning_rate=0.10),
    "L06_leaves15": dict(num_leaves=15),
    "L07_leaves31": dict(num_leaves=31),
    "L08_leaves127": dict(num_leaves=127),
    "L09_mcs2": dict(min_child_samples=2),
    "L10_mcs20": dict(min_child_samples=20),
    "L11_colsample0.5": dict(colsample_bytree=0.5),
    "L12_colsample1.0": dict(colsample_bytree=1.0),
    "L13_bagging_freq1": dict(subsample_freq=1),
}
RF_BASE = dict(n_estimators=400, max_features="sqrt", min_samples_leaf=2, random_state=42)
RF_VARIANTS = {
    "R01_trees100": dict(n_estimators=100),
    "R02_trees1000": dict(n_estimators=1000),
    "R03_msl1": dict(min_samples_leaf=1),
    "R04_msl5": dict(min_samples_leaf=5),
    "R05_maxfeat_log2": dict(max_features="log2"),
    "R06_maxfeat0.3": dict(max_features=0.3),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--base-only", action="store_true", help="refit only the base arms (reproducibility check)")
    args = ap.parse_args()
    seed = args.seed
    OUT.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in train.columns if c not in META_COLS]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
        np.arange(len(train)), train["Fault_Mode"].values, train["Serial_Number"].values))

    lgb_arms = {"lgbm_base": {}} if args.base_only else {"lgbm_base": {}, **LGB_VARIANTS}
    rf_arms = {"rf_base": {}} if args.base_only else {"rf_base": {}, **RF_VARIANTS}
    parts = []
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr, te = train.iloc[tr_idx], train.iloc[te_idx]
        aug_long = augment_training_set(long_df, set(tr["Experiment_Index"]), target_per_class=160,
                                        noise_scale=0.01, random_state=seed + k)
        aug = extract_features(aug_long).reset_index(drop=True)
        y = np.concatenate([tr["Fault_Mode"].values, aug["Fault_Mode"].values])
        X = np.vstack([tr[cols].values, aug[cols].values]).astype(float)
        Xte = te[cols].values.astype(float)
        sw = compute_sample_weights(y, "freq_sev")
        out = te[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        out.insert(0, "fold", k)
        out.insert(0, "seed", seed)
        for name, upd in lgb_arms.items():
            m = lgb.LGBMClassifier(**{**LGB_BASE, **upd, "n_jobs": args.threads}).fit(X, y, sample_weight=sw)
            out[name] = m.predict(Xte)
        imp = SimpleImputer(strategy="median")
        Xi, Xtei = imp.fit_transform(X), imp.transform(Xte)
        for name, upd in rf_arms.items():
            rf = RandomForestClassifier(**{**RF_BASE, **upd, "n_jobs": args.threads}).fit(Xi, y, sample_weight=sw)
            out[name] = rf.predict(Xtei)
        parts.append(out)
        print(f"seed {seed} fold {k} done", flush=True)

    name = f"base_seed{seed}.csv" if args.base_only else f"preds_seed{seed}.csv"
    pd.concat(parts).to_csv(OUT / name, index=False)


if __name__ == "__main__":
    main()
