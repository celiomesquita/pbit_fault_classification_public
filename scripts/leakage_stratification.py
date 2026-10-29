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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, SEVERITY_MAP, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
from population import load_train, out_dir  # noqa: E402
OUT = out_dir(DATA / "leakage")
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)


def make_folds(train, arm, seed):
    X = np.arange(len(train))
    fm = train["Fault_Mode"].values
    groups = train["Serial_Number"].values
    if arm == "grouped_faultmode":
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, fm, groups))
    if arm == "grouped_severity":
        y = np.array([SEVERITY_MAP[c] for c in fm])
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))
    if arm == "random_stratified":
        return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, fm))
    raise ValueError(arm)


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
    parts = []
    for arm in ("grouped_faultmode", "grouped_severity", "random_stratified"):
        for k, (tr_idx, te_idx) in enumerate(make_folds(train, arm, seed)):
            tr, te = train.iloc[tr_idx], train.iloc[te_idx]
            aug = extract_features(augment_training_set(
                long_df, set(tr["Experiment_Index"]), target_per_class=160, noise_scale=0.01,
                random_state=seed + k)).reset_index(drop=True)
            y = np.concatenate([tr["Fault_Mode"].values, aug["Fault_Mode"].values])
            X = np.vstack([tr[cols].values, aug[cols].values]).astype(float)
            Xt = te[cols].values.astype(float)
            sw = compute_sample_weights(y, "freq_sev")
            out = te[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
            out.insert(0, "fold", k)
            out.insert(0, "arm", arm)
            out.insert(0, "seed", seed)
            m = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": args.threads}).fit(X, y, sample_weight=sw)
            out["lgbm"] = m.predict(Xt)
            imp = SimpleImputer(strategy="median")
            rf = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                        random_state=42, n_jobs=args.threads)
            rf.fit(imp.fit_transform(X), y, sample_weight=sw)
            out["rf"] = rf.predict(imp.transform(Xt))
            parts.append(out)
        print(f"seed {seed} {arm} done", flush=True)
    pd.concat(parts).to_csv(OUT / f"preds_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main()
