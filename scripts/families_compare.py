from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
from pbit.modeling import META_COLS, XGBWrapper, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
from population import load_train, out_dir  # noqa: E402
OUT = out_dir(DATA / "families")


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
        xgb = XGBWrapper(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="mlogloss", random_state=42,
                         verbosity=0, n_jobs=args.threads)
        xgb.fit(X, y, sample_weight=sw)
        out["xgb_F246"] = xgb.predict(Xte)
        imp = SimpleImputer(strategy="median")
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(imp.fit_transform(X), y, sample_weight=sw)
        out["gb_F246"] = gb.predict(imp.transform(Xte))
        parts.append(out)
        print(f"seed {seed} fold {k} done", flush=True)
    pd.concat(parts).to_csv(OUT / f"preds_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main()
