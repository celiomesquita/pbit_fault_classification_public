"""E9: leakage contrast with a fixed model (docs/protocolos/E9.md)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from leakage_stratification import UNTUNED, make_folds  # noqa: E402
from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

OUT = ROOT / "data/e9_fixed_model"
META = ["Experiment_Index", "Serial_Number", "Fault_Mode"]


def split_fit_leak(train_units: pd.DataFrame, seed: int):
    """80% fit / 20% reserved cycles within the training units; reserved cycles of units without
    any fit cycle are dropped (they would not be leakage)."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed+1000).split(
        np.arange(len(train_units)), train_units["Fault_Mode"].values))
    fit_idx, reserved_idx = folds[0]
    fit, reserved = train_units.iloc[fit_idx], train_units.iloc[reserved_idx]
    seen = reserved.Serial_Number.isin(set(fit.Serial_Number))
    return fit, reserved[seen], int((~seen).sum())


def run_seed(seed: int, threads: int, train: pd.DataFrame, long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = [c for c in train.columns if c not in META_COLS]
    parts, dropped = [], []
    for k, (tr_idx, te_idx) in enumerate(make_folds(train, "grouped_faultmode", seed)):
        units_train, unseen = train.iloc[tr_idx], train.iloc[te_idx]
        assert set(units_train.Serial_Number).isdisjoint(unseen.Serial_Number)
        fit, leak, n_dropped = split_fit_leak(units_train, seed)
        assert set(fit.Experiment_Index).isdisjoint(leak.Experiment_Index)
        assert set(leak.Serial_Number) <= set(fit.Serial_Number)
        assert set(fit.Serial_Number).isdisjoint(unseen.Serial_Number)
        dropped.append(n_dropped)
        aug = extract_features(augment_training_set(
            long_df, set(fit["Experiment_Index"]), target_per_class=160, noise_scale=0.01,
            random_state=seed + k)).reset_index(drop=True)
        y = np.concatenate([fit["Fault_Mode"].values, aug["Fault_Mode"].values])
        X = np.vstack([fit[cols].values, aug[cols].values]).astype(float)
        sw = compute_sample_weights(y, "freq_sev")
        model_lgb = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": threads}).fit(X, y, sample_weight=sw)
        imputer = SimpleImputer(strategy="median")
        model_rf = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                          random_state=42, n_jobs=threads)
        model_rf.fit(imputer.fit_transform(X), y, sample_weight=sw)
        for name, frame in (("leak", leak), ("unseen", unseen)):
            out = frame[META].copy()
            Xt = frame[cols].values.astype(float)
            out.insert(0, "split", name)
            out.insert(0, "fold", k)
            out.insert(0, "seed", seed)
            out["lgbm"] = model_lgb.predict(Xt)
            out["rf"] = model_rf.predict(imputer.transform(Xt))
            parts.append(out)
    frame = pd.concat(parts, ignore_index=True)
    return frame, dict(seed=seed, dropped_reserved_cycles=dropped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/train_real.csv")
    long_df = load_long_dataset(ROOT / "data/PBIT_atuador_DESCARAC_r1_long_clean.csv")
    frame, info = run_seed(args.seed, args.threads, train, long_df)
    frame.to_csv(OUT / f"preds_seed{args.seed}.csv", index=False)
    (OUT / f"info_seed{args.seed}.json").write_text(json.dumps(info), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
