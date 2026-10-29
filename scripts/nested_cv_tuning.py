from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import (  # noqa: E402
    META_COLS,
    compute_safety_metrics,
    compute_sample_weights,
    evaluate_predictions,
)

DATA = PROJECT_ROOT / "data"
SEED = 42
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)


def folds_for(df: pd.DataFrame, n: int):
    X = np.arange(len(df))
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=SEED).split(
        X, df["Fault_Mode"].values, df["Serial_Number"].values))


def augmented(long_df, df_part, tr_idx, rs):
    tr = df_part.iloc[tr_idx]
    ids = set(tr["Experiment_Index"].tolist())
    aug = augment_training_set(long_df, ids, target_per_class=160, noise_scale=0.01, random_state=rs)
    return pd.concat([tr, extract_features(aug)], ignore_index=True) if len(aug) else tr


def weights(y, f05_boost=None, sev3_boost=None):
    sw = compute_sample_weights(y, "freq_sev")
    if f05_boost is not None:
        sw[y == "F05"] *= f05_boost
    if sev3_boost is not None:
        sw[np.isin(y, ["F03", "F06"])] *= sev3_boost
    return sw


def suggest(trial, with_sev3):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42, "verbose": -1,
    }
    f05 = trial.suggest_float("f05_boost", 1.0, 10.0)
    sev3 = trial.suggest_float("sev3_boost", 1.0, 20.0) if with_sev3 else None
    return params, f05, sev3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["cmr", "cmr_rel", "f1"], required=True)
    ap.add_argument("--outer", type=int, required=True)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--features", choices=["F246", "F115", "F18"], default="F246")
    args = ap.parse_args()
    n_trials = args.trials or (50 if args.objective == "f1" else 100)
    with_sev3 = args.objective in ("cmr", "cmr_rel")

    train = pd.read_csv(DATA / "train_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    feat_cols = [c for c in train.columns if c not in META_COLS]
    if args.features == "F115":
        feat_cols = [c for c in feat_cols if not (c.startswith("roll5_") or c.startswith("seg"))]
    real18 = None
    if args.features == "F18":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ablation_features import f18_table
        real18 = f18_table(long_df[long_df["Experiment_Index"].isin(set(train["Experiment_Index"]))])

    def real_X(rows):
        if args.features == "F18":
            return real18.reindex(rows["Experiment_Index"].values).values.astype(float)
        return rows[feat_cols].values.astype(float)

    def train_xy(df_part, tr_idx, rs):
        tr = df_part.iloc[tr_idx]
        aug_long = augment_training_set(long_df, set(tr["Experiment_Index"].tolist()),
                                        target_per_class=160, noise_scale=0.01, random_state=rs)
        aug246 = extract_features(aug_long).reset_index(drop=True)
        if args.features == "F18":
            Xa = f18_table(aug_long).reindex(aug246["Experiment_Index"].values).values
        else:
            Xa = aug246[feat_cols].values
        X = np.vstack([real_X(tr), Xa]).astype(float)
        y = np.concatenate([tr["Fault_Mode"].values, aug246["Fault_Mode"].values])
        return X, y

    out_dir = DATA / "nested_cv"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.objective}_outer{args.outer}" if args.features == "F246" else f"{args.objective}_{args.features}_outer{args.outer}"

    otr, ote = folds_for(train, 5)[args.outer]
    outer_train = train.iloc[otr].reset_index(drop=True)
    outer_test = train.iloc[ote].reset_index(drop=True)
    inner = folds_for(outer_train, 5)

    cache = []
    for j, (itr, ite) in enumerate(inner):
        Xt_, yt_ = train_xy(outer_train, itr, 42 + j)
        cache.append((Xt_, yt_, real_X(outer_train.iloc[ite]),
                      outer_train.iloc[ite]["Fault_Mode"].values))

    bal_floor = 0.50
    if args.objective == "cmr_rel":
        bals0 = []
        for Xt, yt, Xv, yv in cache:
            m0 = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": args.threads}).fit(
                Xt, yt, sample_weight=weights(yt))
            bals0.append(evaluate_predictions(yv, m0.predict(Xv))["balanced_accuracy"])
        bal_floor = float(np.mean(bals0)) - 0.037
        print(f"{tag}: bal_acc floor = {bal_floor:.4f}", flush=True)

    def objective(trial):
        params, f05, sev3 = suggest(trial, with_sev3)
        params["n_jobs"] = args.threads
        cmrs, bals, f1s = [], [], []
        for Xt, yt, Xv, yv in cache:
            model = lgb.LGBMClassifier(**params).fit(Xt, yt, sample_weight=weights(yt, f05, sev3))
            yp = model.predict(Xv)
            m = evaluate_predictions(yv, yp)
            cmrs.append(compute_safety_metrics(yv, yp)["cmr"])
            bals.append(m["balanced_accuracy"])
            f1s.append(m["macro_f1"])
        if args.objective == "f1":
            return float(np.mean(f1s))
        trial.set_user_attr("bal_acc", float(np.mean(bals)))
        if np.mean(bals) < bal_floor:
            return 1.0
        return float(np.nanmean([v for v in cmrs if not np.isnan(v)] or [1.0]))

    t0 = time.time()
    study = optuna.create_study(direction="maximize" if args.objective == "f1" else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    inner_time = time.time() - t0
    best = dict(study.best_params)
    f05 = best.pop("f05_boost")
    sev3 = best.pop("sev3_boost", None)
    best.update(random_state=42, verbose=-1, n_jobs=args.threads)

    # outer evaluation: augmented outer-train (random_state=42), predict outer test
    X_tr, y_tr = train_xy(outer_train, np.arange(len(outer_train)), 42)
    X_te = real_X(outer_test)

    arms = {}
    arms["tuned"] = lgb.LGBMClassifier(**best).fit(X_tr, y_tr, sample_weight=weights(y_tr, f05, sev3))
    arms["untuned"] = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": args.threads}).fit(
        X_tr, y_tr, sample_weight=weights(y_tr))
    if args.features == "F246":
        paper_file = "best_lgbm_params.json" if args.objective == "f1" else "best_lgbm_cmr_params.json"
        pp = json.loads((DATA / paper_file).read_text())
        pf05 = pp.pop("f05_boost")
        ps3 = pp.pop("sev3_boost", None)
        arms["paper_params"] = lgb.LGBMClassifier(**{**pp, "n_jobs": args.threads}).fit(
            X_tr, y_tr, sample_weight=weights(y_tr, pf05, ps3))

    preds = outer_test[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
    for name, model in arms.items():
        preds[f"pred_{name}"] = model.predict(X_te)
    preds.to_csv(out_dir / f"preds_{tag}.csv", index=False)
    json.dump({
        "objective": args.objective, "features": args.features, "outer": args.outer, "n_trials": n_trials, "bal_floor": bal_floor,
        "inner_best_value": study.best_value, "inner_seconds": round(inner_time, 1),
        "best_params": {**study.best_params},
        "n_outer_train": len(outer_train), "n_outer_test": len(outer_test),
    }, open(out_dir / f"study_{tag}.json", "w"), indent=2)
    print(f"{tag}: inner best {study.best_value:.4f} in {inner_time / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
