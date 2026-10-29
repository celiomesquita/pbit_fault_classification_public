"""E8: SMOTEWB-LGBM (Li et al., 2024) under serial-number-grouped and random partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn
import smote_variants as sv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.getLogger("smote_variants").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from imbalance_ablation import UNTUNED, WEIGHTS  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "e8_smotewb"
PROTOCOL = ROOT / "docs/protocolos/E8.md"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
STRATEGY = "eq_1_vs_many_successive"
LI_TABLE4 = dict(n_iters=96, max_depth=16, lambda_l1=0.1049, lambda_l2=0.2042, num_leaves=17,
                 feature_fraction=0.5204, bagging_fraction=0.9395, learning_rate=0.0979)
TRIALS = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def outer_folds(train, seed, partition):
    rows, y, groups = np.arange(len(train)), train["Fault_Mode"].values, train["Serial_Number"].values
    if partition == "grouped":
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(rows, y, groups))
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rows, y))


def inner_folds(train, seed, partition):
    return outer_folds(train, seed, partition)


def oversample(name, X, y, params, random_state):
    sampler = sv.MulticlassOversampling(oversampler=name, oversampler_params={**params, "random_state": random_state},
                                        strategy=STRATEGY)
    return sampler.sample(X, y)


def resampling_log(X, y, X_out, y_out):
    """Counts before/after per class and share of generated rows identical to a real row."""
    record = {}
    generated, labels = X_out[len(X):], y_out[len(X):]
    for label in sorted(set(y_out)):
        real = X[y == label]
        made = generated[labels == label]
        copies = sum(bool(np.all(real == row, axis=1).any()) for row in made) if len(made) else 0
        record[str(label)] = dict(before=int(len(real)), after=int(np.sum(y_out == label)),
                                  generated=int(len(made)), exact_copies=int(copies))
    return record


def li_lgbm(params, seed, threads):
    return lgb.LGBMClassifier(n_estimators=100, reg_alpha=params["lambda_l1"], reg_lambda=params["lambda_l2"],
                              num_leaves=params["num_leaves"], colsample_bytree=params["feature_fraction"],
                              subsample=params["bagging_fraction"], learning_rate=params["learning_rate"],
                              random_state=seed, verbose=-1, n_jobs=threads)


def li_pipeline(X_train, y_train, X_test, params, seed, threads):
    """Median imputation and Min-Max fitted on real training rows, SMOTEWB, LightGBM."""
    imputer = SimpleImputer(strategy="median").fit(X_train)
    scaler = MinMaxScaler().fit(imputer.transform(X_train))
    X_tr = scaler.transform(imputer.transform(X_train))
    X_te = scaler.transform(imputer.transform(X_test))
    X_os, y_os = oversample("SMOTEWB", X_tr, y_train,
                            dict(n_iters=params["n_iters"], max_depth=params["max_depth"]), seed)
    model = li_lgbm(params, seed, threads).fit(X_os, y_os)
    return model.predict(X_te), resampling_log(X_tr, y_train, X_os, y_os)


def tune(train_part, X, y, seed, partition, threads, trials):
    folds = inner_folds(train_part, seed, partition)

    def objective(trial):
        params = dict(n_iters=trial.suggest_int("n_iters", 30, 100),
                      max_depth=trial.suggest_int("max_depth", 5, 30),
                      lambda_l1=trial.suggest_float("lambda_l1", 0, 10),
                      lambda_l2=trial.suggest_float("lambda_l2", 0, 10),
                      num_leaves=trial.suggest_int("num_leaves", 3, 128),
                      feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1),
                      bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1),
                      learning_rate=trial.suggest_float("learning_rate", 0.001, 1))
        scores = []
        for tr, va in folds:
            predicted, _ = li_pipeline(X[tr], y[tr], X[va], params, seed, threads)
            scores.append(accuracy_score(y[va], predicted))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return study


def run(seed, partition, threads, trials, folds_to_run, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train_real.csv")
    cols = [c for c in train.columns if c not in META_COLS]
    parts, logs, started = [], [], time.perf_counter()
    for k, (tr_idx, te_idx) in enumerate(outer_folds(train, seed, partition)):
        if folds_to_run is not None and k not in folds_to_run:
            continue
        fold_started = time.perf_counter()
        tr, te = train.iloc[tr_idx], train.iloc[te_idx]
        y = tr["Fault_Mode"].values
        X, Xt = tr[cols].values.astype(float), te[cols].values.astype(float)
        out = te[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        out.insert(0, "fold", k)
        out.insert(0, "seed", seed)
        log = dict(seed=seed, partition=partition, fold=k, train_counts=pd.Series(y).value_counts().to_dict())
        suffix = "" if partition == "grouped" else "_random"
        if partition == "grouped":
            imputer = SimpleImputer(strategy="median").fit(X)
            X_real = imputer.transform(X)
            for name, label in [("SMOTEWB", "smotewb"), ("SMOTE", "smote_bal")]:
                X_os, y_os = oversample(name, X_real, y, {}, seed + k)
                log[label] = resampling_log(X_real, y, X_os, y_os)
                X_te = SimpleImputer(strategy="median").fit(X_os).transform(Xt)
                for w in WEIGHTS:
                    weights = compute_sample_weights(y_os, w)
                    model = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": threads}).fit(X_os, y_os, sample_weight=weights)
                    out[f"lgbm_{w}_{label}"] = model.predict(X_te)
                    forest = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                                    random_state=42, n_jobs=threads)
                    out[f"rf_{w}_{label}"] = forest.fit(X_os, y_os, sample_weight=weights).predict(X_te)
        predicted, log["li_fixed"] = li_pipeline(X, y, Xt, LI_TABLE4, seed, threads)
        out[f"li_fixed{suffix}"] = predicted
        study = tune(tr.reset_index(drop=True), X, y, seed, partition, threads, trials)
        best = {**study.best_params}
        predicted, log["li_tuned"] = li_pipeline(X, y, Xt, best, seed, threads)
        out[f"li_tuned_{partition}"] = predicted
        log.update(li_tuned_best=best, li_tuned_inner_micro_f1=study.best_value,
                   seconds=round(time.perf_counter() - fold_started, 1))
        study.trials_dataframe().to_csv(out_dir / f"trials_seed{seed}_fold{k}.csv", index=False)
        parts.append(out)
        logs.append(log)
        print(json.dumps(dict(seed=seed, partition=partition, fold=k, seconds=log["seconds"])), flush=True)
    predictions = out_dir / f"preds_seed{seed}.csv"
    pd.concat(parts).to_csv(predictions, index=False)
    log_path = out_dir / f"resampling_seed{seed}.json"
    log_path.write_text(json.dumps(logs, indent=2, default=str), encoding="utf-8")
    inputs = [PROTOCOL, OUT / "protocol_freeze.json", Path(__file__), ROOT / "scripts/imbalance_ablation.py",
              ROOT / "src/pbit/modeling.py", DATA / "train_real.csv", ROOT / "tests/test_e8.py"]
    outputs = [predictions, log_path, *sorted(out_dir.glob(f"trials_seed{seed}_fold*.csv"))]
    manifest = dict(item="E8", seed=seed, partition=partition, trials=trials, folds=sorted(int(p["fold"].iloc[0]) for p in parts),
                    finished_utc=utc(), seconds=round(time.perf_counter() - started, 1), threads=threads,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, lightgbm=lgb.__version__, optuna=optuna.__version__,
                    smote_variants=sv.__version__,
                    input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    output_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs})
    (out_dir / f"manifest_seed{seed}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partition", choices=["grouped", "random"], required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    freeze = json.loads((OUT / "protocol_freeze.json").read_text(encoding="utf-8-sig"))
    assert freeze["protocol_sha256"] == sha(PROTOCOL), "E8 protocol changed after freezing"
    assert sv.__version__ == freeze["smote_variants"]
    if args.pilot:
        run(args.seed, args.partition, args.threads, 3, {0}, OUT / "pilot" / args.partition)
    else:
        run(args.seed, args.partition, args.threads, TRIALS, None, OUT / args.partition)


if __name__ == "__main__":
    main()
