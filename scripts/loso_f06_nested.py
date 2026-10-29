"""E5: fully nested leave-one-unit-out for the F06 units (docs/protocolos/E5.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nested_cv_controlled import CLASSES, fit_predict, grouped_folds, metrics, train_cache  # noqa: E402
from pbit.modeling import META_COLS  # noqa: E402

OUT = ROOT / "data/e5_nested_loso"
TARGETS = ["SN 045", "SN 052", "SN 054", "SN 056"]
SEEDS = [42, 0, 7, 13, 21]
ARMS = [("F1_tuned", "macro_F1", "fixed"), ("CMR_tuned", "CMR", "optimized")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_population():
    audit = pd.read_csv(ROOT/"data/audit_dataset_r1.csv")
    keep = set(audit.loc[audit["in_D_primary"], "Experiment_Index"])
    features = pd.read_csv(ROOT/"data/PBIT_features_r1.csv")
    features = features[features.Experiment_Index.isin(keep)].reset_index(drop=True)
    assert len(features) == 1487 and features.Serial_Number.nunique() == 65
    long_df = pd.read_csv(ROOT/"data/PBIT_atuador_DESCARAC_r1_long_clean.csv")
    long_df = long_df[long_df.Experiment_Index.isin(keep)]
    return features, long_df


def counts(frame):
    per_class = frame.groupby("Fault_Mode").Serial_Number.nunique()
    return dict(F06_units=int(per_class.get("F06", 0)), F03_units=int(per_class.get("F03", 0)),
                F06_cycles=int((frame.Fault_Mode == "F06").sum()), cycles=len(frame))


def run(target, seed, trials, folds, threads, destination, smoke):
    features, long_df = load_population()
    columns = [c for c in features if c not in META_COLS]
    assert len(columns) == 246
    train = features[features.Serial_Number != target].reset_index(drop=True)
    test = features[features.Serial_Number == target].reset_index(drop=True)
    assert len(test) > 0 and train.Serial_Number.nunique() == 64 and target not in set(train.Serial_Number)
    long_train = long_df[long_df.Experiment_Index.isin(train.Experiment_Index)]
    assert not set(long_train.Experiment_Index) & set(test.Experiment_Index)
    inner_folds = grouped_folds(train, folds, seed)
    inner_cache, inner_counts = [], []
    for k, (i, j) in enumerate(inner_folds):
        inner_train, inner_eval = train.iloc[i], train.iloc[j]
        assert target not in set(inner_train.Serial_Number) | set(inner_eval.Serial_Number)
        inner_cache.append(train_cache(inner_train, inner_eval, long_train, columns, seed+k))
        inner_counts.append(dict(fold=k, train=counts(inner_train), evaluation=counts(inner_eval)))
    inner_y = np.concatenate([c[3] for c in inner_cache])
    assert len(inner_y) == len(train)
    base = dict(n_estimators=20 if smoke else 400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                n_jobs=threads)
    base_probability = np.concatenate([fit_predict(c, base) for c in inner_cache])
    floor = max(0, metrics(inner_y, CLASSES[base_probability.argmax(axis=1)])["balanced_accuracy"]-0.037)
    final_cache = train_cache(train, test, long_train, columns, seed)
    predictions = test[META_COLS].copy()
    selections = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name, objective, boost_mode in ARMS:
        def evaluate(trial):
            params = {**base,
                      "n_estimators": trial.suggest_int("n_estimators", 10, 20) if smoke
                      else trial.suggest_int("n_estimators", 200, 800),
                      "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                      "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                      "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
                      "subsample": trial.suggest_float("subsample", 0.5, 1),
                      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                      "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                      "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True)}
            f05 = trial.suggest_float("f05_boost", 1, 10)
            s3 = trial.suggest_float("sev3_boost", 1, 20) if boost_mode == "optimized" else 1
            probability = np.concatenate([fit_predict(c, params, f05, s3) for c in inner_cache])
            result = metrics(inner_y, CLASSES[probability.argmax(axis=1)])
            feasible = result["balanced_accuracy"] >= floor and np.isfinite(result[objective])
            trial.set_user_attr("feasible", bool(feasible))
            return result[objective] if feasible else (-1 if objective == "macro_F1" else 2)

        study = optuna.create_study(direction="maximize" if objective == "macro_F1" else "minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(evaluate, n_trials=trials)
        feasible = [t for t in study.trials if t.user_attrs.get("feasible", False)]
        failed = not feasible
        if failed:
            params, f05, s3 = base, 1, 1
        else:
            params = {**base, **study.best_params}
            f05, s3 = params.pop("f05_boost"), params.pop("sev3_boost", 1)
        probability = fit_predict(final_cache, params, f05, s3)
        predictions[name] = CLASSES[probability.argmax(axis=1)]
        selections.append(dict(arm=name, search_failed=failed, feasible_trials=len(feasible), f05_boost=f05,
                               sev3_boost=s3, parameters=params,
                               best_inner_value=None if failed else float(study.best_value)))
    probability = fit_predict(final_cache, base)
    predictions["untuned"] = CLASSES[probability.argmax(axis=1)]
    destination.mkdir(parents=True)
    predictions.to_csv(destination/"predictions.csv", index=False)
    record = dict(item="E5", target=target, seed=seed, trials=trials, folds=folds, smoke=smoke,
                  balanced_accuracy_floor=floor, selections=selections, training=counts(train),
                  inner=inner_counts, checks=dict(target_absent_from_all_training_and_selection=True),
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 [ROOT/"data/audit_dataset_r1.csv", ROOT/"data/PBIT_features_r1.csv",
                                  ROOT/"data/PBIT_atuador_DESCARAC_r1_long_clean.csv", Path(__file__),
                                  ROOT/"scripts/nested_cv_controlled.py", ROOT/"docs/protocolos/E5.md"]})
    (destination/"run_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.smoke_test:
        args.trials, args.folds = 1, 2
    tag = args.tag or f"{args.target.replace(' ', '')}_seed{args.seed}"
    destination = OUT/("smoke" if args.smoke_test else "runs")/tag
    if destination.exists():
        raise SystemExit(f"Output exists: {destination}")
    record = run(args.target, args.seed, args.trials, args.folds, args.threads, destination, args.smoke_test)
    print(json.dumps(dict(output=str(destination), training=record["training"]), indent=1))


if __name__ == "__main__":
    main()
