"""Controlled retrospective LightGBM comparison with inner-only policy selection.

New runs use audited source features and separate outputs; historical fits are not
relabelled as results of this implementation. --smoke-test checks execution only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.augmentation import augment_training_set
from pbit.features import extract_features
from pbit.inference import confusion_metrics
from pbit.modeling import META_COLS, FAULT_ORDER, SEVERITY_MAP, compute_sample_weights

CLASSES = np.array(FAULT_ORDER)
SEVERITY = np.array([SEVERITY_MAP[c] for c in CLASSES])
ARMS = [(objective, boost) for objective in ["macro_F1", "CMR"] for boost in ["fixed", "optimized"]]


def metrics(y, prediction):
    lookup = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((8, 8))
    np.add.at(cm, ([lookup[c] for c in y], [lookup[c] for c in prediction]), 1)
    return {k: float(v) for k, v in confusion_metrics(cm).items()}


def grouped_folds(frame, count, seed):
    folds = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(
        np.arange(len(frame)), frame.Fault_Mode, frame.Serial_Number))
    for train, evaluation in folds:
        assert set(frame.iloc[train].Serial_Number).isdisjoint(frame.iloc[evaluation].Serial_Number)
    return folds


def aligned_probability(model, X):
    result = np.zeros((len(X), len(CLASSES)))
    probability = model.predict_proba(pd.DataFrame(X, columns=model.feature_name_))
    for i, label in enumerate(model.classes_):
        result[:, np.flatnonzero(CLASSES == label)[0]] = probability[:, i]
    assert np.allclose(result.sum(axis=1), 1)
    return result


def threshold_predictions(probability, threshold):
    prediction = CLASSES[probability.argmax(axis=1)].copy()
    score = probability[:, SEVERITY == 3].sum(axis=1)
    prediction[score >= threshold] = CLASSES[SEVERITY == 3][probability[score >= threshold][:, SEVERITY == 3].argmax(axis=1)]
    return prediction


def select_threshold(y, probability, budget):
    scores = probability[:, SEVERITY == 3].sum(axis=1)
    candidates = np.r_[np.inf, np.sort(np.unique(scores))[::-1], 0]
    eligible = []
    for threshold in candidates:
        result = metrics(y, threshold_predictions(probability, threshold))
        if np.isfinite(result["CMR"]) and np.isfinite(result["FA0"]) and result["FA0"] <= budget:
            eligible.append((result["CMR"], result["FA0"], -threshold, threshold))
    return (min(eligible)[3], False) if eligible else (np.inf, True)


def train_cache(train, evaluation, long_df, columns, seed):
    augmented = augment_training_set(long_df, set(train.Experiment_Index), random_state=seed)
    expanded = pd.concat([train, extract_features(augmented)], ignore_index=True) if len(augmented) else train
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return (imputer.fit_transform(expanded[columns]), expanded.Fault_Mode.to_numpy(),
            imputer.transform(evaluation[columns]), evaluation.Fault_Mode.to_numpy())


def fit_predict(cache, params, f05=1, s3=1):
    X, y, evaluation, _ = cache
    weights = compute_sample_weights(y, "freq_sev")
    weights[y == "F05"] *= f05
    weights[np.isin(y, ["F03", "F06"])] *= s3
    model = lgb.LGBMClassifier(**params).fit(X, y, sample_weight=weights)
    return aligned_probability(model, evaluation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fa0-budget", type=float, default=0.4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2 or not 0 <= args.fa0_budget <= 1:
        parser.error("Trials must be positive, folds at least two, and FA0 budget in [0,1].")
    if args.smoke_test:
        args.trials, args.folds = 1, 2
    destination = ROOT / "data/restart" / ("smoke" if args.smoke_test else "controlled") / f"seed{args.seed}_outer{args.outer}"
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    source = ROOT / "data/PBIT_features_r1.csv"
    historical = pd.read_csv(ROOT / "data/train_real.csv")
    population = pd.read_csv(source)
    population = population[~population.Experiment_Index.isin([931, 949, 1107])]
    train = population[population.Serial_Number.isin(historical.Serial_Number)].reset_index(drop=True)
    long_df = pd.read_csv(ROOT / "data/PBIT_atuador_DESCARAC_r1_long_clean.csv")
    long_df = long_df[long_df.Experiment_Index.isin(train.Experiment_Index)]
    columns = [c for c in train if c not in META_COLS]
    assert len(train) == 1195 and len(columns) == 246 and train.Experiment_Index.is_unique
    folds = grouped_folds(train, args.folds, args.seed)
    if not 0 <= args.outer < len(folds):
        parser.error("Outer fold index is outside the available folds.")
    tr, te = folds[args.outer]
    outer_train, outer_test = train.iloc[tr].reset_index(drop=True), train.iloc[te].reset_index(drop=True)
    inner_folds = grouped_folds(outer_train, args.folds, args.seed)
    inner_cache = [train_cache(outer_train.iloc[i], outer_train.iloc[j], long_df, columns, args.seed + k)
                   for k, (i, j) in enumerate(inner_folds)]
    inner_y = np.concatenate([c[3] for c in inner_cache])
    assert len(inner_y) == len(outer_train)
    base = dict(n_estimators=20 if args.smoke_test else 400, learning_rate=0.05, num_leaves=63,
                min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=args.seed, verbose=-1, n_jobs=args.threads)
    base_probability = np.concatenate([fit_predict(c, base) for c in inner_cache])
    floor = max(0, metrics(inner_y, CLASSES[base_probability.argmax(axis=1)])["balanced_accuracy"] - 0.037)
    outer_cache = train_cache(outer_train, outer_test, long_df, columns, args.seed)
    predictions = outer_test[META_COLS].copy()
    selections = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for objective, boost_mode in ARMS:
        def objective_function(trial):
            params = {**base,
                      "n_estimators": trial.suggest_int("n_estimators", 10, 20) if args.smoke_test else trial.suggest_int("n_estimators", 200, 800),
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
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective_function, n_trials=args.trials)
        feasible = [t for t in study.trials if t.user_attrs.get("feasible", False)]
        search_failed = not feasible
        if search_failed:
            params, f05, s3, inner_probability = base, 1, 1, base_probability
        else:
            params = {**base, **study.best_params}
            f05, s3 = params.pop("f05_boost"), params.pop("sev3_boost", 1)
            inner_probability = np.concatenate([fit_predict(c, params, f05, s3) for c in inner_cache])
        threshold, budget_failed = select_threshold(inner_y, inner_probability, args.fa0_budget)
        external_probability = fit_predict(outer_cache, params, f05, s3)
        name = f"{objective}_{boost_mode}"
        predictions[f"{name}_base"] = CLASSES[external_probability.argmax(axis=1)]
        predictions[f"{name}_threshold"] = threshold_predictions(external_probability, threshold)
        selections.append({"arm": name, "parameters": params, "f05_boost": f05, "sev3_boost": s3,
                           "search_failed": search_failed, "feasible_trials": len(feasible),
                           "threshold": float(threshold) if np.isfinite(threshold) else "infinity",
                           "budget_failed": budget_failed,
                           "inner_metrics": metrics(inner_y, threshold_predictions(inner_probability, threshold)),
                           "outer_metrics": metrics(outer_cache[3], predictions[f"{name}_threshold"].to_numpy())})
    threshold, budget_failed = select_threshold(inner_y, base_probability, args.fa0_budget)
    external_probability = fit_predict(outer_cache, base)
    predictions["untuned_base"] = CLASSES[external_probability.argmax(axis=1)]
    predictions["untuned_threshold"] = threshold_predictions(external_probability, threshold)
    selections.append({"arm": "untuned", "threshold": float(threshold) if np.isfinite(threshold) else "infinity",
                       "budget_failed": budget_failed, "outer_metrics": metrics(outer_cache[3], predictions.untuned_threshold.to_numpy())})
    destination.mkdir(parents=True)
    predictions.to_csv(destination / "predictions.csv", index=False)
    partition = outer_train[META_COLS].assign(role="outer_train")
    partition = pd.concat([partition, outer_test[META_COLS].assign(role="outer_test")])
    partition.to_csv(destination / "partition.csv", index=False)
    inner_assignment = pd.concat([outer_train.iloc[j][META_COLS].assign(inner_fold=k) for k, (_, j) in enumerate(inner_folds)])
    inner_assignment.to_csv(destination / "inner_assignment.csv", index=False)
    record = {"scope": "execution_check_only" if args.smoke_test else "retrospective_controlled_comparison",
              "arguments": vars(args), "balanced_accuracy_floor": floor, "selections": selections,
              "checks": {"disjoint_unit_partitions": True, "pooled_inner_selection": True, "shared_feasibility_and_budget": True},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source, ROOT / "data/train_real.csv", ROOT / "data/PBIT_atuador_DESCARAC_r1_long_clean.csv",
                                          Path(__file__), ROOT / "src/pbit/inference.py", ROOT / "src/pbit/features.py",
                                          ROOT / "src/pbit/augmentation.py", ROOT / "src/pbit/modeling.py"]}}
    (destination / "run_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(destination), "scope": record["scope"], "checks": record["checks"]}, indent=2))


if __name__ == "__main__":
    main()
