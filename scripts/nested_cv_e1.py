"""E1 full-budget timing pilot; outer performance is deliberately not calculated."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.augmentation import augment_training_set
from pbit.cost_policies import cost_matrix, minimum_cost_predictions, select_cost_policy
from pbit.features import extract_features
from pbit.modeling import META_COLS, compute_sample_weights
from nested_cv_controlled import (CLASSES, ARMS, aligned_probability, grouped_folds,
                                 metrics, select_threshold, threshold_predictions)

FROZEN = ROOT / "data/ress_revision/freeze_v1"
CONFIG = dict(seed=42, outer=0, trials=100, inner_folds=5, threads=4,
              budgets=[.3, .4, .5], kappas=[2, 3, 5, 10, 20], bags=10,
              primary_budget=.4, scope="full_budget_timing_pilot_no_outer_performance")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialisable(value):
    if isinstance(value, dict):
        return {k: serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialisable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "infinity" if value == np.inf else None
    return value


def write_json(path, record):
    Path(path).write_text(json.dumps(serialisable(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def base_params(cfg):
    return dict(n_estimators=20 if cfg.get("smoke") else 400, learning_rate=.05,
                num_leaves=63, min_child_samples=5, subsample=.8, subsample_freq=1,
                colsample_bytree=.8, random_state=cfg["seed"], n_jobs=cfg["threads"], verbose=-1)


def source_paths():
    return [ROOT / "docs/protocolos/E1.md", Path(__file__), ROOT / "scripts/nested_cv_controlled.py",
            ROOT / "scripts/verify_e1_pilot.py", ROOT / "tests/test_e1_cost_policies.py",
            ROOT / "src/pbit/cost_policies.py", ROOT / "src/pbit/augmentation.py",
            ROOT / "src/pbit/features.py", ROOT / "src/pbit/modeling.py", ROOT / "src/pbit/inference.py",
            FROZEN / "freeze_manifest.json", FROZEN / "train_primary.csv", FROZEN / "long_primary.csv",
            FROZEN / "outer_assignment.csv", FROZEN / "inner_assignment.csv"]


def freeze(out, cfg):
    path = out / "protocol_freeze.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert existing["config"] == cfg
        for name, expected in existing["source_sha256"].items():
            assert sha(ROOT / name) == expected, name
        return
    assert not out.exists(), "Use a new output version."
    old = json.loads((FROZEN / "freeze_manifest.json").read_text(encoding="utf-8"))
    for p in source_paths():
        if p.name in ["train_primary.csv", "long_primary.csv", "outer_assignment.csv", "inner_assignment.csv"]:
            assert sha(p) == old["output_sha256"][p.relative_to(ROOT).as_posix()]
    out.mkdir(parents=True)
    for p in source_paths():
        if p.parent != FROZEN:
            snapshot = out / "frozen_sources" / p.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(p.read_bytes())
    write_json(path, dict(config=cfg, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths()},
        environment={p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "optuna"]}, python=sys.version))
    print("Protocol and sources frozen before training.", flush=True)


def prepare(out, cfg):
    population = pd.read_csv(FROZEN / "train_primary.csv")
    long = pd.read_csv(FROZEN / "long_primary.csv")
    columns = [c for c in population if c not in META_COLS]
    assert len(population) == 1195 and population.Serial_Number.nunique() == 54 and len(columns) == 246
    if cfg.get("smoke"):
        tr, te = grouped_folds(population, cfg["inner_folds"], cfg["seed"])[cfg["outer"]]
        train, test = population.iloc[tr].reset_index(drop=True), population.iloc[te].reset_index(drop=True)
        folds = grouped_folds(train, cfg["inner_folds"], cfg["seed"])
    else:
        outer = pd.read_csv(FROZEN / "outer_assignment.csv")
        outer = outer[outer.seed.eq(cfg["seed"])]
        test_ids = set(outer.loc[outer.outer_fold.eq(cfg["outer"]), "Experiment_Index"])
        train = population[~population.Experiment_Index.isin(test_ids)].reset_index(drop=True)
        test = population[population.Experiment_Index.isin(test_ids)].reset_index(drop=True)
        assignment = pd.read_csv(FROZEN / "inner_assignment.csv")
        assignment = assignment[assignment.seed.eq(cfg["seed"]) & assignment.outer_fold.eq(cfg["outer"])]
        assert set(assignment.Experiment_Index) == set(train.Experiment_Index)
        folds = []
        for k in range(5):
            held = train.Experiment_Index.isin(assignment.loc[assignment.inner_fold.eq(k), "Experiment_Index"])
            folds.append((np.flatnonzero(~held), np.flatnonzero(held)))
    assert set(train.Serial_Number).isdisjoint(test.Serial_Number)
    pd.concat([train[META_COLS].assign(role="outer_train"), test[META_COLS].assign(role="outer_test")]).to_csv(
        out / "partition.csv", index=False)
    cache_dir = out / "cache"
    cache_dir.mkdir()
    records, assignments = [], []
    start = time.monotonic()
    parts = [(f"inner{k}", train.iloc[i], train.iloc[j], cfg["seed"]+k)
             for k, (i, j) in enumerate(folds)] + [("outer", train, test, cfg["seed"])]
    for name, fitting, evaluation, seed in parts:
        assert set(fitting.Serial_Number).isdisjoint(evaluation.Serial_Number)
        assert set(fitting.Serial_Number).isdisjoint(test.Serial_Number)
        aug_long = augment_training_set(long, set(fitting.Experiment_Index), random_state=seed)
        assert set(aug_long.Serial_Number) <= set(fitting.Serial_Number)
        aug = extract_features(aug_long) if len(aug_long) else fitting.iloc[:0]
        expanded = pd.concat([fitting, aug], ignore_index=True)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X = imputer.fit_transform(expanded[columns])
        X_eval = imputer.transform(evaluation[columns])
        np.savez_compressed(cache_dir / f"{name}.npz", X=X, y=expanded.Fault_Mode.to_numpy(dtype=str),
                            X_eval=X_eval, y_eval=evaluation.Fault_Mode.to_numpy(dtype=str))
        expanded[META_COLS].to_csv(cache_dir / f"{name}_training_metadata.csv", index=False)
        evaluation[META_COLS].to_csv(cache_dir / f"{name}_evaluation_metadata.csv", index=False)
        pd.DataFrame(dict(feature=columns, statistic=imputer.statistics_)).to_csv(
            cache_dir / f"{name}_imputation.csv", index=False)
        records.append(dict(cache=name, real_cycles=len(fitting), synthetic_cycles=len(aug),
                            training_units=fitting.Serial_Number.nunique(), evaluation_cycles=len(evaluation),
                            training_classes=sorted(fitting.Fault_Mode.unique())))
        if name != "outer":
            assignments.append(evaluation[META_COLS].assign(inner_fold=int(name[5:])))
    pd.concat(assignments, ignore_index=True).to_csv(out / "inner_assignment.csv", index=False)
    write_json(out / "cache_manifest.json", dict(preparation_seconds=time.monotonic()-start, caches=records,
        output_sha256={str(p.relative_to(out)): sha(p) for p in sorted(cache_dir.iterdir())}))
    return [str(cache_dir / f"inner{k}.npz") for k in range(len(folds))]


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def fit_probability(cache, params, f05=1, s3=1, labels=None, save=None):
    y = cache["y"] if labels is None else labels
    weights = compute_sample_weights(y, "freq_sev")
    weights[y == "F05"] *= f05
    weights[np.isin(y, ["F03", "F06"])] *= s3
    model = lgb.LGBMClassifier(**params).fit(cache["X"], y, sample_weight=weights)
    if save:
        model.booster_.save_model(str(save))
    return aligned_probability(model, cache["X_eval"])


def search_arm(task):
    out_name, cfg, cache_paths, objective, boost_mode, floor = task
    out = Path(out_name) / "search" / f"{objective}_{boost_mode}"
    out.mkdir(parents=True)
    caches = [load_cache(p) for p in cache_paths]
    y = np.concatenate([c["y_eval"] for c in caches])
    base = base_params(cfg)
    rows, best_value, started = [], None, time.monotonic()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def evaluate_trial(trial):
        nonlocal best_value
        params = {**base, "n_estimators": trial.suggest_int("n_estimators", 10, 20) if cfg.get("smoke") else trial.suggest_int("n_estimators", 200, 800),
                  "learning_rate": trial.suggest_float("learning_rate", .01, .2, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                  "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
                  "subsample": trial.suggest_float("subsample", .5, 1),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1),
                  "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                  "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True)}
        f05 = trial.suggest_float("f05_boost", 1, 10)
        s3 = trial.suggest_float("sev3_boost", 1, 20) if boost_mode == "optimized" else 1
        t0 = time.monotonic()
        probability = np.concatenate([fit_probability(c, params, f05, s3) for c in caches])
        result = metrics(y, CLASSES[probability.argmax(axis=1)])
        feasible = result["balanced_accuracy"] >= floor and np.isfinite(result[objective])
        value = result[objective] if feasible else (-1 if objective == "macro_F1" else 2)
        trial.set_user_attr("feasible", bool(feasible))
        rows.append(dict(trial=trial.number, feasible=bool(feasible), value=value,
                         fit_seconds=time.monotonic()-t0, **result, parameters=json.dumps(trial.params)))
        pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)
        better = best_value is None or (value > best_value if objective == "macro_F1" else value < best_value)
        if feasible and better:
            best_value = value
            np.save(out / "best_probability.npy", probability)
            write_json(out / "best.json", dict(trial=trial.number, params=params, f05=f05, s3=s3, metrics=result))
        if (trial.number + 1) % 5 == 0 or trial.number + 1 == cfg["trials"]:
            print(f"{objective}/{boost_mode}: {trial.number+1}/{cfg['trials']} trials, {time.monotonic()-started:.0f}s", flush=True)
        return value

    study = optuna.create_study(direction="maximize" if objective == "macro_F1" else "minimize",
                               sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(evaluate_trial, n_trials=cfg["trials"])
    write_json(out / "manifest.json", dict(trials=len(rows), feasible_trials=sum(r["feasible"] for r in rows),
        search_fits=len(rows)*len(caches), runtime_seconds=time.monotonic()-started,
        best_trial=None if best_value is None else json.loads((out / "best.json").read_text())["trial"]))
    return f"{objective}_{boost_mode}"


def metacost_posterior(cache, cfg, index):
    weights = compute_sample_weights(cache["y"], "freq_sev")
    rng = np.random.default_rng(cfg["seed"] + 10000 + index)
    posterior = np.zeros((len(cache["X"]), len(CLASSES)))
    indices = []
    for bag in range(cfg["bags"]):
        ix = rng.integers(0, len(weights), len(weights))
        params = {**base_params(cfg), "n_estimators": 10 if cfg.get("smoke") else 200,
                  "random_state": cfg["seed"] + 100 + bag}
        model = lgb.LGBMClassifier(**params).fit(cache["X"][ix], cache["y"][ix], sample_weight=weights[ix])
        posterior += aligned_probability(model, cache["X"])
        indices.append(ix)
    return posterior/cfg["bags"], np.stack(indices)


def complete_policies(out, cfg, cache_paths, base_inner, inner_y):
    outer = load_cache(out / "cache/outer.npz")
    metadata = pd.read_csv(out / "cache/outer_evaluation_metadata.csv")
    predictions = metadata.copy()
    internal_meta = pd.concat([pd.read_csv(Path(p).with_name(Path(p).stem + "_evaluation_metadata.csv"))
                               for p in cache_paths], ignore_index=True)
    decisions, counts = [], []
    models = out / "models"
    models.mkdir()
    base_external = fit_probability(outer, base_params(cfg), save=models / "untuned.txt")
    predictions["untuned_base"] = CLASSES[base_external.argmax(axis=1)]
    np.save(out / "untuned_external_probability.npy", base_external)
    for name in ["untuned"] + [f"{o}_{b}" for o, b in ARMS]:
        if name == "untuned":
            inside, outside, best, failed = base_inner, base_external, dict(params=base_params(cfg), f05=1, s3=1), False
        else:
            directory = out / "search" / name
            failed = not (directory / "best.json").exists()
            best = dict(params=base_params(cfg), f05=1, s3=1) if failed else json.loads((directory / "best.json").read_text())
            inside = base_inner if failed else np.load(directory / "best_probability.npy")
            outside = fit_probability(outer, best["params"], best["f05"], best["s3"], save=models / f"{name}.txt")
            np.save(out / f"{name}_external_probability.npy", outside)
            predictions[f"{name}_base"] = CLASSES[outside.argmax(axis=1)]
        for budget in cfg["budgets"]:
            threshold, failed_budget = select_threshold(inner_y, inside, budget)
            tag = f"{name}_b{int(budget*100)}"
            predictions[tag] = threshold_predictions(outside, threshold)
            internal_meta[tag] = threshold_predictions(inside, threshold)
            decisions.append(dict(arm=name, budget=budget, threshold=threshold, budget_failed=failed_budget,
                                  search_failed=failed, **best, inner_metrics=metrics(inner_y, internal_meta[tag].to_numpy())))
    mec = {k: minimum_cost_predictions(base_inner, k) for k in cfg["kappas"]}
    mc_parts, relabel_rows = {k: [] for k in cfg["kappas"]}, []
    cost_start = time.monotonic()
    for index, path in enumerate(cache_paths):
        cache = load_cache(path)
        posterior, ix = metacost_posterior(cache, cfg, index)
        np.save(out / f"metacost_inner{index}_posterior.npy", posterior)
        np.save(out / f"metacost_inner{index}_bag_indices.npy", ix)
        for k in cfg["kappas"]:
            labels = minimum_cost_predictions(posterior, k)
            relabel_rows.append(dict(inner_fold=index, kappa=k, relabelled_fraction=float((labels != cache["y"]).mean())))
            probability = fit_probability(cache, base_params(cfg), labels=labels)
            mc_parts[k].append(probability)
        print(f"MetaCost: inner fold {index+1}/{len(cache_paths)} done.", flush=True)
    mc_prob = {k: np.concatenate(parts) for k, parts in mc_parts.items()}
    mc = {k: CLASSES[p.argmax(axis=1)] for k, p in mc_prob.items()}
    for k, p in mc_prob.items():
        np.save(out / f"metacost_k{k}_inner_probability.npy", p)
    selected = []
    selections = {}
    for name, candidates in [("minimum_cost", mec), ("metacost", mc)]:
        for budget in cfg["budgets"]:
            k, rows = select_cost_policy(inner_y, candidates, budget, metrics)
            counts.extend(dict(arm=name, **row) for row in rows)
            selections[name, budget] = k
            if name == "metacost" and k is not None:
                selected.append(k)
    outer_mc, outer_relabel = {}, []
    if selected:
        posterior, ix = metacost_posterior(outer, cfg, 5)
        np.save(out / "metacost_outer_posterior.npy", posterior)
        np.save(out / "metacost_outer_bag_indices.npy", ix)
        for k in sorted(set(selected)):
            labels = minimum_cost_predictions(posterior, k)
            outer_relabel.append(dict(kappa=k, relabelled_fraction=float((labels != outer["y"]).mean())))
            outer_mc[k] = fit_probability(outer, base_params(cfg), labels=labels, save=models / f"metacost_k{k}.txt")
            np.save(out / f"metacost_k{k}_external_probability.npy", outer_mc[k])
    for (name, budget), k in selections.items():
        tag = f"{name}_b{int(budget*100)}"
        inside = CLASSES[base_inner.argmax(axis=1)] if k is None else (mec if name == "minimum_cost" else mc)[k]
        outside = predictions.untuned_base.to_numpy() if k is None else (
            minimum_cost_predictions(base_external, k) if name == "minimum_cost" else CLASSES[outer_mc[k].argmax(axis=1)])
        predictions[tag], internal_meta[tag] = outside, inside
        decisions.append(dict(arm=name, budget=budget, kappa=k, budget_failed=k is None,
                              inner_metrics=metrics(inner_y, inside)))
    predictions.to_csv(out / "external_predictions.csv", index=False)
    internal_meta.to_csv(out / "inner_predictions.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "cost_candidates.csv", index=False)
    pd.DataFrame(relabel_rows).to_csv(out / "relabelled_fractions.csv", index=False)
    write_json(out / "decisions.json", decisions)
    write_json(out / "cost_timing.json", dict(seconds=time.monotonic()-cost_start,
        bag_fits=len(cache_paths)*cfg["bags"]+(cfg["bags"] if selected else 0),
        relabelled_fits=len(cache_paths)*len(cfg["kappas"])+len(set(selected)), outer_relabelling=outer_relabel))


def run(out, cfg, workers):
    freeze(out, cfg)
    assert not (out / "run_manifest.json").exists(), "Completed run exists."
    assert not (out / "cache").exists(), "Interrupted run: preserve it and use a new version."
    start = time.monotonic()
    paths = prepare(out, cfg)
    caches = [load_cache(p) for p in paths]
    inner_y = np.concatenate([c["y_eval"] for c in caches])
    baseline_start = time.monotonic()
    base_inner = np.concatenate([fit_probability(c, base_params(cfg)) for c in caches])
    np.save(out / "untuned_inner_probability.npy", base_inner)
    floor = max(0, metrics(inner_y, CLASSES[base_inner.argmax(axis=1)])["balanced_accuracy"]-.037)
    write_json(out / "baseline.json", dict(seconds=time.monotonic()-baseline_start, balanced_accuracy_floor=floor))
    search_start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search_arm, (str(out), cfg, paths, objective, boost, floor)) for objective, boost in ARMS]
        for future in as_completed(futures):
            print("Search completed: " + future.result(), flush=True)
    search_seconds = time.monotonic()-search_start
    policy_start = time.monotonic()
    complete_policies(out, cfg, paths, base_inner, inner_y)
    policy_seconds = time.monotonic()-policy_start
    costs = json.loads((out / "cost_timing.json").read_text())
    fits = 4*cfg["trials"]*len(paths)+len(paths)+5+costs["bag_fits"]+costs["relabelled_fits"]
    freeze(out, cfg)
    write_json(out / "run_manifest.json", dict(config=cfg, protocol_freeze_sha256=sha(out / "protocol_freeze.json"),
        runtime_seconds=time.monotonic()-start, search_seconds=search_seconds, policy_seconds=policy_seconds,
        model_fits=fits, outer_performance_calculated=False,
        output_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*"))
                       if p.is_file() and p.name not in ["run_manifest.json", "protocol_freeze.json"]}))
    print(f"Pilot finished: {fits} fits; {(time.monotonic()-start)/60:.1f} minutes. Outer performance not calculated.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "run"], default="run")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="data/e1_pilot")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.smoke:
        cfg.update(smoke=True, trials=1, inner_folds=2, bags=2, threads=2,
                   budgets=[.3, .4, .5, 1.0], scope="execution_check_only")
    destination = ROOT / args.output
    if args.mode == "prepare":
        freeze(destination, cfg)
    else:
        run(destination, cfg, args.workers)
