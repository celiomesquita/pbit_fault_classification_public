"""Audit E1 selection and fit provenance without evaluating outer performance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.modeling import FAULT_ORDER, SEVERITY_MAP
from nested_cv_controlled import metrics, select_threshold, threshold_predictions
from nested_cv_e1 import load_cache

CLASSES = np.array(FAULT_ORDER)
SEVERITY = np.array([SEVERITY_MAP[c] for c in CLASSES])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_cost_prediction(probability, kappa):
    cost = np.zeros((8, 8))
    for truth in range(8):
        for decision in range(8):
            d = SEVERITY[truth] - SEVERITY[decision]
            cost[truth, decision] = 0 if truth == decision else kappa*d if d > 0 else -d if d < 0 else 1
    return CLASSES[(probability @ cost).argmin(axis=1)]


def main(out):
    manifest = json.loads((out / "run_manifest.json").read_text())
    cfg = manifest["config"]
    frozen = json.loads((out / "protocol_freeze.json").read_text())
    assert manifest["protocol_freeze_sha256"] == sha(out / "protocol_freeze.json")
    assert manifest["outer_performance_calculated"] is False
    for name, expected in manifest["output_sha256"].items():
        assert sha(out / name) == expected, name
    for name, expected in frozen["source_sha256"].items():
        snapshot = out / "frozen_sources" / name
        assert sha(snapshot if snapshot.exists() else ROOT / name) == expected, name
    partitions = pd.read_csv(out / "partition.csv")
    train_units = set(partitions.loc[partitions.role.eq("outer_train"), "Serial_Number"])
    test_units = set(partitions.loc[partitions.role.eq("outer_test"), "Serial_Number"])
    assert train_units.isdisjoint(test_units)
    observed_ids = []
    for index in range(cfg["inner_folds"]):
        train = pd.read_csv(out / f"cache/inner{index}_training_metadata.csv")
        evaluation = pd.read_csv(out / f"cache/inner{index}_evaluation_metadata.csv")
        assert set(train.Serial_Number) <= train_units
        assert set(train.Serial_Number).isdisjoint(evaluation.Serial_Number)
        assert set(evaluation.Serial_Number).isdisjoint(test_units)
        observed_ids.extend(evaluation.Experiment_Index)
        cache = load_cache(out / f"cache/inner{index}.npz")
        assert len(train) == len(cache["X"]) and len(evaluation) == len(cache["X_eval"])
        assert np.array_equal(cache["y"], train.Fault_Mode)
        assert np.array_equal(cache["y_eval"], evaluation.Fault_Mode)
        bags = np.load(out / f"metacost_inner{index}_bag_indices.npy")
        assert bags.shape == (cfg["bags"], len(train))
        assert bags.min() >= 0 and bags.max() < len(train)
        posterior = np.load(out / f"metacost_inner{index}_posterior.npy")
        np.testing.assert_allclose(posterior.sum(axis=1), 1, atol=1e-12)
    assert len(observed_ids) == len(set(observed_ids))
    assert set(observed_ids) == set(partitions.loc[partitions.role.eq("outer_train"), "Experiment_Index"])
    inside = pd.read_csv(out / "inner_predictions.csv")
    outside = pd.read_csv(out / "external_predictions.csv")
    assert inside.Experiment_Index.tolist() == observed_ids
    assert set(outside.Experiment_Index) == set(partitions.loc[partitions.role.eq("outer_test"), "Experiment_Index"])
    floor = json.loads((out / "baseline.json").read_text())["balanced_accuracy_floor"]
    base = np.load(out / "untuned_inner_probability.npy")
    for directory in (out / "search").iterdir():
        trials = pd.read_csv(directory / "trials.csv", float_precision="round_trip")
        assert len(trials) == cfg["trials"] and trials.trial.tolist() == list(range(cfg["trials"]))
        objective = "CMR" if directory.name.startswith("CMR") else "macro_F1"
        eligible = trials[trials.feasible]
        assert eligible.balanced_accuracy.ge(floor).all()
        if len(eligible):
            best = json.loads((directory / "best.json").read_text())
            sorted_trials = eligible.sort_values([objective, "trial"], ascending=[objective == "CMR", True])
            assert best["trial"] == sorted_trials.trial.iloc[0]
            p = np.load(directory / "best_probability.npy")
            result = metrics(inside.Fault_Mode, CLASSES[p.argmax(axis=1)])
            for k, v in result.items():
                np.testing.assert_allclose(v, best["metrics"][k], atol=1e-12)
    candidates = pd.read_csv(out / "cost_candidates.csv", float_precision="round_trip")
    decisions = json.loads((out / "decisions.json").read_text())
    assert len(decisions) == 7*len(cfg["budgets"])
    outer_cache = load_cache(out / "cache/outer.npz")
    for d in decisions:
        arm, budget = d["arm"], d["budget"]
        tag = f"{arm}_b{int(budget*100)}"
        if arm in ["minimum_cost", "metacost"]:
            pool = []
            for k in cfg["kappas"]:
                pred = scalar_cost_prediction(base, k) if arm == "minimum_cost" else CLASSES[
                    np.load(out / f"metacost_k{k}_inner_probability.npy").argmax(axis=1)]
                result = metrics(inside.Fault_Mode, pred)
                record = candidates[candidates.arm.eq(arm) & candidates.budget.eq(budget) & candidates.kappa.eq(k)].iloc[0]
                for name, value in result.items():
                    np.testing.assert_allclose(value, record[name], atol=1e-12)
                if result["FA0"] <= budget and np.isfinite(result["CMR"]):
                    pool.append((result["CMR"], result["FA0"], k))
            chosen = min(pool)[2] if pool else None
            assert chosen == d["kappa"] and d["budget_failed"] == (chosen is None)
            if chosen is None:
                assert np.array_equal(outside[tag], outside.untuned_base)
            elif arm == "minimum_cost":
                pred = scalar_cost_prediction(np.load(out / "untuned_external_probability.npy"), chosen)
                assert np.array_equal(pred, outside[tag])
            else:
                p = np.load(out / f"metacost_k{chosen}_external_probability.npy")
                assert np.array_equal(CLASSES[p.argmax(axis=1)], outside[tag])
                posterior = np.load(out / "metacost_outer_posterior.npy")
                labels = scalar_cost_prediction(posterior, chosen)
                available = np.unique(labels)
                model = lgb.Booster(model_file=str(out / f"models/metacost_k{chosen}.txt"))
                native = model.predict(outer_cache["X_eval"])
                mapped = np.zeros_like(p)
                for col, label in enumerate(available):
                    mapped[:, np.flatnonzero(CLASSES == label)[0]] = native[:, col] if native.ndim > 1 else (1-native if col == 0 else native)
                np.testing.assert_allclose(mapped, p, atol=1e-12)
        else:
            directory = out / "search" / arm
            p = base if arm == "untuned" or d["search_failed"] else np.load(directory / "best_probability.npy")
            threshold, failed = select_threshold(inside.Fault_Mode.to_numpy(), p, budget)
            assert d["budget_failed"] == failed
            assert d["threshold"] == ("infinity" if threshold == np.inf else threshold)
            assert np.array_equal(threshold_predictions(p, threshold), inside[tag])
            p_outer = np.load(out / f"{arm}_external_probability.npy")
            assert np.array_equal(threshold_predictions(p_outer, threshold), outside[tag])
        assert "outer_metrics" not in d
        np.testing.assert_allclose(metrics(inside.Fault_Mode, inside[tag])["FA0"], d["inner_metrics"]["FA0"], atol=1e-12)
    record = dict(all_checks_passed=True, hashes=len(manifest["output_sha256"]),
                  internal_decisions=len(decisions), outer_performance_calculated=False,
                  run_manifest_sha256=sha(out / "run_manifest.json"))
    (out / "verification.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(f"Verified {len(decisions)} internal decisions and {record['hashes']} hashes; outer outcomes not evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/e1_pilot")
    main(ROOT / parser.parse_args().output)
