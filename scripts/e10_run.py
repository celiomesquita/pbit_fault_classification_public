"""E10: frozen inner-only selection, a timing pilot, then the fixed full batch."""
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

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.triage_policy import BUDGETS, TARGETS, curve, evaluate, predict, rank, select_curve
from nested_cv_e1 import base_params, fit_probability, load_cache

OUT = ROOT / "data/e10_v1"
EXISTING = ["untuned", "macro_F1_fixed", "macro_F1_optimized", "CMR_fixed", "CMR_optimized"]
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
SCENARIOS = [(r, b) for r in TARGETS for b in BUDGETS]
TRIALS, PILOT_TRIALS, THREADS, WORKERS = 100, 10, 4, 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "infinity" if value == np.inf else None
    return value


def write(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def peak_memory_mb():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess
    handle.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    ok = get_info(handle(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 1024**2 if ok else None


def sources():
    return [Path(__file__), ROOT / "src/pbit/triage_policy.py", ROOT / "scripts/e10_summarize.py",
            ROOT / "scripts/nested_cv_e1.py", ROOT / "scripts/nested_cv_controlled.py",
            ROOT / "src/pbit/modeling.py", ROOT / "src/pbit/augmentation.py", ROOT / "src/pbit/features.py",
            ROOT / "src/pbit/inference.py", ROOT / "src/pbit/cost_policies.py",
            ROOT / "docs/protocolos/E10.md", ROOT / "tests/test_e10_triage.py", ROOT / "tests/test_e10_run.py",
            ROOT / "data/e10_inventory/manifest.json", ROOT / "data/e1_full_v1/completion_manifest.json"]


def freeze():
    values = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources()}
    config = dict(trials=TRIALS, pilot_trials=PILOT_TRIALS, threads=THREADS, workers=WORKERS,
                  seeds=SEEDS, targets=TARGETS, budgets=BUDGETS,
                  primary_classes=["F01", "F02", "F03", "F04", "F05"], exploratory_classes=["F06"])
    if (OUT / "freeze.json").exists():
        old = read(OUT / "freeze.json")
        assert old["source_sha256"] == values, "A frozen source changed; preserve this version and amend."
        assert old["config"] == serial(config)
        return
    assert not OUT.exists(), "Use a new output version."
    # Verify audited sources before any new fitting or selection.
    inventory = read(ROOT / "data/e10_inventory/manifest.json")
    for name, expected in inventory["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    OUT.mkdir()
    for p in sources():
        if p.parts[-2] != "e10_inventory" and "data" not in p.relative_to(ROOT).parts:
            target = OUT / "frozen_sources" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    write(OUT / "freeze.json", dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
          source_sha256=values, config=config, python=sys.version,
          environment={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "lightgbm", "optuna", "scikit-learn")}))


def candidate(trial, seed):
    params = {**base_params(dict(seed=seed, threads=THREADS)),
              "n_estimators": trial.suggest_int("n_estimators", 200, 800),
              "learning_rate": trial.suggest_float("learning_rate", .01, .2, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 15, 127),
              "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
              "subsample": trial.suggest_float("subsample", .5, 1),
              "colsample_bytree": trial.suggest_float("colsample_bytree", .5, 1),
              "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
              "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True)}
    return dict(params=params, f05=trial.suggest_float("f05_boost", 1, 10),
                s3=trial.suggest_float("sev3_boost", 1, 20))


def run_partition(task):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    item, stage = task
    seed, outer = item["seed"], item["outer"]
    source = ROOT / item["output"]
    folder = OUT / f"seed_{seed}_outer_{outer}"
    if (folder / "complete.json").exists():
        return read(folder / "complete.json")
    folder.mkdir(exist_ok=True)
    historical = read(source / "run_manifest.json")
    assert sha(source / "run_manifest.json") == item["run_manifest_sha256"]
    used = {}

    def verified(relative):
        path = source / relative
        value = sha(path)
        assert value == historical["output_sha256"][relative], path
        used[relative] = value
        return path

    inner = pd.read_csv(verified("inner_predictions.csv"))
    caches = [load_cache(verified(f"cache/inner{k}.npz")) for k in range(5)]
    y = np.concatenate([c["y_eval"] for c in caches])
    assert np.array_equal(y, inner.Fault_Mode.to_numpy())
    # Cached augmented features are original E1 preprocessing; verify grouped roles.
    for k in range(5):
        train_meta = pd.read_csv(verified(f"cache/inner{k}_training_metadata.csv"))
        valid_meta = pd.read_csv(verified(f"cache/inner{k}_evaluation_metadata.csv"))
        assert set(train_meta.Serial_Number).isdisjoint(valid_meta.Serial_Number)
    study = optuna.create_study(study_name="E10", direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed + 1009 * outer),
                storage=f"sqlite:///{(folder / 'study.sqlite3').as_posix()}", load_if_exists=True)
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials), "Incomplete/failed trial requires an amendment."
    if stage == "full" and (folder / "full_stage.json").exists():
        assert len(study.trials) in (0, PILOT_TRIALS, TRIALS), "Interrupted adaptive search requires a recorded amendment."
    # Replaying sampler RNG from saved trials is not guaranteed: the fixed 10-trial
    # pilot is imported into a fresh full study whose adaptive sampler starts here.
    if stage == "full" and not (folder / "full_stage.json").exists():
        assert len(study.trials) in (0, PILOT_TRIALS)
        write(folder / "full_stage.json", dict(pilot_trials_imported=len(study.trials),
              sampler_seed=seed + 1009 * outer, sampler_reinitialised_after_pilot=bool(study.trials)))
    begin = time.monotonic()

    def objective(trial):
        started = time.monotonic()
        cfg = candidate(trial, seed)
        probability = np.concatenate([fit_probability(c, cfg["params"], cfg["f05"], cfg["s3"]) for c in caches])
        np.save(folder / f"trial_{trial.number:03d}_inner_probability.npy", probability)
        values = curve(y, probability)
        selections = [select_curve(values, r, b) for r, b in SCENARIOS]
        primary = selections[SCENARIOS.index((.90, .40))]
        write(folder / f"trial_{trial.number:03d}.json", dict(trial=trial.number, configuration=cfg,
              seconds=time.monotonic()-started, selections=selections))
        trial.set_user_attr("seconds", time.monotonic()-started)
        if primary["feasible"]:
            value = primary["FA0"]
        elif primary["status"] == "recall_infeasible":
            value = 1 + max(0., .90 - primary["min_recall"])
        else:
            value = 3.
        print(f"E10 {seed}/{outer} trial {trial.number+1}/{TRIALS}: {time.monotonic()-started:.1f}s", flush=True)
        return value

    limit = PILOT_TRIALS if stage == "pilot" else TRIALS
    study.optimize(objective, n_trials=max(0, limit-len(study.trials)))
    records = [read(folder / f"trial_{t.number:03d}.json") for t in study.trials]
    for record in records:
        for selection in record["selections"]:
            if selection["threshold"] == "infinity":
                selection["threshold"] = np.inf
    seconds = sum(r["seconds"] for r in records)
    write(folder / "inputs.json", used)
    if stage == "pilot":
        result = dict(seed=seed, outer=outer, completed_trials=len(records), internal_fits=5*len(records),
                      fit_seconds=seconds, elapsed_seconds=time.monotonic()-begin, peak_memory_mb=peak_memory_mb(),
                      external_scores_loaded=False, external_performance_calculated=False,
                      single_worker_projected_internal_hours=seconds/len(records)*TRIALS*50/3600,
                      caveat="Linear fit-time projection only; adaptive configurations, refits and contention differ.")
        write(OUT / "pilot.json", result)
        return result

    # All B3 choices are fixed from inner predictions before opening outer labels.
    chosen = []
    for i, (target, budget) in enumerate(SCENARIOS):
        winner = min(records, key=lambda r: rank(r["selections"][i], r["trial"]))
        chosen.append(dict(arm="B3_joint", trial=winner["trial"], **winner["selections"][i],
                           configuration=winner["configuration"]))
    for arm in EXISTING:
        name = "untuned_inner_probability.npy" if arm == "untuned" else f"search/{arm}/best_probability.npy"
        probability = np.load(verified(name))
        modes = [("B1_complete", False), ("B0_qpromotion", True)] if arm == "untuned" else [("B2_"+arm, False)]
        for label, promotion in modes:
            values = curve(y, probability, promotion)
            for target, budget in SCENARIOS:
                chosen.append(dict(arm=label, source_arm=arm, promotion=promotion,
                                    **select_curve(values, target, budget)))
    # Historical B0 is a continuity control, not the isolated unilateral comparison.
    historical_decisions = read(verified("decisions.json"))
    for target, budget in SCENARIOS:
        previous = next(d for d in historical_decisions if d["arm"] == "untuned" and np.isclose(d["budget"], budget))
        tag = f"untuned_b{int(round(budget*100))}"
        measured = evaluate(y, inner[tag].to_numpy())
        valid = np.isfinite(measured["min_recall"]) and np.isfinite(measured["FA0"])
        meets = bool(valid and measured["min_recall"] >= target-1e-12 and measured["FA0"] <= budget+1e-12)
        chosen.append(dict(arm="B0_historical", target=target, budget=budget,
                          status="historical_not_selected_for_recall", feasible=meets,
                          historical_budget_fallback=previous["budget_failed"],
                          threshold=previous["threshold"], **measured))
    write(folder / "inner_selections.json", chosen)

    outer_meta = pd.read_csv(verified("external_predictions.csv"))
    assert set(inner.Serial_Number).isdisjoint(outer_meta.Serial_Number)
    outer_cache = load_cache(verified("cache/outer.npz"))
    assert np.array_equal(outer_cache["y_eval"], outer_meta.Fault_Mode.to_numpy())
    scores = {arm: np.load(verified(f"{arm}_external_probability.npy")) for arm in EXISTING}
    trials = {}
    for selection in chosen:
        if selection["arm"] == "B3_joint" and selection["trial"] not in trials:
            cfg = selection["configuration"]
            trial = selection["trial"]
            scores_out = fit_probability(outer_cache, cfg["params"], cfg["f05"], cfg["s3"],
                                        save=folder / f"trial_{trial:03d}_model.txt")
            np.save(folder / f"trial_{trial:03d}_external_probability.npy", scores_out)
            trials[trial] = scores_out
    predictions, summaries, unit_summaries = [], [], []
    for selection in chosen:
        arm, target, budget = (selection[k] for k in ("arm", "target", "budget"))
        if arm == "B0_historical":
            decision = outer_meta[f"untuned_b{int(round(budget*100))}"].to_numpy()
        else:
            probability = trials[selection["trial"]] if arm == "B3_joint" else scores[selection["source_arm"]]
            decision = predict(probability, selection["threshold"], selection.get("promotion", False))
        frame = outer_meta[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        frame["prediction"], frame["arm"], frame["target"], frame["budget"] = decision, arm, target, budget
        frame["seed"], frame["outer"] = seed, outer
        predictions.append(frame)
        summaries.append(dict(arm=arm, target=target, budget=budget, seed=seed, outer=outer,
                              inner_feasible=selection["feasible"], inner_status=selection["status"],
                              **evaluate(outer_meta.Fault_Mode.to_numpy(), decision)))
        for unit, group in frame.groupby("Serial_Number"):
            unit_summaries.append(dict(arm=arm, target=target, budget=budget, seed=seed, outer=outer,
                                      Serial_Number=unit, **evaluate(group.Fault_Mode.to_numpy(), group.prediction.to_numpy())))
    pd.concat(predictions, ignore_index=True).to_csv(folder / "external_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(folder / "fold_metrics.csv", index=False)
    pd.DataFrame(unit_summaries).to_csv(folder / "unit_metrics.csv", index=False)
    write(folder / "inputs.json", used)
    result = dict(seed=seed, outer=outer, completed_trials=len(records), internal_fits=5*len(records),
                  outer_refits=len(trials), fit_seconds=seconds, stage_elapsed_seconds=time.monotonic()-begin,
                  peak_memory_mb=peak_memory_mb(), status="complete",
                  output_sha256={p.name: sha(p) for p in folder.iterdir()
                                 if p.is_file() and p.name not in ("complete.json", "study.sqlite3", "study.sqlite3-shm", "study.sqlite3-wal")})
    write(folder / "complete.json", result)
    return result


def run(stage):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    freeze()
    items = read(ROOT / "data/e1_full_v1/completion_manifest.json")["completed"]
    if stage == "pilot":
        item = next(i for i in items if i["seed"] == 42 and i["outer"] == 0)
        print(json.dumps(run_partition((item, stage)), indent=2), flush=True)
        return
    assert (OUT / "pilot.json").exists(), "Complete the timing pilot first."
    pilot = read(OUT / "pilot.json")
    assert not pilot["external_performance_calculated"] and pilot["completed_trials"] == PILOT_TRIALS
    # Immutable budget set before dispatch; no expansion in response to endpoints.
    if not (OUT / "batch_start.json").exists():
        write(OUT / "batch_start.json", dict(started_utc=datetime.now(timezone.utc).isoformat(),
              partitions=50, trials_per_partition=100, internal_fit_budget=25000,
              workers=WORKERS, threads_per_worker=THREADS,
              timing_projection=pilot, projected_compute_hours_divided_by_workers=
              pilot["single_worker_projected_internal_hours"]/WORKERS))
    completed = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(run_partition, (item, "full")): item for item in items}
        for future in as_completed(jobs):
            completed.append(future.result())
            write(OUT / "progress.json", dict(completed=len(completed), total=50,
                  updated_utc=datetime.now(timezone.utc).isoformat(), partitions=completed))
            print(f"E10 completed partitions: {len(completed)}/50", flush=True)
    write(OUT / "completion.json", dict(completed=completed, status="complete",
          finished_utc=datetime.now(timezone.utc).isoformat()))
    from e10_summarize import summarize
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    run(parser.parse_args().stage)
