"""Small end-to-end E10 partition, with fits replaced by an uninformative model."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e10_run as runner


def test_partition_roundtrip_preserves_infeasible_infinite_thresholds(tmp_path, monkeypatch):
    source = tmp_path / "historical"
    source.mkdir()
    (source / "cache").mkdir()
    inner = pd.DataFrame(dict(Experiment_Index=np.arange(40), Serial_Number="inner",
                              Fault_Mode=np.tile(runner.np.array(["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]), 5)))
    outer = inner.iloc[:8].copy()
    outer.Experiment_Index += 100
    outer.Serial_Number = "outer"
    for budget in (.3, .4, .5):
        inner[f"untuned_b{int(budget*100)}"] = "N01"
        outer[f"untuned_b{int(budget*100)}"] = "N01"
    inner.to_csv(source / "inner_predictions.csv", index=False)
    outer.to_csv(source / "external_predictions.csv", index=False)
    for k in range(5):
        start = k*8
        part = inner.iloc[start:start+8]
        pd.DataFrame(dict(Serial_Number=["train"])).to_csv(source / f"cache/inner{k}_training_metadata.csv", index=False)
        part.to_csv(source / f"cache/inner{k}_evaluation_metadata.csv", index=False)
        np.savez(source / f"cache/inner{k}.npz", X=np.zeros((8, 2)), y=part.Fault_Mode.to_numpy(dtype=str),
                 X_eval=np.zeros((8, 2)), y_eval=part.Fault_Mode.to_numpy(dtype=str))
    np.savez(source / "cache/outer.npz", X=np.zeros((8, 2)), y=outer.Fault_Mode.to_numpy(dtype=str),
             X_eval=np.zeros((8, 2)), y_eval=outer.Fault_Mode.to_numpy(dtype=str))
    for arm in runner.EXISTING:
        np.save(source / f"{arm}_external_probability.npy", np.full((8, 8), 1/8))
        inside = source / ("untuned_inner_probability.npy" if arm == "untuned" else f"search/{arm}/best_probability.npy")
        inside.parent.mkdir(parents=True, exist_ok=True)
        np.save(inside, np.full((40, 8), 1/8))
    runner.write(source / "decisions.json", [dict(arm="untuned", budget=b, budget_failed=False, threshold="infinity") for b in (.3, .4, .5)])
    manifest = dict(output_sha256={p.relative_to(source).as_posix(): runner.sha(p) for p in source.rglob("*") if p.is_file()})
    runner.write(source / "run_manifest.json", manifest)
    out = tmp_path / "new"
    out.mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", out)
    monkeypatch.setattr(runner, "TRIALS", 2)
    monkeypatch.setattr(runner, "peak_memory_mb", lambda: None)
    original_load = runner.load_cache
    def load(path):
        if Path(path).name == "outer.npz":
            assert (out / "seed_42_outer_0/inner_selections.json").exists()
        return original_load(path)
    monkeypatch.setattr(runner, "load_cache", load)
    def fit(cache, params, f05, s3, save=None):
        if save:
            save.write_text("mock model")
        return np.full((len(cache["y_eval"]), 8), 1/8)
    monkeypatch.setattr(runner, "fit_probability", fit)
    result = runner.run_partition((dict(seed=42, outer=0, output="historical",
                                    run_manifest_sha256=runner.sha(source / "run_manifest.json")), "full"))
    assert result["status"] == "complete" and result["completed_trials"] == 2
    metrics = pd.read_csv(out / "seed_42_outer_0/fold_metrics.csv")
    assert len(metrics) == 72
    assert not metrics.inner_feasible.any()
    assert metrics.FA0.eq(0).all()
    assert metrics.min_recall.eq(0).all()
