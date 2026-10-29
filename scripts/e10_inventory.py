"""Audit E1 score reuse and grouped assignments before E10 policy selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e10_inventory"
ARMS = ["untuned", "macro_F1_fixed", "macro_F1_optimized", "CMR_fixed", "CMR_optimized"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run():
    completion = ROOT / "data/e1_full_v1/completion_manifest.json"
    rows, files = [], {}
    partitions = read(completion)["completed"]
    assert len(partitions) == 50
    all_outer = {}
    for item in partitions:
        folder = ROOT / item["output"]
        manifest_path = folder / "run_manifest.json"
        assert sha(manifest_path) == item["run_manifest_sha256"]
        manifest = read(manifest_path)

        def verify(relative):
            path = folder / relative
            value = sha(path)
            assert value == manifest["output_sha256"][relative], path
            files[path.relative_to(ROOT).as_posix()] = value
            return path

        inner = pd.read_csv(verify("inner_predictions.csv"))
        outer = pd.read_csv(verify("external_predictions.csv"))
        assignment = pd.read_csv(verify("inner_assignment.csv"))
        assert inner.Experiment_Index.is_unique and outer.Experiment_Index.is_unique
        assert inner.Experiment_Index.tolist() == assignment.Experiment_Index.tolist()
        assert set(inner.Serial_Number).isdisjoint(outer.Serial_Number)
        assert assignment.groupby("Serial_Number").inner_fold.nunique().eq(1).all()
        assert len(inner) + len(outer) == 1195
        assert set(inner.Experiment_Index).isdisjoint(outer.Experiment_Index)
        all_outer.setdefault(item["seed"], []).append(outer.assign(outer_fold=item["outer"]))
        evaluation = []
        for k in range(5):
            train = pd.read_csv(verify(f"cache/inner{k}_training_metadata.csv"))
            valid = pd.read_csv(verify(f"cache/inner{k}_evaluation_metadata.csv"))
            assert set(train.Serial_Number).isdisjoint(valid.Serial_Number)
            assert set(train.Serial_Number).isdisjoint(outer.Serial_Number)
            assert valid.Experiment_Index.tolist() == assignment.loc[
                assignment.inner_fold == k, "Experiment_Index"].tolist()
            evaluation.append(valid)
        valid = pd.concat(evaluation, ignore_index=True)
        assert valid.Experiment_Index.tolist() == inner.Experiment_Index.tolist()
        for arm in ARMS:
            inside_name = "untuned_inner_probability.npy" if arm == "untuned" else f"search/{arm}/best_probability.npy"
            outside_name = f"{arm}_external_probability.npy"
            selected = None if arm == "untuned" else read(verify(f"search/{arm}/best.json"))
            inside_path, outside_path = verify(inside_name), verify(outside_name)
            inside, outside = np.load(inside_path), np.load(outside_path)
            for p, n in ((inside, len(inner)), (outside, len(outer))):
                assert p.shape == (n, 8) and np.isfinite(p).all()
                assert (p >= 0).all() and (p <= 1).all()
                assert np.allclose(p.sum(axis=1), 1)
            verify(f"models/{arm}.txt")
            frozen = verify("frozen_sources/scripts/nested_cv_e1.py")
            assert "best_probability.npy" in frozen.read_text(encoding="utf-8")
            rows.append(dict(seed=item["seed"], outer=item["outer"], arm=arm,
                             inner_rows=len(inner), outer_rows=len(outer), grouped=True,
                             selected_trial=None if selected is None else selected["trial"],
                             inner_score=inside_path.relative_to(ROOT).as_posix(),
                             external_score=outside_path.relative_to(ROOT).as_posix(),
                             provenance="hashes + frozen producer + selected configuration",
                             numerical_refit_repeated=False))
    for seed, parts in all_outer.items():
        data = pd.concat(parts)
        assert len(parts) == 5 and len(data) == 1195 and data.Experiment_Index.is_unique
        assert data.Serial_Number.nunique() == 54
        assert data.groupby("Serial_Number").outer_fold.nunique().eq(1).all()
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        scope="E10.a only; no new threshold selection or external endpoint analysis",
        completion_sha256=sha(completion), partitions=50, model_pairs=len(rows),
        score_pairs_verified=True, input_sha256=files,
        limitations=["Hash and producer audit; inner fits were not numerically regenerated.",
                     "Historical population has already been inspected; E10 remains descriptive."]
    ), indent=2) + "\n", encoding="utf-8")
    print(f"E10.a: {len(rows)} score pairs, 50 grouped partitions; {len(files)} input hashes verified.")


if __name__ == "__main__":
    run()
