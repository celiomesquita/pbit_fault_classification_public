"""Independent scalar verification of E12 errors and sampled neighbour distances."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e12_diagnostic"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify():
    completion = read(OUT / "completion.json")
    frozen = read(OUT / "freeze.json")
    for rel, digest in frozen["input_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    for rel, digest in completion["output_sha256"].items():
        assert hashlib.sha256((OUT / rel).read_bytes()).hexdigest() == digest, rel
    counter = defaultdict(lambda: dict(ids=set(), miss=0, detected=0, exact=0))
    items = read(ROOT / "data/e10_v1/completion.json")["completed"]
    for item in items:
        folder = ROOT / f"data/e10_v1/seed_{item['seed']}_outer_{item['outer']}"
        data = pd.read_csv(folder / "external_predictions.csv")
        for row in data.itertuples():
            if row.target != .90 or row.budget != .40 or row.Fault_Mode not in ("F02", "F05"):
                continue
            result = counter[row.arm, row.Fault_Mode, row.Serial_Number]
            result["ids"].add(row.Experiment_Index)
            result["miss"] += int(row.prediction in ("N01", "N02"))
            result["detected"] += int(row.prediction.startswith("F"))
            result["exact"] += int(row.prediction == row.Fault_Mode)
    units = pd.read_csv(OUT / "unit_errors.csv")
    maximum = 0.
    for row in units.itertuples():
        result = counter[row.arm, row.Fault_Mode, row.Serial_Number]
        assert len(result["ids"]) == row.n_cycles
        denominator = 10*len(result["ids"])
        assert result["miss"] + result["detected"] == denominator
        for value, observed in ((result["detected"]/denominator, row.detect_recall),
                                (result["exact"]/denominator, row.exact_recall),
                                (result["miss"]/10, row.mean_misses_per_seed)):
            error = abs(value-observed)
            maximum = max(maximum, error)
            assert error < 1e-12
    assert len(counter) == len(units)
    e10 = pd.read_csv(ROOT / "data/e10_v1/summary.csv")
    concentration = pd.read_csv(OUT / "error_concentration.csv")
    for row in concentration.itertuples():
        old = e10[(e10.arm == row.arm) & (e10.target == .90) & (e10.budget == .40)].iloc[0]
        assert abs(row.cycle_weighted_recall-old[f"detect_{row.Fault_Mode}"]) < 1e-12
    # Reconstruct raw and descriptor coordinates without using the diagnostic NN helper.
    population = pd.read_csv(ROOT / "data/ress_revision/freeze_v1/train_primary.csv").set_index("Experiment_Index")
    long = pd.read_csv(ROOT / "data/ress_revision/freeze_v1/long_primary.csv")
    long = long[long.Experiment_Index.isin(population.index)]
    raw = long.pivot(index="Experiment_Index", columns="Sample_Index", values="Measurement")
    std = raw.std(axis=1, ddof=0)
    shape = raw.sub(raw.mean(axis=1), axis=0).div(std.where(std > 1e-9, 1), axis=0)
    shape.loc[std <= 1e-9] = 0
    with warnings.catch_warnings():
        from ablation_features import f18_table
        f18 = f18_table(long)
    matrices = dict(raw_absolute=raw, raw_shape=shape, F18_IQR=f18,
                    F246_IQR=population.drop(columns=["Serial_Number", "Serial_Run_Index", "Fault_Mode", "Severity_Index"]))
    geometry = pd.read_csv(OUT / "geometry.csv")
    assert geometry.normal_neighbor_SN.ne(geometry.Serial_Number).all()
    assert geometry.fault_neighbor_SN.ne(geometry.Serial_Number).all()
    original = read(ROOT / "data/e1_full_v1/completion_manifest.json")["completed"]
    selected = geometry.groupby(["seed", "representation", "Fault_Mode"], sort=True).head(1)
    distance_error = 0.
    for row in selected.itertuples():
        origin = next(x for x in original if x["seed"] == row.seed and x["outer"] == row.outer)
        meta = pd.read_csv(ROOT / origin["output"] / "inner_predictions.csv")
        assert row.Serial_Number not in set(meta.Serial_Number)
        fitting = matrices[row.representation].loc[meta.Experiment_Index].to_numpy(float)
        query = matrices[row.representation].loc[row.Experiment_Index].to_numpy(float)
        if row.representation.endswith("IQR"):
            medians = np.nanquantile(fitting, .5, axis=0)
            medians[~np.isfinite(medians)] = 0
            for j in range(fitting.shape[1]):
                fitting[~np.isfinite(fitting[:, j]), j] = medians[j]
                if not np.isfinite(query[j]):
                    query[j] = medians[j]
            q25, q75 = np.quantile(fitting, [.25, .75], axis=0)
            retained = q75 > q25
            fitting = (fitting[:, retained]-medians[retained])/(q75-q25)[retained]
            query = (query[retained]-medians[retained])/(q75-q25)[retained]
        distances = np.sqrt(np.mean((fitting-query)**2, axis=1))
        for mask, expected_id, expected_distance in ((meta.Fault_Mode.isin(["N01", "N02"]), row.normal_neighbor_id, row.distance_normal),
                        (meta.Fault_Mode == row.Fault_Mode, row.fault_neighbor_id, row.distance_same_fault)):
            reference = np.flatnonzero(mask)
            chosen = reference[np.argmin(distances[reference])]
            assert int(meta.Experiment_Index.iloc[chosen]) == int(expected_id)
            error = abs(distances[chosen]-expected_distance)
            distance_error = max(distance_error, error)
            assert np.isclose(distances[chosen], expected_distance, rtol=1e-10, atol=1e-10)
    matches = pd.read_csv(OUT / "cross_label_signal_matches.csv")
    trace = pd.read_csv(OUT / "label_traceability.csv")
    assert all(trace[c].all() for c in trace if c.endswith("matches") or c.endswith("label") or c.endswith("unit"))
    result = dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
          input_hashes=len(frozen["input_sha256"]), output_hashes=len(completion["output_sha256"]),
          unit_arm_class_recomputations=len(units), max_error_rate_difference=maximum,
          independently_checked_neighbor_pairs=2*len(selected), max_distance_difference=distance_error,
          duplicate_cross_label_pairs=len(matches), label_records_verified=len(trace), classifier_fits=0)
    (OUT / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
