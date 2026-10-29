"""Independent scalar check of completed E10 detection rates and output hashes."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e10_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    completion = json.loads((OUT / "completion.json").read_text())
    assert len(completion["completed"]) == 50
    verified, frames = 0, []
    for item in completion["completed"]:
        folder = OUT / f"seed_{item['seed']}_outer_{item['outer']}"
        assert item == json.loads((folder / "complete.json").read_text())
        assert item["completed_trials"] == 100 and item["internal_fits"] == 500
        for name, digest in item["output_sha256"].items():
            assert sha(folder / name) == digest, folder / name
            verified += 1
        frames.append(pd.read_csv(folder / "external_predictions.csv"))
    data = pd.concat(frames, ignore_index=True)
    saved = pd.read_csv(OUT / "metrics_by_seed.csv").set_index(["seed", "arm", "target", "budget"])
    maximum = 0.
    checked = 0
    for key, group in data.groupby(["seed", "arm", "target", "budget"]):
        assert len(group) == 1195 and group.Experiment_Index.is_unique
        assert group.Serial_Number.nunique() == 54
        actual, detected = Counter(), Counter()
        exact, normal_alarms, critical_misses, severity_hits = Counter(), 0, 0, 0
        for row in group.itertuples():
            y, pred = row.Fault_Mode, row.prediction
            actual[y] += 1
            alarm = pred.startswith("F")
            if alarm:
                detected[y] += 1
            if y == pred:
                exact[y] += 1
            if y.startswith("N") and alarm:
                normal_alarms += 1
            if y in ("F03", "F06"):
                critical_misses += int(not alarm)
                severity_hits += int(pred in ("F03", "F06"))
        values = dict(FA0=normal_alarms/(actual["N01"]+actual["N02"]),
                      CMR=critical_misses/(actual["F03"]+actual["F06"]),
                      R3=severity_hits/(actual["F03"]+actual["F06"]))
        for cls in ("F01", "F02", "F03", "F04", "F05", "F06"):
            values[f"detect_{cls}"] = detected[cls]/actual[cls]
            values[f"exact_{cls}"] = exact[cls]/actual[cls]
        values["min_recall"] = min(values[f"detect_{c}"] for c in ("F01", "F02", "F03", "F04", "F05"))
        values["mean_recall"] = sum(values[f"detect_{c}"] for c in ("F01", "F02", "F03", "F04", "F05"))/5
        row = saved.loc[key]
        for metric, value in values.items():
            error = abs(value-row[metric])
            maximum = max(maximum, error)
            assert error < 1e-12, (key, metric, value, row[metric])
        assert bool(row.external_meets) == (values["min_recall"] >= key[2]-1e-12 and values["FA0"] <= key[3]+1e-12)
        checked += 1
    summary = pd.read_csv(OUT / "summary.csv")
    for row in summary.itertuples():
        part = saved.reset_index()
        part = part[(part.arm == row.arm) & np.isclose(part.target, row.target) & np.isclose(part.budget, row.budget)]
        assert len(part) == 10
        for metric in ("FA0", "CMR", "min_recall", "mean_recall", "detect_F01", "detect_F02", "detect_F03", "detect_F04", "detect_F05", "detect_F06", "exact_F06"):
            assert abs(part[metric].mean()-getattr(row, metric)) < 1e-12
    report = dict(verified_utc=datetime.now(timezone.utc).isoformat(), status="passed",
                  completed_partitions=50, candidates=5000, internal_fits=25000,
                  outer_refits=sum(x["outer_refits"] for x in completion["completed"]),
                  output_hashes_verified=verified, seed_arm_scenario_checks=checked,
                  maximum_scalar_metric_error=maximum,
                  primary_internal_feasible_total=int(summary.inner_feasible_folds.sum()),
                  primary_external_meets_total=int(summary.external_meets_seeds.sum()),
                  summary_sha256=sha(OUT / "summary.csv"), completion_sha256=sha(OUT / "completion.json"))
    (ROOT / "data/e10_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
