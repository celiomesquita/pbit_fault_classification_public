"""Independent scalar check of E17 (mapping M2) on two sources, without importing the producer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e17_severity_v1"
S, NORMAL = {"F02", "F03", "F06"}, {"N01", "N02"}
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]


def cmr(frame, column):
    crit = frame[frame.Fault_Mode.isin(S)]
    return float(crit[column].isin(NORMAL).mean())


a = pd.read_csv(OUT / "stage_a.csv")
checks = {}
diffs = []
for seed in SEEDS:
    f = pd.read_csv(ROOT / f"data/leakage/preds_seed{seed}.csv")
    diffs.append(cmr(f[f.arm.eq("random_stratified")], "rf") - cmr(f[f.arm.eq("grouped_faultmode")], "rf"))
produced = a[(a.stage == "A1_leakage") & (a.mapping == "M2") & (a.A == "rf_random")].estimate.iloc[0]
checks["leakage_rf_M2"] = dict(scalar=float(np.mean(diffs)), produced=float(produced))
test = pd.read_csv(ROOT / "data/final_test_predictions.csv")
crit = test[test.Fault_Mode.isin(S)]
miss_units = int(crit[crit["pred_main_CMR-tuned"].isin(NORMAL)].Serial_Number.nunique())
row = a[(a.stage == "A5_test") & (a.mapping == "M2") & (a.A == "pred_main_CMR-tuned")].iloc[0]
checks["test_CMR_tuned_M2_units_with_miss"] = dict(scalar=miss_units, produced=int(row.estimate))
ok = all(abs(v["scalar"] - v["produced"]) < 1e-12 for v in checks.values())
(OUT / "verification.json").write_text(json.dumps({"status": "passed" if ok else "failed", "checks": checks}, indent=2))
print(json.dumps(checks, indent=2), "passed" if ok else "FAILED")
