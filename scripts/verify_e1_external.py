"""Independent scalar recomputation of E1 external metrics (stdlib only) against metrics_by_seed.csv."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEV = {"N01": 0, "N02": 0, "F01": 1, "F02": 2, "F03": 3, "F04": 2, "F05": 2, "F06": 3}
CASES = [("untuned", "b40"), ("CMR_optimized", "b40"), ("macro_F1_fixed", "b50"),
         ("metacost", "b50"), ("minimum_cost", "b30"), ("CMR_fixed", "base")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT/"data/e1_full_v1/completion_manifest.json").read_text(encoding="utf-8-sig"))
    per_seed = defaultdict(list)
    for item in manifest["completed"]:
        with open(ROOT/item["output"]/"external_predictions.csv", newline="", encoding="utf-8") as f:
            per_seed[item["seed"]].extend(csv.DictReader(f))
    stored = {}
    with open(ROOT/"data/e1_external/metrics_by_seed.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stored[(int(row["seed"]), row["arm"], row["policy"])] = row
    worst, checked = 0.0, 0
    for seed, rows in per_seed.items():
        for arm, policy in CASES:
            col = f"{arm}_{policy}"
            crit = miss = norm = alarm = down = 0
            loss, alarm_rate = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
            for r in rows:
                t, p = SEV[r["Fault_Mode"]], SEV[r[col]]
                down += p < t
                if t == 3:
                    crit += 1; miss += p == 0
                    loss[r["Serial_Number"]][0] += p == 0; loss[r["Serial_Number"]][1] += 1
                if t == 0:
                    norm += 1; alarm += p >= 1
                    alarm_rate[r["Serial_Number"]][0] += p >= 1; alarm_rate[r["Serial_Number"]][1] += 1
            mine = dict(CMR=miss/crit, FA0=alarm/norm, DR=down/len(rows),
                        Lbar=sum(a/b for a, b in loss.values())/len(loss),
                        FA0_unit=sum(a/b for a, b in alarm_rate.values())/len(alarm_rate))
            if (seed, arm, policy) not in stored:
                continue
            for name, value in mine.items():
                worst = max(worst, abs(value-float(stored[(seed, arm, policy)][name])))
            checked += 1
    result = dict(item="E1_externo_independent_check", cases=[f"{a}_{p}" for a, p in CASES],
                  seed_case_pairs_checked=checked, max_abs_difference=worst, passed=bool(worst < 1e-12 and checked == 60),
                  metrics_by_seed_sha256=sha(ROOT/"data/e1_external/metrics_by_seed.csv"))
    (ROOT/"data/e1_external/independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
