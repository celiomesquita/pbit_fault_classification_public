"""E7: queue the four comparison scripts on the audited population (resumable)."""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e7_audited"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
SCRIPTS = {"leakage": "leakage_stratification.py", "families": "families_compare.py",
           "ablation": "ablation_features.py", "imbalance": "imbalance_ablation.py"}
EXPECTED_ROWS = {"leakage": 3*1195, "families": 1195, "ablation": 1195, "imbalance": 1195}
WORKERS = 8
THREADS = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(name, seed):
    path = OUT/name/f"preds_seed{seed}.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path, usecols=["Experiment_Index"])
    return len(frame) == EXPECTED_ROWS[name] and 1107 not in set(frame.Experiment_Index)


def task(item):
    name, seed = item
    if complete(name, seed):
        return name, seed, "skipped", 0.0
    (OUT/"logs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = {**os.environ, "PBIT_POPULATION": "audited"}
    with open(OUT/"logs"/f"{name}_seed{seed}.log", "w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(ROOT/"scripts"/SCRIPTS[name]), "--seed", str(seed),
                                 "--threads", str(THREADS)], env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = result.returncode == 0 and complete(name, seed)
    return name, seed, "ok" if ok else f"FAILED({result.returncode})", time.perf_counter()-started


def main():
    freeze = json.loads((OUT/"protocol_freeze.json").read_text(encoding="utf-8-sig"))
    for rel, digest in freeze["sha256"].items():
        assert sha(ROOT/rel) == digest, f"{rel} changed after freezing"
    items = [(name, seed) for seed in SEEDS for name in SCRIPTS]
    failed = False
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for name, seed, status, seconds in pool.map(task, items):
            print(json.dumps(dict(script=name, seed=seed, status=status, seconds=round(seconds, 1))), flush=True)
            failed |= status.startswith("FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
