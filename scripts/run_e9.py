"""E9: resumable queue, low priority so that it does not starve concurrent runs."""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data/e9_fixed_model"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
WORKERS, THREADS = 3, 2
BELOW_NORMAL = 0x00004000   # Windows priority class


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(seed):
    path = OUT/f"preds_seed{seed}.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path, usecols=["split", "Experiment_Index"])
    return frame.split.eq("unseen").sum() == 1196 and frame.split.eq("leak").sum() > 0


def task(seed):
    if complete(seed):
        return seed, "skipped", 0.0
    (OUT/"logs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(OUT/"logs"/f"seed{seed}.log", "w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(ROOT/"scripts/leakage_fixed_model.py"), "--seed", str(seed),
                                 "--threads", str(THREADS)], stdout=log, stderr=subprocess.STDOUT,
                                creationflags=BELOW_NORMAL if sys.platform == "win32" else 0)
    ok = result.returncode == 0 and complete(seed)
    return seed, "ok" if ok else f"FAILED({result.returncode})", time.perf_counter()-started


def main():
    freeze = json.loads((OUT/"protocol_freeze.json").read_text(encoding="utf-8-sig"))
    for rel, digest in freeze["sha256"].items():
        assert sha(ROOT/rel) == digest, f"{rel} changed after freezing"
    failed = False
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for seed, status, seconds in pool.map(task, SEEDS):
            print(json.dumps(dict(seed=seed, status=status, seconds=round(seconds, 1))), flush=True)
            failed |= status.startswith("FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
