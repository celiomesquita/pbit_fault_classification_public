"""E5: resumable queue of nested LOSO runs (target x seed), several processes in parallel."""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data/e5_nested_loso"
TARGETS = ["SN 054", "SN 056", "SN 052", "SN 045"]
SEEDS = [42, 0, 7, 13, 21]
WORKERS = 5
THREADS = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def done(tag):
    manifest = OUT/"runs"/tag/"run_manifest.json"
    return manifest.exists() and (OUT/"runs"/tag/"predictions.csv").exists()


def task(item):
    target, seed = item
    tag = f"{target.replace(' ', '')}_seed{seed}"
    if done(tag):
        return tag, "skipped", 0.0
    partial = OUT/"runs"/tag
    if partial.exists():       # incomplete earlier attempt: keep it under a new name
        partial.rename(OUT/"runs"/f"{tag}_incomplete_{int(time.time())}")
    (OUT/"logs").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(OUT/"logs"/f"{tag}.log", "w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(ROOT/"scripts/loso_f06_nested.py"), "--target", target,
                                 "--seed", str(seed), "--threads", str(THREADS)], stdout=log,
                                stderr=subprocess.STDOUT)
    ok = result.returncode == 0 and done(tag)
    return tag, "ok" if ok else f"FAILED({result.returncode})", time.perf_counter()-started


def main():
    freeze = json.loads((OUT/"protocol_freeze.json").read_text(encoding="utf-8-sig"))
    for rel, digest in freeze["sha256"].items():
        assert sha(ROOT/rel) == digest, f"{rel} changed after freezing"
    items = [(t, s) for s in SEEDS for t in TARGETS]
    failed = False
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for tag, status, seconds in pool.map(task, items):
            print(json.dumps(dict(run=tag, status=status, seconds=round(seconds, 1))), flush=True)
            failed |= status.startswith("FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
