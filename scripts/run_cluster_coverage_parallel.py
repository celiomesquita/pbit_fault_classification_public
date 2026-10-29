"""Resume E3 across independent scenarios without changing frozen simulation code."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import multiprocessing
import sys
import time

import numpy as np

import cluster_coverage_simulation as simulation


def run_one(config, calibration, profile):
    started = time.perf_counter()
    records, arrays = simulation.simulate(config, calibration, profile, 5000, 2000, 32)
    destination = simulation.OUT / "run"
    csv = destination / f"scenario_{config['id']:02d}.csv"
    npz = destination / f"scenario_{config['id']:02d}.npz"
    records.to_csv(csv, index=False)
    np.savez_compressed(npz, **arrays)
    return config['id'], time.perf_counter()-started, {
        str(p.relative_to(simulation.ROOT)): simulation.sha(p) for p in [csv, npz]}


def main():
    calibration, profiles, _, inputs = simulation.setup()
    inputs["data/cluster_coverage/protocol_freeze.json"] = simulation.sha(simulation.OUT / "protocol_freeze.json")
    manifest_path = simulation.OUT / "run/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_sha256"] == inputs
    for name, digest in manifest["output_sha256"].items():
        assert simulation.sha(simulation.ROOT / name) == digest, name
    configs = [cfg for cfg in simulation.scenarios() if cfg['id'] not in manifest['completed']]
    manifest['execution_pool'] = dict(workers=4, batch=32, resumed_utc=simulation.utc(),
                                     launcher_sha256=simulation.sha(Path(__file__)),
                                     note="Independent scenario RNGs; frozen protocol/code and replicate counts unchanged")
    simulation.write_json(manifest_path, manifest)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, cfg, calibration[cfg['metric']], profiles[cfg['profile'], cfg['metric']]): cfg
                   for cfg in configs}
        for future in as_completed(futures):
            scenario_id, elapsed, hashes = future.result()
            manifest['completed'].append(scenario_id)
            manifest['output_sha256'].update(hashes)
            simulation.write_json(manifest_path, manifest)
            print(json.dumps(dict(scenario=scenario_id, seconds=round(elapsed, 2),
                                  completed=len(manifest['completed']), total=40)), flush=True)
    sys.argv = [str(simulation.ROOT / 'scripts/cluster_coverage_simulation.py'), '--mode', 'run']
    simulation.main()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
