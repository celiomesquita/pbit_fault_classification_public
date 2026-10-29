"""Re-predict saved E1 models; check probabilities without scoring outer labels."""
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from nested_cv_e1 import load_cache
from verify_e1_pilot import CLASSES, scalar_cost_prediction

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/e1_pilot")
out = ROOT / parser.parse_args().output
cache = load_cache(out / "cache/outer.npz")
names = []
for path in sorted((out / "models").glob("*.txt")):
    name = path.stem
    labels = cache["y"]
    if name.startswith("metacost_k"):
        kappa = int(name.split("_k")[1])
        labels = scalar_cost_prediction(np.load(out / "metacost_outer_posterior.npy"), kappa)
    available = np.unique(labels)
    native = lgb.Booster(model_file=str(path)).predict(cache["X_eval"])
    saved = np.load(out / f"{name}_external_probability.npy")
    aligned = np.zeros_like(saved)
    for column, label in enumerate(available):
        aligned[:, np.flatnonzero(CLASSES == label)[0]] = native[:, column] if native.ndim > 1 else (
            1-native if column == 0 else native)
    np.testing.assert_allclose(aligned, saved, rtol=1e-12, atol=1e-12)
    names.append(name)
record = dict(all_checks_passed=True, models=names, outer_performance_calculated=False,
              verification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out / "model_verification.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(f"Saved-model probability reproduction passed for {len(names)} models; outer performance not calculated.")
