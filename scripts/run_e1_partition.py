"""Run one full-budget E1 partition using the unchanged pilot implementation."""
import argparse
from pathlib import Path

from nested_cv_e1 import CONFIG, ROOT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outer", type=int, choices=range(5), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = dict(CONFIG, seed=args.seed, outer=args.outer,
               scope="full_nested_training_no_outer_performance")
    run(ROOT / Path(args.output), cfg, workers=4)
