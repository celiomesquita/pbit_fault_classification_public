from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pbit.data_validation import (  # noqa: E402
    assert_expected_long_dataset,
    load_long_dataset,
    print_summary,
    summarize_long_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the long/tidy PBIT dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "PBIT_atuador_DESCARAC_r1_long.csv",
        help="Path to the long/tidy PBIT CSV.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Print diagnostics without enforcing the known r1 invariants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_long_dataset(args.input)
    summary = summarize_long_dataset(df)
    print_summary(summary)
    if not args.no_strict:
        assert_expected_long_dataset(summary)
        print()
        print("Strict validation passed.")


if __name__ == "__main__":
    main()
