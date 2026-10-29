"""E7: select the audited population (experiment 1107 excluded) through PBIT_POPULATION=audited."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

AUDITED = os.environ.get("PBIT_POPULATION") == "audited"
EXCLUDED_BY_AUDIT = {1107}   # 931 and 949 are already absent from train_real.csv


def load_train(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if AUDITED:
        frame = frame[~frame["Experiment_Index"].isin(EXCLUDED_BY_AUDIT)].reset_index(drop=True)
    return frame


def out_dir(default: Path) -> Path:
    return default.parent / "e7_audited" / default.name if AUDITED else default
