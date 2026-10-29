import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e12_diagnostic as diagnostic


def test_error_concentration_keeps_cycle_and_unit_denominators_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "OUT", tmp_path)
    population = pd.DataFrame(dict(Experiment_Index=range(10), Serial_Number=["A"]*9+["B"], Fault_Mode="F02"))
    frames = []
    for seed in range(10):
        frame = population.copy()
        frame["seed"], frame["arm"] = seed, "B1_complete"
        frame["prediction"] = ["N01"]*9 + ["F01"]
        frames.append(frame)
    _, cycles = diagnostic.error_tables(pd.concat(frames), population)
    summary = pd.read_csv(tmp_path / "error_concentration.csv").iloc[0]
    assert summary.n_cycles == 10 and summary.n_units == 2
    assert np.isclose(summary.cycle_weighted_recall, .1)
    assert np.isclose(summary.equal_unit_recall, .5)
    assert summary.top1_miss_share == 1 and summary.top1_exposure_share == .9
    assert summary.always_missed_cycles == 9
    assert cycles.exact_recall.eq(0).all()
