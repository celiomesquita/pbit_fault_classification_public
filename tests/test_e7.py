from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_population(mode):
    previous = os.environ.get("PBIT_POPULATION")
    if mode:
        os.environ["PBIT_POPULATION"] = mode
    else:
        os.environ.pop("PBIT_POPULATION", None)
    try:
        import population
        return importlib.reload(population)
    finally:
        if previous is None:
            os.environ.pop("PBIT_POPULATION", None)
        else:
            os.environ["PBIT_POPULATION"] = previous


class PopulationTests(unittest.TestCase):
    def test_historical_is_untouched(self):
        module = load_population(None)
        frame = module.load_train(ROOT / "data/train_real.csv")
        self.assertEqual(len(frame), 1196)
        self.assertIn(1107, set(frame.Experiment_Index))
        self.assertEqual(module.out_dir(ROOT / "data/leakage"), ROOT / "data/leakage")

    def test_audited_excludes_only_1107(self):
        module = load_population("audited")
        frame = module.load_train(ROOT / "data/train_real.csv")
        raw = pd.read_csv(ROOT / "data/train_real.csv")
        self.assertEqual(len(frame), 1195)
        self.assertNotIn(1107, set(frame.Experiment_Index))
        self.assertEqual(frame.Serial_Number.nunique(), 54)
        self.assertEqual(set(raw.Experiment_Index) - set(frame.Experiment_Index), {1107})
        self.assertEqual(module.out_dir(ROOT / "data/leakage"), ROOT / "data/e7_audited/leakage")
        self.assertTrue(frame.index.equals(pd.RangeIndex(len(frame))))

    def test_n02_loses_one_cycle_of_sn045(self):
        module = load_population("audited")
        frame = module.load_train(ROOT / "data/train_real.csv")
        raw = pd.read_csv(ROOT / "data/train_real.csv")
        self.assertEqual((raw.Fault_Mode == "N02").sum() - (frame.Fault_Mode == "N02").sum(), 1)
        self.assertEqual(raw[(raw.Fault_Mode == "N02") & (raw.Serial_Number == "SN 045")].shape[0] - 1,
                         frame[(frame.Fault_Mode == "N02") & (frame.Serial_Number == "SN 045")].shape[0])


if __name__ == "__main__":
    unittest.main()
