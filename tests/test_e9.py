from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import leakage_fixed_model as e9
from pbit.data_validation import load_long_dataset  # noqa: E402


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(ROOT / "data/train_real.csv")

    def test_fit_leak_split_properties(self):
        from leakage_stratification import make_folds
        for seed in (42, 7):
            for tr_idx, te_idx in make_folds(self.train, "grouped_faultmode", seed):
                units = self.train.iloc[tr_idx]
                fit, leak, dropped = e9.split_fit_leak(units, seed)
                self.assertTrue(set(fit.Experiment_Index).isdisjoint(leak.Experiment_Index))
                self.assertTrue(set(leak.Serial_Number) <= set(fit.Serial_Number))
                self.assertEqual(len(fit) + len(leak) + dropped, len(units))
                self.assertAlmostEqual(len(fit)/len(units), 0.8, delta=0.02)
                self.assertTrue(set(self.train.iloc[te_idx].Serial_Number).isdisjoint(fit.Serial_Number))

    def test_small_run_shapes_and_coverage(self):
        long_df = load_long_dataset(ROOT / "data/PBIT_atuador_DESCARAC_r1_long_clean.csv")
        frame, info = e9.run_seed(42, 2, self.train, long_df)
        unseen = frame[frame.split == "unseen"]
        self.assertEqual(len(unseen), len(self.train))
        self.assertEqual(set(unseen.Experiment_Index), set(self.train.Experiment_Index))
        leak = frame[frame.split == "leak"]
        # a cycle can be reserved in more than one fold (its unit is in training in four folds),
        # but only once within a fold
        self.assertTrue(all(part.Experiment_Index.is_unique for _, part in leak.groupby("fold")))
        # reserved cycles come from units that are not the fold's test units
        for fold, part in frame.groupby("fold"):
            self.assertTrue(set(part[part.split == "leak"].Serial_Number).isdisjoint(
                part[part.split == "unseen"].Serial_Number))
        self.assertEqual(len(info["dropped_reserved_cycles"]), 5)
        self.assertTrue(np.isin(frame.lgbm, ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]).all())


if __name__ == "__main__":
    unittest.main()
