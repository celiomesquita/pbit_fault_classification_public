from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import loso_f06_nested as e5

TARGET = "SN 054"


class PopulationTests(unittest.TestCase):
    def test_population_and_f06_units(self):
        features, long_df = e5.load_population()
        self.assertEqual((len(features), features.Serial_Number.nunique()), (1487, 65))
        self.assertEqual(set(features[features.Fault_Mode == "F06"].Serial_Number), set(e5.TARGETS))
        self.assertTrue(set(long_df.Experiment_Index) <= set(features.Experiment_Index))
        self.assertTrue(features.Experiment_Index.isin([931, 949, 1107]).sum() == 0)

    def test_counts_helper(self):
        features, _ = e5.load_population()
        train = features[features.Serial_Number != TARGET]
        info = e5.counts(train)
        self.assertEqual(info["F06_units"], 3)
        self.assertEqual(info["cycles"], len(train))


class SmokeRunTests(unittest.TestCase):
    def test_target_absent_and_reproducible(self):
        outputs = []
        for tag in ("test_a", "test_b"):
            destination = e5.OUT/"smoke"/tag
            if destination.exists():
                shutil.rmtree(destination)
            record = e5.run(TARGET, 42, 1, 2, 2, destination, True)
            self.assertTrue(record["checks"]["target_absent_from_all_training_and_selection"])
            self.assertEqual(record["training"]["F06_units"], 3)
            outputs.append(pd.read_csv(destination/"predictions.csv"))
        pd.testing.assert_frame_equal(outputs[0], outputs[1])
        features, _ = e5.load_population()
        target_cycles = set(features[features.Serial_Number == TARGET].Experiment_Index)
        self.assertEqual(set(outputs[0].Experiment_Index), target_cycles)
        for tag in ("test_a", "test_b"):
            shutil.rmtree(e5.OUT/"smoke"/tag)


if __name__ == "__main__":
    unittest.main()
