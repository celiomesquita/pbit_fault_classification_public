from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pbit.inference import confusion_metrics, holm_adjust, paired_comparison
from pbit.modeling import FAULT_ORDER
from nested_cv_controlled import grouped_folds, metrics, select_threshold, threshold_predictions


class CorrectedEvaluationTests(unittest.TestCase):
    def test_pooled_metrics_match_sklearn_with_missing_classes(self):
        y = np.array(["N01", "N01", "F03", "F03", "F06"])
        prediction = np.array(["N02", "N01", "F03", "N01", "F03"])
        result = metrics(y, prediction)
        self.assertAlmostEqual(result["macro_F1"], f1_score(y, prediction, average="macro", zero_division=0))
        self.assertAlmostEqual(result["balanced_accuracy"], balanced_accuracy_score(y, prediction))
        self.assertAlmostEqual(result["CMR"], 1 / 3)
        self.assertAlmostEqual(result["FA0"], 0)

    def test_absent_denominators_are_undefined(self):
        cm = np.zeros((2, 8, 8))
        cm[0, 0, 0] = 1
        cm[1, 4, 4] = 1
        result = confusion_metrics(cm)
        self.assertTrue(np.isnan(result["CMR"][0]))
        self.assertTrue(np.isnan(result["FA0"][1]))

    def test_multiplicity_can_remove_equivalence(self):
        draws = np.linspace(-0.025, 0.015, 10000)
        self.assertTrue(paired_comparison(draws, -0.005, 0.024, 1)["equivalent"])
        self.assertFalse(paired_comparison(draws, -0.005, 0.024, 8)["equivalent"])
        self.assertAlmostEqual(paired_comparison(np.r_[draws, np.nan], -0.005, 0.024, 8)["discard_fraction"], 1 / 10001)

    def test_centered_null_and_holm(self):
        draws = np.linspace(-0.01, 0.01, 1001)
        self.assertEqual(paired_comparison(draws, 0, 0.02)["p_centered"], 1)
        self.assertLess(paired_comparison(draws + 0.1, 0.1, 0.02)["p_centered"], 0.01)
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])

    def test_policy_budget_and_infeasible_fallback(self):
        probability = np.zeros((4, len(FAULT_ORDER)))
        probability[:, 0] = [0.9, 0.8, 0.7, 0.6]
        probability[:, 4] = 1 - probability[:, 0]
        y = np.array(["N01", "N01", "F03", "F03"])
        threshold, failed = select_threshold(y, probability, 0)
        self.assertFalse(failed)
        prediction = threshold_predictions(probability, threshold)
        self.assertEqual(metrics(y, prediction)["CMR"], 0)
        self.assertEqual(metrics(y, prediction)["FA0"], 0)
        alarm_probability = probability.copy()
        alarm_probability[0, 0], alarm_probability[0, 4] = 0.1, 0.9
        threshold, failed = select_threshold(y, alarm_probability, 0)
        self.assertTrue(failed)
        self.assertEqual(threshold, np.inf)

    def test_grouping_excludes_entire_unit(self):
        frame = pd.DataFrame({"Serial_Number": np.repeat(np.arange(12), 3),
                              "Fault_Mode": np.repeat(["N01", "F03"] * 6, 3)})
        folds = grouped_folds(frame, 3, 42)
        covered = []
        for train, evaluation in folds:
            self.assertTrue(set(frame.iloc[train].Serial_Number).isdisjoint(frame.iloc[evaluation].Serial_Number))
            covered.extend(evaluation)
        self.assertEqual(sorted(covered), list(range(len(frame))))


if __name__ == "__main__":
    unittest.main()
