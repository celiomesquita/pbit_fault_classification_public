"""Checks for training-only balancing and paired unit-level comparisons."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_xgboost as comparison
import train_binary_model as model


class XGBoostComparisonTests(unittest.TestCase):
    def test_balancing_uses_current_training_labels(self):
        classifier = model.make_model("xgboost", model.XGBOOST_GRID[0], 7)
        with patch.object(XGBClassifier, "fit", return_value=classifier) as fitted:
            classifier.fit(np.zeros((4, 2)), np.array([0, 0, 0, 1]))
        np.testing.assert_allclose(fitted.call_args.kwargs["sample_weight"], [2 / 3, 2 / 3, 2 / 3, 2])

    def test_xgboost_training_is_reproducible(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        first = model.make_model("xgboost", model.XGBOOST_GRID[0], 7).fit(X, y).predict_proba(X)
        second = model.make_model("xgboost", model.XGBOOST_GRID[0], 7).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(first, second)

    def test_paired_comparison_aligns_cycles_and_preserves_unit_dependence(self):
        frame = pd.DataFrame({"repetition": [0, 0, 1, 1], "experimento_idx": [1, 2, 1, 2],
                              "numero_serie": ["A", "B", "A", "B"], "condicao": ["N01", "F03"] * 2,
                              "y": [0, 1] * 2, "score": [0.2, 0.8] * 2})
        for alpha in (5, 10, 20):
            frame[f"threshold_{alpha:02d}"] = 0.5
        rows = comparison.paired_intervals(frame, frame.iloc[::-1], "same", np.array([[1, 1], [2, 0], [0, 2]]))
        for row in rows:
            self.assertEqual((row["estimate"], row["ci_low"], row["ci_high"]), (0, 0, 0))
        changed = frame.copy()
        changed.loc[0, "experimento_idx"] = 99
        with self.assertRaises(AssertionError):
            comparison.paired_intervals(frame, changed, "mismatch", np.array([[1, 1]]))


if __name__ == "__main__":
    unittest.main()
