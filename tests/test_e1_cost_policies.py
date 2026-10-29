from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pbit.cost_policies import CLASSES, SEVERITY, cost_matrix, minimum_cost_predictions, select_cost_policy
from nested_cv_controlled import metrics


class CostPolicyTests(unittest.TestCase):
    def test_cost_orientation_matches_scalar_bayes_risk(self):
        probability = np.random.default_rng(13).dirichlet(np.ones(8), size=20)
        for kappa in [2, 3, 5, 10, 20]:
            scalar = np.zeros((20, 8))
            for row, p in enumerate(probability):
                for decision in range(8):
                    for truth in range(8):
                        difference = SEVERITY[truth] - SEVERITY[decision]
                        c = 0 if truth == decision else (kappa*difference if difference > 0 else
                                                         -difference if difference < 0 else 1)
                        scalar[row, decision] += p[truth]*c
            np.testing.assert_allclose(probability @ cost_matrix(kappa), scalar)
            np.testing.assert_array_equal(minimum_cost_predictions(probability, kappa), CLASSES[scalar.argmin(axis=1)])

    def test_penalty_can_promote_a_normal_argmax(self):
        p = np.zeros((1, 8))
        p[0, 0], p[0, 4] = .9, .1
        self.assertEqual(minimum_cost_predictions(p, 2)[0], "N01")
        self.assertEqual(minimum_cost_predictions(p, 10)[0], "F03")

    def test_cost_selection_is_budgeted_and_ties_are_fixed(self):
        y = np.array(["N01", "N01", "F03", "F03"])
        candidates = {20: np.array(["F01", "F01", "F03", "F03"]),
                      3: np.array(["N01", "N01", "F03", "F03"]),
                      2: np.array(["N01", "N01", "F03", "F03"])}
        selected, rows = select_cost_policy(y, candidates, 0, metrics)
        self.assertEqual(selected, 2)
        self.assertFalse(next(r for r in rows if r["kappa"] == 20)["eligible"])
        selected, _ = select_cost_policy(y, {20:candidates[20]}, 0, metrics)
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
