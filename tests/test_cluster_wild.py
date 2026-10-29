from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cluster_ratio_methods import ratio_se
from cluster_ratio_wild import (confidence_set, one_sided, ratio_parts, rademacher,
                                two_sided)

N_UNITS = np.array([2, 5, 11, 1, 3])
CYCLES = [np.array(c, dtype=float) for c in
          ([0.3, -1.0], [1.0, 0.0, -0.1, 0.4, 0.2], [0.1]*6 + [-0.2]*5, [0.9], [-0.5, 0.0, 0.2])]


def cluster_design():
    y = np.concatenate(CYCLES)
    unit = np.repeat(np.arange(len(CYCLES)), [len(c) for c in CYCLES])
    return y, unit


def ols_mean_t(y, unit, h, variance):
    """Intercept-only OLS t statistic for beta=h with explicit cluster formulas."""
    g, n = unit.max()+1, len(y)
    beta = y.mean()
    u = y - beta
    if variance == "cv1":
        scores = np.array([u[unit == j].sum() for j in range(g)])
        se = np.sqrt(g/(g-1)*(n-1)/(n-1)*np.sum(scores**2)) / n
    else:
        loo = np.array([y[unit != j].mean() for j in range(g)])
        se = np.sqrt((g-1)/g*np.sum((loo-beta)**2))
    return (beta-h)/se, beta, se


class WildClusterTests(unittest.TestCase):
    def setUp(self):
        y, unit = cluster_design()
        self.y, self.unit = y, unit
        self.d = np.array([[c.sum() for c in CYCLES]])
        self.n = N_UNITS.astype(float)

    def test_ratio_is_clustered_intercept_and_cv1_matches_e3(self):
        theta, se1, se3, _ = ratio_parts(self.d, self.n)
        _, beta, cv1 = ols_mean_t(self.y, self.unit, 0, "cv1")
        _, _, cv3 = ols_mean_t(self.y, self.unit, 0, "cv3")
        e3_theta, e3_se, _ = ratio_se(self.d, self.n, len(self.n))
        self.assertAlmostEqual(theta[0], beta)
        self.assertAlmostEqual(se1[0], cv1)
        self.assertAlmostEqual(se1[0], e3_se[0])
        self.assertAlmostEqual(se3[0], cv3)

    def test_transformed_restricted_scores_equal_restricted_scores(self):
        h, x = 0.07, np.ones(len(self.y))
        free = np.zeros((len(self.y), 0))
        projection = np.eye(len(self.y)) - free @ np.linalg.pinv(free.T @ free) @ free.T
        for j in range(len(CYCLES)):
            rows = self.unit == j
            block = projection[np.ix_(rows, rows)]
            dot = x[rows] @ np.linalg.solve(block, self.y[rows] - h)
            self.assertAlmostEqual(dot, self.d[0, j] - h*self.n[j])

    def test_transformed_unrestricted_scores_equal_jackknife_form(self):
        x = np.ones((len(self.y), 1))
        projection = np.eye(len(self.y)) - x @ x.T / len(self.y)
        residual = self.y - self.y.mean()
        theta, _, _, loo = ratio_parts(self.d, self.n)
        for j in range(len(CYCLES)):
            rows = self.unit == j
            acute = x[rows, 0] @ np.linalg.solve(projection[np.ix_(rows, rows)], residual[rows])
            self.assertAlmostEqual(acute, self.n.sum()*(theta[0]-loo[0, j]))

    def test_p_values_match_cycle_level_bootstrap_loops(self):
        b, hs = 199, np.array([-0.2, 0.0, 0.05, 0.3])
        v = rademacher(np.random.default_rng(5), 1, b, len(self.n))
        x = np.ones((len(self.y), 1))
        projection = np.eye(len(self.y)) - x @ x.T / len(self.y)
        acute = np.empty_like(self.y)
        for j in range(len(CYCLES)):
            rows = self.unit == j
            acute[rows] = np.linalg.solve(projection[np.ix_(rows, rows)], (self.y - self.y.mean())[rows])
        for method, variance in [("wcr_c", "cv1"), ("wcr_v", "cv3"), ("wcu_b", "cv3")]:
            upper, lower, _ = one_sided(method, self.d, self.n, hs, v)
            for k, h in enumerate(hs):
                observed, beta, _ = ols_mean_t(self.y, self.unit, h, variance)
                stars = []
                for draw in v[0]:
                    if method == "wcu_b":
                        y_star = beta + draw[self.unit]*acute
                        stars.append(ols_mean_t(y_star, self.unit, beta, variance)[0])
                    else:
                        y_star = h + draw[self.unit]*(self.y - h)
                        stars.append(ols_mean_t(y_star, self.unit, h, variance)[0])
                stars = np.array(stars)
                self.assertAlmostEqual(upper[0, k], (1+np.sum(stars >= observed-1e-12))/(b+1))
                self.assertAlmostEqual(lower[0, k], (1+np.sum(stars <= observed+1e-12))/(b+1))

    def test_inverted_bounds_switch_acceptance_and_match_grid_hull(self):
        # Fourteen critical exposures of the historical profile; with five units the
        # constant Rademacher draws recur often enough to make WCR sets unbounded.
        n = np.array([30, 23, 22, 20, 16, 16, 16, 16, 15, 11, 9, 4, 1, 1], dtype=float)
        rng = np.random.default_rng(9)
        d = (rng.binomial(n.astype(int), 0.15, size=(4, len(n)))
             - rng.binomial(n.astype(int), 0.10, size=(4, len(n)))).astype(float)
        v = rademacher(rng, len(d), 399, len(n))
        for method in ["wcr_c", "wcr_v", "wcu_b", "cv3_t"]:
            bounds = confidence_set(method, d, n, v, iterations=50)
            self.assertTrue(np.isfinite(bounds).all())
            width = bounds[:, 1] - bounds[:, 0]
            eps = 1e-6*width
            for side, inward in [(0, 1), (1, -1)]:
                inside = bounds[:, side] + inward*eps
                outside = bounds[:, side] - inward*eps
                up_in, low_in, _ = one_sided(method, d, n, inside[:, None], v)
                up_out, low_out, _ = one_sided(method, d, n, outside[:, None], v)
                self.assertTrue(np.all(two_sided(up_in, low_in)[:, 0] > 0.05))
                self.assertTrue(np.all(two_sided(up_out, low_out)[:, 0] <= 0.05))
            for row in range(len(d)):
                grid = np.linspace(bounds[row, 0] - 3*width[row], bounds[row, 1] + 3*width[row], 3001)
                up, low, _ = one_sided(method, np.repeat(d[row:row+1], len(grid), 0), n,
                                       grid[:, None], np.repeat(v[row:row+1], len(grid), 0))
                accepted = grid[two_sided(up, low)[:, 0] > 0.05]
                np.testing.assert_allclose([accepted.min(), accepted.max()], bounds[row],
                                           atol=2*width[row]*6/3000)


if __name__ == "__main__":
    unittest.main()
