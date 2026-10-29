from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cluster_ratio_wild import ratio_parts, two_sided
from cluster_unit_mean import (lbar_abstentions, lbar_estimate, lbar_one_sided, unit_values)

N = np.array([2.0, 5.0, 11.0, 1.0, 3.0])
D = np.array([[0.5, 1.0, -2.2, 1.0, -0.7],
              [0.0, -0.4, 0.0, 0.0, 0.6]])
HS = np.array([0.03, 0.0, -0.02, 0.02])
TINY_SLACK = 1e-10


def scalar_wcr_c(x, h, v):
    """Explicit loops: restricted Rademacher t* with CV1 and tail p-values with +1."""
    g = len(x)
    mean = x.mean()
    se = np.sqrt(g/(g-1)*np.sum((x-mean)**2))/g
    observed = (mean-h)/se
    upper = lower = 1
    for sign in v:
        star = sign*(x-h)
        delta = star.mean()
        resid = star-delta
        se_star = np.sqrt(g/(g-1)*np.sum(resid**2))/g
        t_star = 0.0 if (delta == 0 and se_star == 0) else delta/se_star
        slack = TINY_SLACK*max(1, abs(observed))
        upper += t_star >= observed-slack
        lower += t_star <= observed+slack
    return upper/(len(v)+1), lower/(len(v)+1)


class UnitMeanTests(unittest.TestCase):
    def test_estimate_is_mean_of_unit_values(self):
        x = unit_values(D, N)
        self.assertTrue(np.allclose(x, D/N))
        self.assertTrue(np.allclose(lbar_estimate(D, N), (D/N).mean(axis=1)))

    def test_cv1_cv3_on_unit_weights_are_standard_errors_of_the_mean(self):
        x = unit_values(D, N)
        g = x.shape[1]
        theta, se1, se3, _ = ratio_parts(x, np.ones(g))
        self.assertTrue(np.allclose(theta, x.mean(axis=1)))
        self.assertTrue(np.allclose(se3, x.std(axis=1, ddof=1)/np.sqrt(g)))
        self.assertTrue(np.allclose(se1, x.std(axis=1, ddof=1)/np.sqrt(g)))

    def test_l_t_matches_scipy_one_sample_t(self):
        rng = np.random.default_rng(1)
        v = rng.choice([-1.0, 1.0], size=(len(D), 7, len(N)))
        upper, lower, _ = lbar_one_sided("L-T", D, N, HS, v)
        x = unit_values(D, N)
        for r in range(len(D)):
            for j, h in enumerate(HS):
                p = ttest_1samp(x[r], h).pvalue
                self.assertAlmostEqual(two_sided(upper[r, j], lower[r, j]), p, places=10)

    def test_sign_flip_matches_scalar_loop_and_exact_enumeration(self):
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(N))))
        v = np.broadcast_to(signs, (len(D), *signs.shape)).copy()
        upper, lower, _ = lbar_one_sided("L-WC", D, N, HS, v)
        x = unit_values(D, N)
        for r in range(len(D)):
            for j, h in enumerate(HS):
                u, l = scalar_wcr_c(x[r], h, signs)
                self.assertAlmostEqual(upper[r, j], u, places=12)
                self.assertAlmostEqual(lower[r, j], l, places=12)

    def test_sign_flip_matches_scalar_loop_with_random_weights(self):
        rng = np.random.default_rng(7)
        v = rng.choice([-1.0, 1.0], size=(len(D), 199, len(N)))
        upper, lower, _ = lbar_one_sided("L-WC", D, N, HS, v)
        x = unit_values(D, N)
        for r in range(len(D)):
            for j, h in enumerate(HS):
                u, l = scalar_wcr_c(x[r], h, v[r])
                self.assertAlmostEqual(upper[r, j], u, places=12)
                self.assertAlmostEqual(lower[r, j], l, places=12)

    def test_abstains_when_all_unit_losses_are_equal(self):
        flat = np.zeros((1, len(N)))
        self.assertTrue(lbar_abstentions("L-WC", flat, N)[0])
        self.assertTrue(lbar_abstentions("L-T", flat, N)[0])
        self.assertFalse(lbar_abstentions("L-T", D[:1], N)[0])

    def test_generator_truth_of_unit_mean_is_delta(self):
        from cluster_coverage_simulation import generate, setup
        calibration, profiles, _, _ = setup()
        profile = profiles["historical", "CMR"]
        for delta, rho in [(-0.04, 0.8), (0.02, 0.2), (0.0, 0.8)]:
            rng = np.random.default_rng(11)
            counts = generate(rng, 20000, 1, profile["n"], calibration["CMR"], delta, rho)
            x = unit_values(counts[:, 0, :], profile["n"].astype(float))
            per_replicate = x.mean(axis=1)
            se = per_replicate.std(ddof=1)/np.sqrt(len(per_replicate))
            self.assertLess(abs(per_replicate.mean()-delta), 4*se, msg=f"delta={delta}, rho={rho}")


if __name__ == "__main__":
    unittest.main()
