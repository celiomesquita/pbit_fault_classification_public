from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats._resampling import _bca_interval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from cluster_ratio_methods import bca_probabilities, infer_batch, paired_draws, ratio_se
from cluster_coverage_simulation import generate, shifted_probability
from pbit.inference import paired_comparison


class ClusterCoverageTests(unittest.TestCase):
    def test_pairs_studentisation_matches_explicit_copied_units(self):
        n = np.array([2., 5., 11., 1.])
        d = np.array([[1., -1., 4., 0.], [-1., 2., 3., 1.]])
        weights = np.array([[[2, 0, 1, 1], [0, 2, 2, 0]], [[1, 1, 1, 1], [3, 0, 0, 1]]])
        estimates, errors = paired_draws(d, n, 4, weights)
        for row in range(2):
            for sample in range(2):
                ix = np.repeat(np.arange(4), weights[row, sample])
                theta, se, _ = ratio_se(d[row, ix], n[ix], 4)
                self.assertAlmostEqual(estimates[row, sample], theta)
                self.assertAlmostEqual(errors[row, sample], se)

    def test_current_percentile_is_reproduced_including_zero_exposure_units(self):
        n, d, units, b = np.array([2., 5., 11.]), np.array([[1., -1., 4.]]), 7, 2000
        rng = np.random.default_rng(32)
        weights = rng.multinomial(units, [1/7, 1/7, 1/7, 4/7], size=(1, b))[..., :3]
        samples, _ = paired_draws(d, n, units, weights)
        reference = paired_comparison(samples[0], d.sum()/n.sum(), 0.03)
        _, methods = infer_batch(d, n, units, np.random.default_rng(32), b)
        result = methods['percentile']
        self.assertAlmostEqual(result['p'][0], reference['p_centered'])
        np.testing.assert_allclose(result['bounds'][0, :2], [reference['ci95_low'], reference['ci95_high']])

    def test_bca_adjustment_matches_scipy_paired_jackknife(self):
        n = np.array([2., 5., 11., 1.])
        d = np.array([[1., -1., 4., 0.]])
        ix = np.array(list(itertools.product(range(4), repeat=4)))
        samples = (d[0, ix].sum(axis=1) / n[ix].sum(axis=1))[None, :]
        theta = np.array([d.sum()/n.sum()])
        def statistic(index, axis=-1):
            return d[0, index.astype(int)].sum(axis=axis)/n[index.astype(int)].sum(axis=axis)
        lower, upper, _ = _bca_interval([np.arange(4)], statistic, -1, 0.025, samples[0], None, np)
        actual = bca_probabilities(d, n, 4, theta, samples, np.array([0.025, 0.975]))
        np.testing.assert_allclose(actual[0], [lower, upper], rtol=1e-12)

    def test_planted_effect_preserves_exact_marginal_mean_and_bounds(self):
        # Two-point distribution with mean mu, independent of exposure weights.
        mu = 0.13
        p, probability = np.array([0., 1.]), np.array([1-mu, mu])
        for delta in [0., -0.04, -0.02, 0.02, 0.04]:
            shifted = shifted_probability(p, mu, delta)
            self.assertTrue(np.all((shifted >= 0) & (shifted <= 1)))
            self.assertAlmostEqual(probability @ shifted - mu, delta)

    def test_common_reference_and_dependent_seeds_conserve_count_bounds(self):
        n = np.array([1, 4, 20, 73])
        calibration = dict(mu=0.2, a=1., b=4.)
        counts = generate(np.random.default_rng(11), 200, 8, n, calibration, -0.02, 0.8)
        self.assertEqual(counts.shape, (200, 8, 4))
        self.assertTrue(np.all(np.abs(counts) <= n))
        np.testing.assert_allclose(counts*10, np.round(counts*10), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
