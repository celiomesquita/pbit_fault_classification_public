from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crc_repeated_partitions import calibration_curve, evaluate


class RepeatedCRCTests(unittest.TestCase):
    def test_unit_loss_does_not_weight_units_by_cycle_count(self):
        data = pd.DataFrame(dict(Serial_Number=["a"] * 9 + ["b"],
                                 Severity_Index=[3] * 10, base_severity=[3] * 9 + [0],
                                 severity3_score=[.8] * 9 + [.2]))
        curve = calibration_curve(data)
        self.assertEqual(curve.mean_unit_loss.iloc[0], .5)
        self.assertAlmostEqual(curve.corrected_criterion.iloc[0], 2 / 3)
        metrics, units = evaluate(data, np.inf)
        self.assertEqual(metrics["mean_unit_loss"], .5)
        self.assertEqual(metrics["CMR"], .1)
        self.assertEqual(units.misses.sum(), 1)

    def test_floor_and_zero_score_alarm_all_endpoint(self):
        data = pd.DataFrame(dict(Serial_Number=list("abcdefg"), Severity_Index=[3] * 7,
                                 base_severity=[0] * 7, severity3_score=[0.] * 7))
        curve = calibration_curve(data)
        self.assertFalse((curve.corrected_criterion <= .10).any())
        eligible = curve[curve.corrected_criterion <= .125]
        self.assertEqual(eligible.threshold.iloc[0], 0)
        self.assertEqual(eligible.mean_unit_loss.iloc[0], 0)
        normal = pd.DataFrame(dict(Serial_Number=["n"], Severity_Index=[0],
                                   base_severity=[0], severity3_score=[0.]))
        metrics, _ = evaluate(pd.concat([data, normal]), 0)
        self.assertEqual(metrics["CMR"], 0)
        self.assertEqual(metrics["FA0"], 1)

    def test_monotone_curve_and_promotion_only(self):
        data = pd.DataFrame(dict(Serial_Number=["a", "a", "b", "b"],
                                 Severity_Index=[3, 3, 3, 0], base_severity=[0, 2, 0, 1],
                                 severity3_score=[.9, .6, .1, .01]))
        curve = calibration_curve(data)
        np.testing.assert_allclose(curve.mean_unit_loss, [.75, .5, .5, 0, 0])
        metrics, _ = evaluate(data, .6)
        self.assertEqual(metrics["CMR"], 1 / 3)
        self.assertEqual(metrics["FA0"], 1)


if __name__ == "__main__":
    unittest.main()
