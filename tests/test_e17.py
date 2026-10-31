import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e17_severity_sensitivity as e17  # noqa: E402


def scalar(frame, column, critical):
    crit = frame[frame.Fault_Mode.isin(critical)]
    norm = frame[frame.Fault_Mode.isin(e17.NORMAL)]
    cmr = crit[column].isin(e17.NORMAL).mean()
    units = crit.assign(m=crit[column].isin(e17.NORMAL)).groupby("Serial_Number").m.mean()
    fa0 = (~norm[column].isin(e17.NORMAL)).mean()
    return cmr, units.mean(), fa0


class TestE17(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        classes = list(e17.FAULT_ORDER)
        self.frame = pd.DataFrame({
            "Serial_Number": rng.choice([f"SN {i:03d}" for i in range(8)], 400),
            "Fault_Mode": rng.choice(classes, 400),
            "pred": rng.choice(classes, 400)})
        self.units = sorted(self.frame.Serial_Number.unique())

    def test_metrics_match_scalar_for_every_mapping(self):
        t = e17.unit_tensor(self.frame, "pred", self.units)
        for critical in e17.MAPS.values():
            m = e17.metrics(t, np.ones((1, len(self.units))), critical)
            cmr, lbar, fa0 = scalar(self.frame, "pred", critical)
            self.assertAlmostEqual(m["CMR"][0], cmr)
            self.assertAlmostEqual(m["Lbar"][0], lbar)
            self.assertAlmostEqual(m["FA0"][0], fa0)

    def test_fa0_does_not_depend_on_mapping(self):
        t = e17.unit_tensor(self.frame, "pred", self.units)
        values = {name: e17.metrics(t, np.ones((1, len(self.units))), c)["FA0"][0] for name, c in e17.MAPS.items()}
        self.assertEqual(len({round(v, 12) for v in values.values()}), 1)

    def test_mappings_are_the_frozen_ones(self):
        self.assertEqual(e17.MAPS["M0"], ["F03", "F06"])
        self.assertEqual(set(e17.MAPS), {"M0", "M1", "M2", "M3", "M4"})
        for critical in e17.MAPS.values():
            self.assertFalse(set(critical) & set(e17.NORMAL))

    def test_crc_curve_alarm_all_has_zero_loss(self):
        cal = pd.DataFrame({"Serial_Number": ["A", "A", "B"], "Fault_Mode": ["F06", "F06", "F06"],
                            "score": [0.2, 0.7, 0.4], "base_prediction": ["N01", "F01", "N01"]})
        candidates, criterion = e17.calibration_curve(cal, ["F06"])
        self.assertEqual(candidates[-1], 0.0)
        self.assertAlmostEqual(criterion[-1], 1 / 3)
        self.assertTrue(np.all(np.diff(criterion) <= 1e-15))


if __name__ == "__main__":
    unittest.main()
