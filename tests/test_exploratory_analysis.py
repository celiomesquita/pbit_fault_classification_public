"""Verify the descriptive evidence cited in the exploratory analysis section."""
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


class ExploratoryAnalysisTests(unittest.TestCase):
    def test_coverage_and_nominal_variation(self):
        summary = pd.read_csv(ROOT / "resultados/exploratory_summary.csv").set_index("quantity").value
        self.assertEqual(summary["cycles"], 1472)
        self.assertEqual(summary["units"], 65)
        self.assertEqual(summary["units_with_multiple_conditions"], 18)
        self.assertEqual(summary["cycles_per_unit_median"], 20)
        self.assertAlmostEqual(summary["n02_dominant_unit_share"], 58 / 65)
        self.assertEqual(round(summary["n01_coefficient_of_variation_percent_median"], 3), .062)
        self.assertEqual(round(summary["n01_coefficient_of_variation_percent_max"], 1), 35.2)

    def test_spectral_summary_weights_units_rather_than_cycles(self):
        data = pd.read_csv(ROOT / "dados/pbit_features_v3.csv")
        spectral = pd.read_csv(ROOT / "dados/pbit_spectral_features_v3.csv")
        data = data.merge(spectral, on="experimento_idx", validate="one_to_one")
        exported = pd.read_csv(ROOT / "resultados/exploratory_unit_condition.csv")
        for condition, expected in (("N01", .10000678798442382), ("F02", .0891235392725293)):
            rows = exported[exported.condicao == condition]
            self.assertEqual(len(rows), data[data.condicao == condition].numero_serie.nunique())
            self.assertAlmostEqual(rows.low_frequency_share_median.median(), expected, places=12)
            self.assertNotAlmostEqual(data[data.condicao == condition].spectral_low_share.median(), expected, places=5)


if __name__ == "__main__":
    unittest.main()
