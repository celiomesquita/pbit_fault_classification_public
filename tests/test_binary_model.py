"""Automated tests for the normal-versus-failure model. Run from the repository root: py -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import experimentos_base as base  # noqa: E402
import analyze_f02_100hz as hz  # noqa: E402
import build_spectral_features as spectral  # noqa: E402
import evaluate_generalization as generalization  # noqa: E402
import evaluate_label_noise as noise  # noqa: E402
import train_binary_model as model  # noqa: E402


class BinaryModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(base.RAIZ / "dados" / "pbit_features_corrigido_v3.csv")
        cls.sets = model.build_feature_sets(cls.df)

    def test_target_maps_normal_and_failure(self):
        self.assertEqual(list(model.make_target(["N01", "N02", "F01", "F06"])), [0, 0, 1, 1])

    def test_identifiers_are_never_features(self):
        for frame in self.sets.values():
            self.assertFalse({"experimento_idx", "numero_serie", "condicao"} & set(frame.columns))

    def test_feature_sets_are_finite(self):
        for frame in self.sets.values():
            self.assertTrue(np.isfinite(frame.values).all())

    def test_set_b_has_no_absolute_level(self):
        self.assertFalse(set(model.LEVEL_COLUMNS) & set(self.sets["B"].columns))

    def test_set_d_ignores_signal_scale(self):
        scaled = self.df.copy()
        for column in ["media", "mediana", "desvio_padrao", "minimo", "maximo", "pico_a_pico", "rms", "valor_inicial", "valor_final",
                       "delta_inicio_fim", "inclinacao", "desvio_primeira_metade", "desvio_segunda_metade", "maior_salto"]:
            scaled[column] = scaled[column] * 2.5
        pd.testing.assert_frame_equal(model.build_feature_sets(scaled)["D"], self.sets["D"], check_exact=False, atol=1e-9)

    def test_half_difference_matches_a_known_signal(self):
        values = np.r_[np.full(61, 1.0), np.full(62, 3.0)]
        frame = pd.DataFrame({"desvio_padrao": [values.std()], "desvio_primeira_metade": [values[:61].std()], "desvio_segunda_metade": [values[61:].std()]})
        self.assertAlmostEqual(model.half_mean_difference(frame).iloc[0], 2.0, places=9)

    def test_grouped_folds_never_share_a_unit(self):
        groups = self.df["numero_serie"].values
        for repetition in range(5):
            fold_ids = base.dobras(self.df, repetition)["E3b"]
            for fold in range(base.N_FOLD):
                self.assertFalse(set(groups[fold_ids == fold]) & set(groups[fold_ids != fold]))

    def test_selection_prefers_the_simplest_within_margin(self):
        candidates = pd.DataFrame({"feature_set": ["A", "D", "A"], "model": ["boosting", "tree", "logistic"], "inner_auc": [0.80, 0.795, 0.70]})
        pick = model.select_simplest(candidates)
        self.assertEqual((pick.feature_set, pick.model), ("D", "tree"))

    def test_selection_keeps_a_clearly_better_complex_model(self):
        candidates = pd.DataFrame({"feature_set": ["A", "D"], "model": ["boosting", "tree"], "inner_auc": [0.80, 0.70]})
        self.assertEqual(model.select_simplest(candidates).model, "boosting")

    def test_no_f02_mode_removes_only_f02(self):
        reduced = model.load_data(True)
        self.assertNotIn("F02", set(reduced["condicao"]))
        self.assertEqual(len(reduced), (self.df["condicao"] != "F02").sum())
        self.assertEqual(list(reduced.index), list(range(len(reduced))))

    def test_spectral_descriptors_find_a_known_tone(self):
        t = np.arange(123)
        signal = 2.0 + 0.1 * np.sin(2 * np.pi * 20 * t / 123)
        table = spectral.descriptors(signal[None, :])
        self.assertAlmostEqual(table["spectral_dominant_frequency"].iloc[0], 20 / 61.5, places=9)
        self.assertGreater(table["spectral_dominant_share"].iloc[0], 0.99)

    def test_spectral_shares_sum_to_one_and_ignore_scale(self):
        signal = np.random.default_rng(0).normal(3.0, 0.01, size=(4, 123))
        first = spectral.descriptors(signal)
        shares = first[["spectral_low_share", "spectral_mid_share", "spectral_high_share"]].sum(axis=1)
        np.testing.assert_allclose(shares, 1.0)
        pd.testing.assert_frame_equal(spectral.descriptors(signal * 2.5), first, check_exact=False, atol=1e-9)

    def test_spectral_table_matches_version_3(self):
        table = pd.read_csv(base.RAIZ / "dados" / "pbit_spectral_features_v3.csv")
        self.assertEqual(table["experimento_idx"].tolist(), self.df["experimento_idx"].tolist())
        self.assertTrue(np.isfinite(table.values).all())

    def test_mahalanobis_flags_a_distant_cycle(self):
        data = np.random.default_rng(0).normal(0, 1, size=(300, 5))
        detector = generalization.MahalanobisDetector().fit(data)
        scores = detector.score(np.vstack([np.zeros(5), np.full(5, 8.0)]))
        self.assertGreater(scores[1], scores[0])

    def test_label_noise_changes_only_what_it_should(self):
        conditions = np.array(["N01"] * 100 + ["F01"] * 50 + ["N02"] * 10)
        train = np.ones(len(conditions), bool)
        mask = noise.corrupt_n01_rows(conditions, train, 0.05, np.random.default_rng(0))
        self.assertEqual(mask.sum(), 105)
        self.assertTrue(mask[:100].all())
        self.assertFalse(mask[150:].any())
        y = model.make_target(conditions)
        flipped = noise.flip_failure_labels(y, train, 0.10, np.random.default_rng(0))
        self.assertEqual(y.sum() - flipped.sum(), 5)
        self.assertTrue((flipped[y == 0] == 0).all())
        self.assertEqual(noise.corrupt_n01_rows(conditions, train, 0.0, np.random.default_rng(0)).sum(), 100)

    def test_forest_predicts_sequentially_and_restores_workers(self):
        X, y = self.sets["D"].values[:200], model.make_target(self.df["condicao"].values)[:200]
        y[0] = 1 - y[0]
        forest = model.make_model("forest", model.GRIDS["forest"][0], 0).fit(X, y)
        forest.predict_proba(X)
        self.assertEqual(forest.n_jobs, -1)

    def test_band_features_place_a_tone_in_the_right_band(self):
        t = np.arange(123) / 100.0
        line = 20  # 20 * 100 / 123 = 16.26 Hz, inside the 14 to 18 Hz band
        signal = 2.0 + 0.1 * np.sin(2 * np.pi * (line * 100 / 123) * t)
        table = hz.band_features(signal[None, :])
        self.assertGreater(table["share_14_18hz"].iloc[0], 0.99)
        self.assertAlmostEqual(table["peak_frequency_5_30hz"].iloc[0], line * 100 / 123, places=9)
        self.assertEqual(hz.full_spectrum(signal[None, :]).shape, (1, 61))

    def test_training_is_reproducible(self):
        X, y = self.sets["D"].values, model.make_target(self.df["condicao"].values)
        for name in model.GRIDS:
            first = model.make_model(name, model.GRIDS[name][0], 7).fit(X, y).predict_proba(X)
            second = model.make_model(name, model.GRIDS[name][0], 7).fit(X, y).predict_proba(X)
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
