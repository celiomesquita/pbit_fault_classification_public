import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e12_geometry import nearest, scale_from_training, shape_rows, unit_weighted_quantiles


def test_reference_cannot_be_same_serial_number():
    distance, index = nearest([[0.], [10.]], [[0.], [2.], [10.]], ["A", "C"], ["A", "B", "C"])
    np.testing.assert_allclose(distance, [2, 8])
    np.testing.assert_array_equal(index, [1, 1])


def test_missing_reference_is_not_zero_distance():
    distance, index = nearest([[0.]], [[0.]], ["A"], ["A"])
    assert np.isnan(distance[0]) and index[0] == -1


def test_evaluation_values_do_not_set_imputation_or_scale():
    train = np.array([[0., 9.], [2., 9.], [4., 9.]])
    _, transformed, keep = scale_from_training(train, [[np.nan, 99.], [100., -50.]])
    np.testing.assert_array_equal(keep, [True, False])
    np.testing.assert_allclose(transformed[:, 0], [0, 49])


def test_shape_separates_level_from_pattern_and_handles_constant_signal():
    transformed, flat = shape_rows([[1, 2, 3], [12, 14, 16], [7, 7, 7]])
    np.testing.assert_allclose(transformed[0], transformed[1])
    np.testing.assert_allclose(transformed[2], 0)
    np.testing.assert_array_equal(flat, [False, False, True])


def test_unit_quantile_resists_duplicate_exposure_of_a_unit():
    a = unit_weighted_quantiles([0, 10], ["A", "B"])
    b = unit_weighted_quantiles([0]*100 + [10], ["A"]*100 + ["B"])
    assert np.isclose(a[1], b[1])
