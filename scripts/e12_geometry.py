"""Training-reference geometry for the exploratory E12 diagnostic."""
import numpy as np
from scipy.spatial.distance import cdist


def shape_rows(values):
    values = np.asarray(values, dtype=float)
    std = values.std(axis=1)
    result = np.zeros_like(values)
    good = std > 1e-9
    result[good] = (values[good] - values[good].mean(axis=1, keepdims=True)) / std[good, None]
    return result, ~good


def scale_from_training(train, evaluation):
    train, evaluation = np.asarray(train, float).copy(), np.asarray(evaluation, float).copy()
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    train = np.where(np.isfinite(train), train, median)
    evaluation = np.where(np.isfinite(evaluation), evaluation, median)
    lo, hi = np.percentile(train, [25, 75], axis=0)
    scale = hi - lo
    keep = scale > 0
    return ((train[:, keep]-median[keep])/scale[keep],
            (evaluation[:, keep]-median[keep])/scale[keep], keep)


def nearest(query, reference, query_units=None, reference_units=None):
    """RMSE and first reference index; missing references remain undefined."""
    query, reference = np.asarray(query), np.asarray(reference)
    if not len(reference) or not query.shape[1]:
        return np.full(len(query), np.nan), np.full(len(query), -1, dtype=int)
    distances = cdist(query, reference, metric="euclidean") / np.sqrt(query.shape[1])
    if query_units is not None:
        same = np.asarray(query_units)[:, None] == np.asarray(reference_units)[None, :]
        distances[same] = np.inf
    indices = distances.argmin(axis=1)
    best = distances[np.arange(len(query)), indices]
    missing = ~np.isfinite(best)
    best[missing], indices[missing] = np.nan, -1
    return best, indices


def unit_weighted_quantiles(values, units, probabilities=(.05, .50, .95)):
    values, units = np.asarray(values), np.asarray(units)
    good = np.isfinite(values)
    values, units = values[good], units[good]
    if not len(values):
        return np.full(len(probabilities), np.nan)
    _, inverse, counts = np.unique(units, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    # Collapse equal distances so replicated observations do not move interpolation.
    values, groups = np.unique(values, return_inverse=True)
    weights = np.bincount(groups, weights=weights)
    positions = (np.cumsum(weights) - .5*weights) / weights.sum()
    return np.interp(probabilities, positions, values)
