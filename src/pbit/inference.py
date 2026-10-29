"""Paired cluster inference; seeds share the same resampled physical units."""

from __future__ import annotations

import numpy as np

from pbit.modeling import SEVERITY_MAP, FAULT_ORDER


def confusion_metrics(cm: np.ndarray) -> dict[str, np.ndarray]:
    severity = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
    total = cm.sum(axis=(-2, -1))
    truth = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)

    def ratio(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b > 0)

    critical = truth[..., severity == 3].sum(axis=-1)
    normal = truth[..., severity == 0].sum(axis=-1)
    misses = cm[..., severity == 3, :][..., severity == 0].sum(axis=(-2, -1))
    alarms = cm[..., severity == 0, :][..., severity > 0].sum(axis=(-2, -1))
    f1 = np.divide(2 * tp, truth + predicted, out=np.zeros_like(tp, dtype=float), where=truth + predicted > 0)
    present = truth + predicted > 0
    recall = np.divide(tp, truth, out=np.zeros_like(tp, dtype=float), where=truth > 0)
    return {
        "CMR": ratio(misses, critical),
        "FA0": ratio(alarms, normal),
        "DR": ratio((cm * (severity[None, :] < severity[:, None])).sum(axis=(-2, -1)), total),
        "macro_F1": ratio(f1.sum(axis=-1), present.sum(axis=-1)),
        "balanced_accuracy": ratio(recall.sum(axis=-1), (truth > 0).sum(axis=-1)),
    }


def paired_comparison(samples, estimate, margin, family_size=1, alpha=0.05):
    values = np.asarray(samples, dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid):
        raise ValueError("No paired bootstrap sample has a defined endpoint.")
    centered = valid - estimate
    p = (1 + np.sum(np.abs(centered) >= abs(estimate))) / (len(valid) + 1)
    tail = alpha / family_size
    lo, hi = np.quantile(valid, [tail, 1 - tail])
    ci = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(estimate), "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
        "p_centered": float(p), "equivalence_low": float(lo), "equivalence_high": float(hi),
        "equivalence_level": 1 - 2 * tail, "family_size": family_size,
        "margin": margin, "delta_min": float(max(abs(lo), abs(hi))),
        "equivalent": bool(lo > -margin and hi < margin),
        "discard_fraction": 1 - len(valid) / len(values),
    }


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return adjusted
