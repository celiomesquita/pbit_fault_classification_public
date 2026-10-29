"""Inner-only normal/fault policy selection for E10."""
from __future__ import annotations

import numpy as np

CLASSES = np.array(["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"])
PRIMARY = tuple(CLASSES[2:7])
TARGETS = (.85, .90, .95)
BUDGETS = (.30, .40, .50)
EPS = 1e-12


def predict(probability, threshold, promotion=False):
    probability = np.asarray(probability)
    alarm = probability[:, 2:].sum(axis=1) >= threshold
    if promotion:
        alarm |= probability.argmax(axis=1) >= 2
    normal = CLASSES[:2][probability[:, :2].argmax(axis=1)]
    fault = CLASSES[2:][probability[:, 2:].argmax(axis=1)]
    return np.where(alarm, fault, normal)


def curve(y, probability, promotion=False):
    """All distinct decisions, retaining >= ties, with cumulative class counts."""
    y, probability = np.asarray(y), np.asarray(probability)
    score = probability[:, 2:].sum(axis=1)
    unique, inverse = np.unique(score, return_inverse=True)
    thresholds = np.r_[np.inf, unique[::-1]]
    if unique[0] > 0:
        thresholds = np.r_[thresholds, 0.]
    initial = probability.argmax(axis=1) >= 2 if promotion else np.zeros(len(y), dtype=bool)
    counts, totals = [], []
    for cls in CLASSES:
        actual = y == cls
        increments = np.bincount(inverse, weights=(actual & ~initial).astype(int), minlength=len(unique))
        detected = np.r_[0., np.cumsum(increments[::-1])] + np.count_nonzero(actual & initial)
        if len(thresholds) > len(detected):
            detected = np.r_[detected, detected[-1]]
        counts.append(detected)
        totals.append(int(actual.sum()))
    counts, totals = np.array(counts).T, np.array(totals)
    recalls = np.divide(counts[:, 2:7], totals[2:7],
                        out=np.full((len(thresholds), 5), np.nan), where=totals[2:7] > 0)
    normals = totals[:2].sum()
    fa0 = counts[:, :2].sum(axis=1) / normals if normals else np.full(len(thresholds), np.nan)
    defined = bool(normals and (totals[2:7] > 0).all())
    return dict(thresholds=thresholds, recalls=recalls, FA0=fa0, defined=defined,
                minimum=recalls.min(axis=1), mean=recalls.mean(axis=1), totals=totals)


def select_curve(values, target=.90, budget=.40):
    """Minimise FA0 subject to each primary-class recall and the alarm budget."""
    thresholds, fa0 = values["thresholds"], values["FA0"]
    within = np.flatnonzero(np.isfinite(fa0) & (fa0 <= budget + EPS))
    feasible = within[values["minimum"][within] >= target - EPS] if values["defined"] else []
    if len(feasible):
        selected = min(feasible, key=lambda i: (fa0[i], -values["minimum"][i],
                                               -values["mean"][i], -thresholds[i]))
        status = "feasible"
    elif len(within) and values["defined"]:
        selected = min(within, key=lambda i: (-values["minimum"][i], fa0[i],
                                              -values["mean"][i], -thresholds[i]))
        status = "recall_infeasible"
    else:
        selected = 0
        status = "endpoint_undefined" if not values["defined"] else "alarm_budget_infeasible"
    return dict(threshold=float(thresholds[selected]), target=target, budget=budget,
                feasible=status == "feasible", status=status, FA0=float(fa0[selected]),
                min_recall=float(values["minimum"][selected]),
                mean_recall=float(values["mean"][selected]),
                recalls={c: float(r) for c, r in zip(PRIMARY, values["recalls"][selected])},
                denominators={str(c): int(n) for c, n in zip(CLASSES, values["totals"])})


def rank(selection, trial=0):
    """Order configurations with the same policy-selection objective and tie rules."""
    if selection["feasible"]:
        return (0, selection["FA0"], -selection["min_recall"], -selection["mean_recall"], trial)
    if selection["status"] == "recall_infeasible":
        return (1, -selection["min_recall"], selection["FA0"], -selection["mean_recall"], trial)
    return (2, trial)


def evaluate(y, prediction):
    """Detection per class is separate from exact identity and critical severity."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    alarm = np.isin(prediction, CLASSES[2:])
    normals, critical = np.isin(y, CLASSES[:2]), np.isin(y, ["F03", "F06"])
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else np.nan
    result = dict(FA0=ratio((normals & alarm).sum(), normals.sum()),
                  CMR=ratio((critical & ~alarm).sum(), critical.sum()),
                  R3=ratio((critical & np.isin(prediction, ["F03", "F06"])).sum(), critical.sum()),
                  normal_n=int(normals.sum()), critical_n=int(critical.sum()))
    for cls in CLASSES[:2]:
        actual = y == cls
        result[f"n_{cls}"] = int(actual.sum())
        result[f"FA0_{cls}"] = ratio((actual & alarm).sum(), actual.sum())
    present = np.union1d(y, prediction)
    f1 = []
    for cls in present:
        f1.append(ratio(2 * ((y == cls) & (prediction == cls)).sum(),
                        (y == cls).sum() + (prediction == cls).sum()))
    result["macro_F1"] = float(np.mean(f1))
    recalls = []
    for cls in CLASSES[2:]:
        actual = y == cls
        n = int(actual.sum())
        result[f"n_{cls}"] = n
        result[f"detect_{cls}"] = ratio((actual & alarm).sum(), n)
        result[f"exact_{cls}"] = ratio((actual & (prediction == cls)).sum(), n)
        if cls in PRIMARY:
            recalls.append(result[f"detect_{cls}"])
    result["min_recall"] = float(np.min(recalls))
    result["mean_recall"] = float(np.mean(recalls))
    return result
