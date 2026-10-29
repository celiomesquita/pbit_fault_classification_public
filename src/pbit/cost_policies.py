"""Severity costs and inner-only selection of a cost-sensitive policy."""
from __future__ import annotations

import numpy as np

from pbit.modeling import FAULT_ORDER, SEVERITY_MAP

CLASSES = np.array(FAULT_ORDER)
SEVERITY = np.array([SEVERITY_MAP[c] for c in CLASSES])


def cost_matrix(kappa: float) -> np.ndarray:
    """Rows are true classes; columns are decisions."""
    difference = SEVERITY[:, None] - SEVERITY[None, :]
    cost = np.where(difference > 0, kappa * difference, -difference).astype(float)
    cost[difference == 0] = 1
    np.fill_diagonal(cost, 0)
    return cost


def minimum_cost_predictions(probability, kappa):
    return CLASSES[(np.asarray(probability) @ cost_matrix(kappa)).argmin(axis=1)]


def select_cost_policy(y, candidates, budget, metrics):
    """Candidates map kappa to held-inner-unit predictions; no outer labels."""
    records, eligible = [], []
    for kappa, prediction in sorted(candidates.items()):
        result = metrics(y, prediction)
        feasible = np.isfinite(result["CMR"]) and np.isfinite(result["FA0"]) and result["FA0"] <= budget
        records.append(dict(kappa=kappa, budget=budget, eligible=bool(feasible), **result))
        if feasible:
            eligible.append((result["CMR"], result["FA0"], kappa))
    return (min(eligible)[2] if eligible else None), records
