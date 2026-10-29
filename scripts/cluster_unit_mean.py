"""E3c: unit-mean loss (L-bar) inference through the E3b wild/jackknife machinery.

L-bar is the simple mean of the per-unit loss differences x_g = D_g / n_g. It equals the
ratio of E3b with D_g replaced by x_g and every exposure replaced by one, so the four
E3b procedures apply unchanged to (x, 1).
"""

from __future__ import annotations

import numpy as np

from cluster_ratio_wild import METHODS as WILD_METHODS, abstentions, one_sided

# E3c identifier -> E3b procedure applied to (x, 1)
L_METHODS = {"L-WC": "wcr_c", "L-WV": "wcr_v", "L-WU": "wcu_b", "L-T": "cv3_t"}
assert set(L_METHODS.values()) == set(WILD_METHODS)


def unit_values(d, n):
    """Per-unit mean loss differences x_g = D_g / n_g for rows of D."""
    d = np.atleast_2d(np.asarray(d, dtype=float))
    n = np.asarray(n, dtype=float)
    assert np.all(n > 0)
    return d / n


def lbar_one_sided(method, d, n, hs, v):
    """Upper/lower p-values (R,H) for nulls Lbar = h, same weights v as the E3b routine."""
    x = unit_values(d, n)
    return one_sided(L_METHODS[method], x, np.ones(x.shape[1]), hs, v)


def lbar_abstentions(method, d, n):
    x = unit_values(d, n)
    return abstentions(L_METHODS[method], x, np.ones(x.shape[1]))


def lbar_estimate(d, n):
    return unit_values(d, n).mean(axis=1)
