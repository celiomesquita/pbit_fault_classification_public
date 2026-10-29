"""E3b: restricted wild and jackknife unit inference for a ratio of totals.

The ratio sum(D)/sum(n) is the least-squares intercept of cycle-level differences,
clustered by SN, so the formulas follow MacKinnon, Nielsen and Webb (2023) with k=1.
Only units with positive exposure enter: they are the clusters of that regression.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

METHODS = ["wcr_c", "wcr_v", "wcu_b", "cv3_t"]
BOOTSTRAP_METHODS = METHODS[:3]
TINY = 1e-14


def pivot(numerator, se):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numerator / se
    return np.where((numerator == 0) & (se == 0), 0, t)


def ratio_parts(d, n):
    """Estimate, CV1, CV3 and leave-one-SN-out estimates for rows of d."""
    d = np.atleast_2d(np.asarray(d, dtype=float))
    n = np.asarray(n, dtype=float)
    g, total = len(n), n.sum()
    theta = d.sum(axis=1) / total
    residual = d - theta[:, None]*n
    se1 = np.sqrt(g/(g-1)*np.sum(residual**2, axis=1)) / total
    loo = (d.sum(axis=1)[:, None] - d) / (total - n)
    se3 = np.sqrt((g-1)/g*np.sum((loo - theta[:, None])**2, axis=1))
    return theta, se1, se3, loo


def rademacher(rng, rows, resamples, units):
    return rng.choice(np.array([-1.0, 1.0]), size=(rows, resamples, units))


def bootstrap_statistics(scores, n, v, variance):
    """Wild bootstrap t* from cluster scores (R,G) and weights v (R,B,G)."""
    g, total = len(n), n.sum()
    star = v * scores[:, None, :]
    star_sum = star.sum(axis=2)
    delta = star_sum / total
    if variance == "cv1":
        squares = np.sum((star - delta[..., None]*n)**2, axis=2)
        se = np.sqrt(g/(g-1)*squares) / total
    else:
        loo = (star_sum[..., None] - star) / (total - n)
        se = np.sqrt((g-1)/g*np.sum((loo - delta[..., None])**2, axis=2))
    return pivot(delta, se), se


def tail_p(t_star, observed):
    """One-sided bootstrap p-values (upper, lower), with the +1 correction."""
    count = t_star.shape[-1]
    reference = observed[..., None]
    # Exact ties (e.g. the identity draw under WCR) must survive rounding.
    slack = 1e-10*np.maximum(1, np.abs(np.where(np.isfinite(reference), reference, 0)))
    upper = (1 + np.sum(t_star >= reference - slack, axis=-1)) / (count + 1)
    lower = (1 + np.sum(t_star <= reference + slack, axis=-1)) / (count + 1)
    return upper, lower


def method_se(method, se1, se3):
    return se1 if method == "wcr_c" else se3


def one_sided(method, d, n, hs, v):
    """Upper/lower p-values (R,H) for nulls theta=h, with shared weights v."""
    d = np.atleast_2d(np.asarray(d, dtype=float))
    n = np.asarray(n, dtype=float)
    hs = np.atleast_2d(np.asarray(hs, dtype=float))
    if hs.shape[0] == 1:
        hs = np.repeat(hs, len(d), axis=0)
    theta, se1, se3, loo = ratio_parts(d, n)
    se = method_se(method, se1, se3)
    observed = pivot(theta[:, None] - hs, se[:, None])
    if method == "cv3_t":
        df = len(n) - 1
        return student_t.sf(observed, df), student_t.cdf(observed, df), np.zeros(len(d))
    if method == "wcu_b":
        scores = n.sum()*(theta[:, None] - loo)
        t_star, se_star = bootstrap_statistics(scores, n, v, "cv3")
        upper, lower = tail_p(t_star[:, None, :], observed)
        return upper, lower, (se_star <= TINY).mean(axis=1)
    upper = np.empty_like(hs)
    lower = np.empty_like(hs)
    bad = np.zeros(len(d))
    variance = "cv1" if method == "wcr_c" else "cv3"
    for j in range(hs.shape[1]):
        restricted = d - hs[:, j:j+1]*n
        t_star, se_star = bootstrap_statistics(restricted, n, v, variance)
        upper[:, j], lower[:, j] = tail_p(t_star, observed[:, j])
        bad += (se_star <= TINY).mean(axis=1)
    return upper, lower, bad/hs.shape[1]


def two_sided(upper, lower):
    return np.minimum(1, 2*np.minimum(upper, lower))


def abstentions(method, d, n):
    _, se1, se3, _ = ratio_parts(d, n)
    se = method_se(method, se1, se3)
    return ~np.isfinite(se) | (se <= TINY)


def confidence_set(method, d, n, v, level=0.95, iterations=40, max_steps=40):
    """Interval hull of {h: two-sided p(h) > 1-level} by outward search and bisection.

    The equal-tailed p-value is a step function of h for bootstrap methods; the
    returned bounds are within (2**-iterations) of the initial bracket width.
    """
    d = np.atleast_2d(np.asarray(d, dtype=float))
    alpha = 1 - level
    theta, se1, se3, _ = ratio_parts(d, n)
    scale = np.maximum(method_se(method, se1, se3), TINY)

    def accepted(h):
        upper, lower, _ = one_sided(method, d, n, h[:, None], v)
        return two_sided(upper[:, 0], lower[:, 0]) > alpha

    bounds = []
    for sign in (-1.0, 1.0):
        inside = theta.copy()
        step = scale.copy()
        outside = theta + sign*step
        open_rows = accepted(outside)
        for _ in range(max_steps):
            if not open_rows.any():
                break
            inside = np.where(open_rows, outside, inside)
            step = np.where(open_rows, 2*step, step)
            outside = np.where(open_rows, theta + sign*step, outside)
            open_rows = open_rows & accepted(outside)
        unbounded = open_rows
        for _ in range(iterations):
            middle = (inside + outside) / 2
            keep = accepted(middle)
            inside = np.where(keep, middle, inside)
            outside = np.where(keep, outside, middle)
        bounds.append(np.where(unbounded, sign*np.inf, (inside + outside)/2))
    return np.stack(bounds, axis=1)
