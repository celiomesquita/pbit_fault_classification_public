"""Paired unit inference for a ratio of totals, used by the E3 simulation."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

METHODS = ["percentile", "pairs_t", "wild_rademacher", "wild_webb", "sign_swap", "bca"]
FAMILIES = [1, 3, 4, 8]
TAILS = np.array([0.025, 0.975, *[q for m in FAMILIES for q in (0.05 / m, 1 - 0.05 / m)]])


def ratio_se(d, n, total_units):
    theta = d.sum(axis=-1) / n.sum()
    residual = d - theta[..., None] * n
    se = np.sqrt(total_units / (total_units - 1) * np.sum(residual**2, axis=-1)) / n.sum()
    return theta, se, residual


def row_quantiles(samples, probabilities):
    """Linear quantiles with potentially different probabilities in each row."""
    sorted_samples = np.sort(samples, axis=1)
    position = probabilities * (samples.shape[1] - 1)
    low = np.floor(position).astype(int)
    high = np.ceil(position).astype(int)
    rows = np.arange(len(samples))[:, None]
    a, b = sorted_samples[rows, low], sorted_samples[rows, high]
    fraction = position - low
    return a + (b - a) * fraction


def paired_draws(d, n, units, weights):
    denom = weights @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.einsum("rbg,rg->rb", weights, d) / denom
        squares = (np.einsum("rbg,rg->rb", weights, d*d)
                   - 2*theta*np.einsum("rbg,rg->rb", weights, d*n)
                   + theta*theta*(weights @ (n*n)))
        se = np.sqrt(units/(units-1)*np.maximum(squares, 0)) / denom
    return theta, se


def pivot(numerator, se):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numerator / se
    return np.where((numerator == 0) & (se == 0), 0, t)


def bca_probabilities(d, n, units, estimate, samples, probabilities=TAILS):
    total = n.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        jack = (d.sum(axis=1)[:, None] - d) / (total - n)
    # Deleting a zero-exposure SN leaves the estimate unchanged.
    jack_mean = (jack.sum(axis=1) + (units-len(n))*estimate) / units
    difference = jack_mean[:, None] - jack
    inactive_difference = jack_mean - estimate
    ss = (difference**2).sum(axis=1) + (units-len(n))*inactive_difference**2
    cubes = (difference**3).sum(axis=1) + (units-len(n))*inactive_difference**3
    with np.errstate(divide="ignore", invalid="ignore"):
        acceleration = cubes / (6*ss**1.5)
    bias_probability = ((samples < estimate[:, None]).sum(axis=1)
                        + 0.5*(samples == estimate[:, None]).sum(axis=1)) / np.isfinite(samples).sum(axis=1)
    z0 = ndtri(np.clip(bias_probability, 1/(2*samples.shape[1]), 1-1/(2*samples.shape[1])))
    z = z0[:, None] + ndtri(probabilities)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = ndtr(z0[:, None] + z/(1-acceleration[:, None]*z))
    return adjusted


def infer_batch(d, n, units, rng, resamples=2000):
    """Independent internal resamples per row; n contains only positive exposures."""
    d = np.atleast_2d(np.asarray(d, dtype=float))
    n = np.asarray(n, dtype=float)
    r, g = d.shape
    assert units >= g and g > 1 and np.all(n > 0)
    estimate, se, residual = ratio_se(d, n, units)
    probabilities = np.r_[np.full(g, 1/units), (units-g)/units]
    weights = rng.multinomial(units, probabilities, size=(r, resamples))[..., :g]
    samples, boot_se = paired_draws(d, n, units, weights)
    centered = samples - estimate[:, None]
    finite = np.isfinite(samples)
    p_centered = (1 + np.sum(finite & (np.abs(centered) >= np.abs(estimate[:, None])), axis=1)) / (1 + finite.sum(axis=1))
    with np.errstate(invalid="ignore"):
        percentile_q = np.nanquantile(samples, TAILS, axis=1).T
    result = {}

    def save(name, bounds, pvalue, internal_bad, force_bad=None):
        bad = ~np.isfinite(bounds).all(axis=1) | ~np.isfinite(pvalue)
        if force_bad is not None:
            bad |= force_bad
        bounds[bad, ::2] = -np.inf
        bounds[bad, 1::2] = np.inf
        pvalue = np.where(bad, 1, pvalue)
        result[name] = dict(bounds=bounds, p=pvalue, abstain=bad, internal_bad=internal_bad)

    save("percentile", percentile_q, p_centered, 1-finite.mean(axis=1))

    def studentised(name, numerator, denominator):
        t = pivot(numerator, denominator)
        observed = pivot(estimate, se)
        valid = ~np.isnan(t)
        pvalue = (1+np.sum(valid & (np.abs(t) >= np.abs(observed[:, None])), axis=1)) / (1+valid.sum(axis=1))
        with np.errstate(invalid="ignore"):
            quantiles = np.nanquantile(t, TAILS, axis=1).T
            bounds = estimate[:, None] - se[:, None]*quantiles[:, np.arange(len(TAILS)) ^ 1]
        save(name, bounds, pvalue, 1-np.isfinite(t).mean(axis=1), se <= 1e-14)

    studentised("pairs_t", centered, boot_se)
    sign = rng.choice([-1.0, 1.0], size=(r, resamples, g))
    webb = rng.choice(np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)]),
                      size=(r, resamples, g))
    for name, multipliers in [("wild_rademacher", sign), ("wild_webb", webb)]:
        perturbation = np.einsum("rbg,rg->rb", multipliers, residual)
        mean = perturbation / n.sum()
        squares = (np.einsum("rbg,rg->rb", multipliers*multipliers, residual*residual)
                   - 2*mean*np.einsum("rbg,rg->rb", multipliers, residual*n)
                   + mean*mean*np.sum(n*n))
        wild_se = np.sqrt(units/(units-1)*np.maximum(squares, 0))/n.sum()
        studentised(name, mean, wild_se)

    signed_d = np.einsum("rbg,rg->rb", sign, d)
    signed_n = sign @ n
    denom = n.sum() - signed_n
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = (d.sum(axis=1)[:, None] - signed_d) / denom
    lower = np.where(denom == 0, -np.inf, inverse)
    upper = np.where(denom == 0, np.inf, inverse)
    bounds = np.empty((r, len(TAILS)))
    for j, q in enumerate(TAILS):
        bounds[:, j] = np.quantile(lower if j % 2 == 0 else upper, q, axis=1,
                                   method="lower" if j % 2 == 0 else "higher")
    p_high = (1+(signed_d >= d.sum(axis=1)[:, None]).sum(axis=1))/(resamples+1)
    p_low = (1+(signed_d <= d.sum(axis=1)[:, None]).sum(axis=1))/(resamples+1)
    save("sign_swap", bounds, np.minimum(1, 2*np.minimum(p_high, p_low)), (denom == 0).mean(axis=1))

    adjusted = bca_probabilities(d, n, units, estimate, samples)
    bounds = np.full_like(adjusted, np.nan)
    ok = np.isfinite(adjusted).all(axis=1) & finite.all(axis=1)
    bounds[ok] = row_quantiles(samples[ok], adjusted[ok])
    save("bca", bounds, p_centered, 1-finite.mean(axis=1))
    return estimate, result
