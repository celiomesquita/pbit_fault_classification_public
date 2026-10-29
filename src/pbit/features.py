"""Feature extraction for the PBIT long/tidy dataset.

Each public function takes a 1-D NumPy array *x* of ordered measurements for
a single PBIT cycle and returns a flat ``dict[str, float]``.  The top-level
``extract_features`` function applies all families to every experiment in the
long DataFrame and returns a wide DataFrame with one row per experiment.

Feature name prefixes
---------------------
wc_   whole-cycle statistics
ep_   endpoint / reversal geometry
seg*_ per-segment descriptors  (8 segments)
sc_   segment contrast (outbound vs. return)
d1_   first-difference (velocity) dynamics
d2_   second-difference (acceleration) dynamics
roll*_ rolling-statistic summaries  (windows 5 and 15)
ls_   local-shape descriptors (peaks / valleys)
tr_   trend descriptors
fq_   frequency descriptors
mq_   missingness / quality indicators
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    valid = x[~np.isnan(x)]
    return float(np.sqrt(np.mean(valid ** 2))) if len(valid) > 0 else np.nan


def _energy(x: np.ndarray) -> float:
    valid = x[~np.isnan(x)]
    return float(np.sum(valid ** 2)) if len(valid) > 0 else np.nan


def _linear_slope(x: np.ndarray) -> float:
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() < 2:
        return np.nan
    t = np.arange(len(x), dtype=float)[valid_mask]
    result = linregress(t, x[valid_mask])
    return float(result.slope)


def _segment_signal(x: np.ndarray, n: int = 8) -> list[np.ndarray]:
    """Split x into n roughly equal contiguous segments."""
    return [arr for arr in np.array_split(x, n)]


def _rolling_slope(s: pd.Series, w: int) -> np.ndarray:
    """Rolling linear slope using fixed time indices [0, …, w-1]."""
    t = np.arange(w, dtype=float)
    t_centered = t - t.mean()
    t_var = float(np.sum(t_centered ** 2))

    def _slope(y: np.ndarray) -> float:
        if np.any(np.isnan(y)):
            return np.nan
        cov = float(np.dot(t_centered, y - y.mean()))
        return cov / t_var

    return s.rolling(w, min_periods=w).apply(_slope, raw=True).values


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------

def whole_cycle_features(x: np.ndarray) -> dict[str, float]:
    """Family 1: Whole-cycle statistics."""
    if (~np.isnan(x)).sum() == 0:
        keys = ["mean", "std", "min", "max", "range", "median",
                "q25", "q75", "iqr", "rms", "energy", "trapz"]
        return {f"wc_{k}": np.nan for k in keys}

    q25, q50, q75 = np.nanpercentile(x, [25, 50, 75])
    valid = x[~np.isnan(x)]
    return {
        "wc_mean":   float(np.nanmean(x)),
        "wc_std":    float(np.nanstd(x)),
        "wc_min":    float(np.nanmin(x)),
        "wc_max":    float(np.nanmax(x)),
        "wc_range":  float(np.nanmax(x) - np.nanmin(x)),
        "wc_median": float(q50),
        "wc_q25":    float(q25),
        "wc_q75":    float(q75),
        "wc_iqr":    float(q75 - q25),
        "wc_rms":    _rms(x),
        "wc_energy": _energy(x),
        "wc_trapz":  float(np.trapezoid(valid)),
    }


def endpoint_features(x: np.ndarray) -> dict[str, float]:
    """Family 2: Endpoint and reversal geometry."""
    keys = ["start", "end", "delta", "abs_delta",
            "peak", "valley", "peak_pos", "valley_pos",
            "peak_pos_norm", "valley_pos_norm"]
    empty = {f"ep_{k}": np.nan for k in keys}

    valid_idx = np.where(~np.isnan(x))[0]
    if len(valid_idx) < 2:
        return empty

    n = len(x)
    start_val = float(x[valid_idx[0]])
    end_val   = float(x[valid_idx[-1]])
    peak_pos  = int(np.nanargmax(x))
    valley_pos = int(np.nanargmin(x))

    return {
        "ep_start":          start_val,
        "ep_end":            end_val,
        "ep_delta":          end_val - start_val,
        "ep_abs_delta":      abs(end_val - start_val),
        "ep_peak":           float(np.nanmax(x)),
        "ep_valley":         float(np.nanmin(x)),
        "ep_peak_pos":       float(peak_pos),
        "ep_valley_pos":     float(valley_pos),
        "ep_peak_pos_norm":  peak_pos / n,
        "ep_valley_pos_norm": valley_pos / n,
    }


def _segment_stats(seg: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute per-segment descriptors for one segment array."""
    stat_keys = ["mean", "std", "min", "max", "range",
                 "median", "rms", "energy", "slope", "start", "end", "delta"]
    if (~np.isnan(seg)).sum() == 0:
        return {f"{prefix}_{k}": np.nan for k in stat_keys}

    valid_idx = np.where(~np.isnan(seg))[0]
    start_val = float(seg[valid_idx[0]])
    end_val   = float(seg[valid_idx[-1]])

    return {
        f"{prefix}_mean":   float(np.nanmean(seg)),
        f"{prefix}_std":    float(np.nanstd(seg)),
        f"{prefix}_min":    float(np.nanmin(seg)),
        f"{prefix}_max":    float(np.nanmax(seg)),
        f"{prefix}_range":  float(np.nanmax(seg) - np.nanmin(seg)),
        f"{prefix}_median": float(np.nanmedian(seg)),
        f"{prefix}_rms":    _rms(seg),
        f"{prefix}_energy": _energy(seg),
        f"{prefix}_slope":  _linear_slope(seg),
        f"{prefix}_start":  start_val,
        f"{prefix}_end":    end_val,
        f"{prefix}_delta":  end_val - start_val,
    }


def segment_features(x: np.ndarray, n_segments: int = 8) -> dict[str, float]:
    """Family 3: Direction-aware segment summaries (8 segments × 12 descriptors)."""
    segments = _segment_signal(x, n_segments)
    features: dict[str, float] = {}
    for i, seg in enumerate(segments, start=1):
        features.update(_segment_stats(seg, f"seg{i}"))
    return features


def segment_contrast_features(x: np.ndarray) -> dict[str, float]:
    """Family 4: Segment contrast — outbound vs. return and mirrored-segment differences."""
    segments = _segment_signal(x, 8)
    outbound      = np.concatenate(segments[:4])
    return_stroke = np.concatenate(segments[4:])

    ob_mean    = float(np.nanmean(outbound))
    ret_mean   = float(np.nanmean(return_stroke))
    ob_energy  = _energy(outbound)
    ret_energy = _energy(return_stroke)

    seg_energies = [_energy(s) for s in segments]
    seg_ranges   = [
        float(np.nanmax(s) - np.nanmin(s)) if (~np.isnan(s)).sum() > 0 else np.nan
        for s in segments
    ]

    valid_energies = [e for e in seg_energies if not np.isnan(e) and e >= 0]
    valid_ranges   = [r for r in seg_ranges   if not np.isnan(r) and r >= 0]
    max_e = max(valid_energies, default=np.nan)
    min_e = min((e for e in valid_energies if e > 0), default=np.nan)
    max_r = max(valid_ranges, default=np.nan)
    min_r = min((r for r in valid_ranges   if r > 0), default=np.nan)

    features: dict[str, float] = {
        "sc_ob_mean":         ob_mean,
        "sc_ret_mean":        ret_mean,
        "sc_mean_diff":       ob_mean - ret_mean,
        "sc_ob_energy":       ob_energy,
        "sc_ret_energy":      ret_energy,
        "sc_energy_diff":     ob_energy - ret_energy,
        "sc_energy_ratio":    ob_energy / ret_energy if ret_energy and ret_energy != 0 else np.nan,
        "sc_max_energy_ratio": max_e / min_e if (not np.isnan(min_e) and min_e != 0) else np.nan,
        "sc_max_range_ratio":  max_r / min_r if (not np.isnan(min_r) and min_r != 0) else np.nan,
    }

    # Mirrored-segment differences: 1/8, 2/7, 3/6, 4/5
    for (i, j), name in zip([(0, 7), (1, 6), (2, 5), (3, 4)],
                             ["1_8",  "2_7",  "3_6",  "4_5"]):
        features[f"sc_mirror_{name}_mean_diff"]   = float(np.nanmean(segments[i])) - float(np.nanmean(segments[j]))
        features[f"sc_mirror_{name}_energy_diff"] = _energy(segments[i]) - _energy(segments[j])

    return features


def first_diff_features(x: np.ndarray) -> dict[str, float]:
    """Family 5: First-difference (velocity) dynamics."""
    keys = ["mean_slope", "slope_std", "max_pos_slope", "max_neg_slope",
            "mean_abs_slope", "max_abs_slope", "sign_change_rate"]
    empty = {f"d1_{k}": np.nan for k in keys}

    valid_x = x[~np.isnan(x)]
    if len(valid_x) < 2:
        return empty

    d1 = np.diff(valid_x)
    sign_changes = int(np.sum(np.diff(np.sign(d1)) != 0)) if len(d1) > 1 else 0

    return {
        "d1_mean_slope":      float(np.mean(d1)),
        "d1_slope_std":       float(np.std(d1)),
        "d1_max_pos_slope":   float(np.max(d1)),
        "d1_max_neg_slope":   float(np.min(d1)),
        "d1_mean_abs_slope":  float(np.mean(np.abs(d1))),
        "d1_max_abs_slope":   float(np.max(np.abs(d1))),
        "d1_sign_change_rate": sign_changes / len(d1) if len(d1) > 0 else np.nan,
    }


def second_diff_features(x: np.ndarray) -> dict[str, float]:
    """Family 6: Second-difference (acceleration) dynamics."""
    keys = ["mean_accel", "accel_std", "max_abs_accel", "sign_change_rate"]
    empty = {f"d2_{k}": np.nan for k in keys}

    valid_x = x[~np.isnan(x)]
    if len(valid_x) < 3:
        return empty

    d2 = np.diff(valid_x, n=2)
    sign_changes = int(np.sum(np.diff(np.sign(d2)) != 0)) if len(d2) > 1 else 0

    return {
        "d2_mean_accel":      float(np.mean(d2)),
        "d2_accel_std":       float(np.std(d2)),
        "d2_max_abs_accel":   float(np.max(np.abs(d2))),
        "d2_sign_change_rate": sign_changes / len(d2) if len(d2) > 0 else np.nan,
    }


def rolling_features(x: np.ndarray, windows: list[int] | None = None) -> dict[str, float]:
    """Families 7–8: Moving statistics and their per-profile summaries.

    Profiles: mean, std, range, rms, slope — for each window in *windows*.
    For each profile, reports: mean, std, max, min, range, argmax, argmin.
    """
    if windows is None:
        windows = [5, 15]

    features: dict[str, float] = {}
    s = pd.Series(x, dtype=float)

    for w in windows:
        roll_mean  = s.rolling(w, min_periods=1).mean().values
        roll_std   = s.rolling(w, min_periods=w).std().values
        roll_min_r = s.rolling(w, min_periods=1).min().values
        roll_max_r = s.rolling(w, min_periods=1).max().values
        roll_range = roll_max_r - roll_min_r
        roll_rms   = s.pow(2).rolling(w, min_periods=1).mean().apply(np.sqrt).values
        roll_slope = _rolling_slope(s, w)

        for stat_name, arr in [
            ("mean",  roll_mean),
            ("std",   roll_std),
            ("range", roll_range),
            ("rms",   roll_rms),
            ("slope", roll_slope),
        ]:
            prefix = f"roll{w}_{stat_name}"
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                for sfx in ["mean", "std", "max", "min", "range", "argmax", "argmin"]:
                    features[f"{prefix}_{sfx}"] = np.nan
            else:
                features[f"{prefix}_mean"]   = float(np.nanmean(arr))
                features[f"{prefix}_std"]    = float(np.nanstd(arr))
                features[f"{prefix}_max"]    = float(np.nanmax(arr))
                features[f"{prefix}_min"]    = float(np.nanmin(arr))
                features[f"{prefix}_range"]  = float(np.nanmax(arr) - np.nanmin(arr))
                features[f"{prefix}_argmax"] = float(np.nanargmax(arr))
                features[f"{prefix}_argmin"] = float(np.nanargmin(arr))

    return features


def local_shape_features(x: np.ndarray) -> dict[str, float]:
    """Family 9: Local-shape descriptors — peaks, valleys, and their segment concentration."""
    base_keys = ["peak_count", "valley_count",
                 "peak_prom_mean", "peak_prom_max",
                 "valley_prom_mean", "valley_prom_max"]
    seg_keys  = [f"seg{i}_peak_frac" for i in range(1, 9)]
    all_keys  = base_keys + seg_keys

    valid_x   = x[~np.isnan(x)]
    valid_pos = np.where(~np.isnan(x))[0]

    if len(valid_x) < 3:
        return {f"ls_{k}": np.nan for k in all_keys}

    peaks,   peak_props   = sp_signal.find_peaks(valid_x,  prominence=0)
    valleys, valley_props = sp_signal.find_peaks(-valid_x, prominence=0)

    peak_proms   = peak_props.get("prominences",   np.array([]))
    valley_proms = valley_props.get("prominences", np.array([]))

    features: dict[str, float] = {
        "ls_peak_count":       float(len(peaks)),
        "ls_valley_count":     float(len(valleys)),
        "ls_peak_prom_mean":   float(np.mean(peak_proms))   if len(peak_proms)   > 0 else 0.0,
        "ls_peak_prom_max":    float(np.max(peak_proms))    if len(peak_proms)   > 0 else 0.0,
        "ls_valley_prom_mean": float(np.mean(valley_proms)) if len(valley_proms) > 0 else 0.0,
        "ls_valley_prom_max":  float(np.max(valley_proms))  if len(valley_proms) > 0 else 0.0,
    }

    # Map peak positions back to original signal coordinates
    n = len(x)
    seg_boundaries = np.linspace(0, n, 9).astype(int)
    peak_orig_pos  = valid_pos[peaks] if len(peaks) > 0 else np.array([], dtype=int)

    for seg_i in range(8):
        lo, hi = seg_boundaries[seg_i], seg_boundaries[seg_i + 1]
        count  = int(np.sum((peak_orig_pos >= lo) & (peak_orig_pos < hi)))
        features[f"ls_seg{seg_i+1}_peak_frac"] = count / len(peaks) if len(peaks) > 0 else 0.0

    return features


def trend_features(x: np.ndarray) -> dict[str, float]:
    """Family 10: Trend descriptors — global and stroke-split linear regression."""
    keys = ["slope", "intercept", "r2", "resid_std", "ob_slope", "ret_slope"]
    empty = {f"tr_{k}": np.nan for k in keys}

    n = len(x)
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() < 2:
        return empty

    t = np.arange(n, dtype=float)[valid_mask]
    y = x[valid_mask]
    res = linregress(t, y)
    predicted = res.slope * t + res.intercept

    features: dict[str, float] = {
        "tr_slope":     float(res.slope),
        "tr_intercept": float(res.intercept),
        "tr_r2":        float(res.rvalue ** 2),
        "tr_resid_std": float(np.std(y - predicted)),
    }

    mid = n // 2
    for part, lo, hi in [("ob", 0, mid), ("ret", mid, n)]:
        mask_part = valid_mask[lo:hi]
        t_part    = np.arange(hi - lo, dtype=float)[mask_part]
        y_part    = x[lo:hi][mask_part]
        if len(y_part) >= 2:
            features[f"tr_{part}_slope"] = float(linregress(t_part, y_part).slope)
        else:
            features[f"tr_{part}_slope"] = np.nan

    return features


def frequency_features(x: np.ndarray) -> dict[str, float]:
    """Family 11: Frequency-domain descriptors via FFT."""
    keys = ["dominant_bin", "spectral_centroid", "spectral_bandwidth",
            "spectral_entropy", "hf_energy_ratio"]
    empty = {f"fq_{k}": np.nan for k in keys}

    valid_x = x[~np.isnan(x)]
    n = len(valid_x)
    if n < 4:
        return empty

    fft_vals  = np.abs(np.fft.rfft(valid_x))
    freqs     = np.fft.rfftfreq(n)
    power     = fft_vals ** 2
    total_pwr = float(np.sum(power))
    if total_pwr == 0:
        return empty

    norm_power    = power / total_pwr
    dominant_bin  = int(np.argmax(power[1:]) + 1)   # skip DC bin
    centroid      = float(np.sum(freqs * norm_power))
    bandwidth     = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * norm_power)))

    p_nz = norm_power[norm_power > 0]
    entropy = float(-np.sum(p_nz * np.log(p_nz)))

    cutoff = max(1, len(freqs) * 3 // 4)
    hf_ratio = float(np.sum(power[cutoff:]) / total_pwr)

    return {
        "fq_dominant_bin":       float(dominant_bin),
        "fq_spectral_centroid":  centroid,
        "fq_spectral_bandwidth": bandwidth,
        "fq_spectral_entropy":   entropy,
        "fq_hf_energy_ratio":    hf_ratio,
    }


def missingness_features(x: np.ndarray) -> dict[str, float]:
    """Family 12: Missingness and quality indicators."""
    n          = len(x)
    is_missing = np.isnan(x)
    n_missing  = int(is_missing.sum())

    max_run = cur_run = 0
    for m in is_missing:
        if m:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    return {
        "mq_valid_count":          n - n_missing,
        "mq_missing_count":        n_missing,
        "mq_missing_frac":         n_missing / n,
        "mq_longest_missing_run":  max_run,
        "mq_highly_incomplete":    int(n_missing / n > 0.5),
    }


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all 12 feature families for every experiment in *df*.

    Parameters
    ----------
    df:
        Long/tidy PBIT dataset as loaded by ``load_long_dataset``.
        Must contain columns: Experiment_Index, Serial_Number,
        Serial_Run_Index, Fault_Mode, Severity_Index, Sample_Index,
        Measurement.

    Returns
    -------
    pd.DataFrame
        One row per experiment.  First columns are the experiment metadata;
        remaining columns are numeric features.
    """
    df_sorted = df.sort_values(["Experiment_Index", "Sample_Index"])
    records: list[dict] = []

    for exp_idx, group in df_sorted.groupby("Experiment_Index", sort=True):
        x = group["Measurement"].values.astype(float)

        row: dict = {
            "Experiment_Index": exp_idx,
            "Serial_Number":    group["Serial_Number"].iloc[0],
            "Serial_Run_Index": int(group["Serial_Run_Index"].iloc[0]),
            "Fault_Mode":       group["Fault_Mode"].iloc[0],
            "Severity_Index":   int(group["Severity_Index"].iloc[0]),
        }

        row.update(whole_cycle_features(x))
        row.update(endpoint_features(x))
        row.update(segment_features(x))
        row.update(segment_contrast_features(x))
        row.update(first_diff_features(x))
        row.update(second_diff_features(x))
        row.update(rolling_features(x))
        row.update(local_shape_features(x))
        row.update(trend_features(x))
        row.update(frequency_features(x))
        row.update(missingness_features(x))

        records.append(row)

    return pd.DataFrame(records)


def feature_names() -> list[str]:
    """Return the ordered list of feature column names (without metadata columns).

    Useful for checking which columns are features vs. metadata.
    """
    dummy = np.ones(123, dtype=float)
    row = {}
    row.update(whole_cycle_features(dummy))
    row.update(endpoint_features(dummy))
    row.update(segment_features(dummy))
    row.update(segment_contrast_features(dummy))
    row.update(first_diff_features(dummy))
    row.update(second_diff_features(dummy))
    row.update(rolling_features(dummy))
    row.update(local_shape_features(dummy))
    row.update(trend_features(dummy))
    row.update(frequency_features(dummy))
    row.update(missingness_features(dummy))
    return list(row.keys())
