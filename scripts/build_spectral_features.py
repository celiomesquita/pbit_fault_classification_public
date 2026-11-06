"""Build per-cycle frequency descriptors for data version 3 from the raw 123-sample signals.

Each descriptor depends only on the cycle itself. Bands are fractions of the Nyquist frequency, so no result depends on the
sampling rate.
Usage: py build_spectral_features.py. Output: dados/pbit_spectral_features_v3.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_PATH = ROOT / "dados" / "pbit_signals_v3.csv"
OUTPUT_PATH = ROOT / "dados" / "pbit_spectral_features_v3.csv"
N_SAMPLES = 123
FEATURES_PATH = ROOT / "dados" / "pbit_features_v3.csv"


def spectrum(signals):
    """Power by spectral line (excluding zero frequency) after removing each cycle's least-squares line."""
    t = np.arange(signals.shape[1], dtype=float)
    t -= t.mean()
    slopes = (signals @ t) / (t @ t)
    residual = signals - signals.mean(axis=1, keepdims=True) - np.outer(slopes, t)
    return np.abs(np.fft.rfft(residual, axis=1))[:, 1:] ** 2, residual


def descriptors(signals):
    power, residual = spectrum(signals)
    total = power.sum(axis=1, keepdims=True)
    share = np.divide(power, total, out=np.full_like(power, 1 / power.shape[1]), where=total > 0)
    lines = np.arange(1, power.shape[1] + 1)
    frequency = lines / (signals.shape[1] / 2)  # fraction of Nyquist
    entropy = -(share * np.log(np.clip(share, 1e-15, None))).sum(axis=1) / np.log(power.shape[1])
    flatness = np.exp(np.log(np.clip(share, 1e-15, None)).mean(axis=1)) / share.mean(axis=1)

    def lag_correlation(lag):
        numerator = (residual[:, :-lag] * residual[:, lag:]).sum(axis=1)
        denominator = (residual**2).sum(axis=1)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    level = np.abs(signals.mean(axis=1))
    return pd.DataFrame({
        "spectral_log_power_ratio": np.log10(np.clip(total[:, 0], 1e-15, None) / np.clip(level, 1e-6, None) ** 2),
        "spectral_low_share": share[:, frequency <= 0.1].sum(axis=1),
        "spectral_mid_share": share[:, (frequency > 0.1) & (frequency <= 0.4)].sum(axis=1),
        "spectral_high_share": share[:, frequency > 0.4].sum(axis=1),
        "spectral_centroid": (share * frequency).sum(axis=1),
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "spectral_dominant_frequency": frequency[share.argmax(axis=1)],
        "spectral_dominant_share": share.max(axis=1),
        "lag1_correlation": lag_correlation(1),
        "lag2_correlation": lag_correlation(2),
    })


def load_signals(indices):
    raw = pd.read_csv(SIGNALS_PATH)
    raw = raw[raw["experimento_idx"].isin(indices)]
    wide = raw.pivot(index="experimento_idx", columns="sample_index", values="value").reindex(index=indices, columns=range(1, N_SAMPLES + 1))
    return wide.interpolate(axis=1, limit_direction="both").values


def main():
    indices = pd.read_csv(FEATURES_PATH)["experimento_idx"].tolist()
    table = descriptors(load_signals(indices))
    table.insert(0, "experimento_idx", indices)
    table.to_csv(OUTPUT_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
