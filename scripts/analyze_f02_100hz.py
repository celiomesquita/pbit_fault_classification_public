"""Plan action P23: try to tell F02 from N01 with frequency descriptors defined in Hz, assuming 100 Hz.

The real sampling rate is unknown; 100 Hz comes from a reference on another actuator. Assumed values: 1.23 s window,
0.813 Hz between spectral lines, 50 Hz maximum, actuator bandwidth of 15 to 17 Hz in the reference.
Usage, from scripts/: py analyze_f02_100hz.py. Outputs: resultados/f02_100hz.csv, resultados/f02_100hz_same_unit.csv.
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import analyze_f02_vs_n01 as previous
import build_spectral_features as spectral
import experimentos_base as base

ASSUMED_RATE_HZ = 100.0
BANDS_HZ = [(0, 5), (5, 10), (10, 14), (14, 18), (18, 25), (25, 50.01)]
N_PERMUTATIONS = {"logistic": 200, "forest": 50}
N_BOOT = 2000


def line_frequencies(n_samples=spectral.N_SAMPLES, rate=ASSUMED_RATE_HZ):
    return np.fft.rfftfreq(n_samples, d=1 / rate)[1:]


def band_features(signals):
    """Share and level of power per band, plus where the power sits between 5 and 30 Hz."""
    power, _ = spectral.spectrum(signals)
    frequency = line_frequencies(signals.shape[1])
    total = power.sum(axis=1, keepdims=True)
    share = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    table = {}
    for low, high in BANDS_HZ:
        mask = (frequency >= low) & (frequency < high)
        table[f"share_{low}_{int(high)}hz"] = share[:, mask].sum(axis=1)
        table[f"log_power_{low}_{int(high)}hz"] = np.log10(np.clip(power[:, mask].sum(axis=1), 1e-15, None))
    window = (frequency >= 5) & (frequency < 30)
    window_share = share[:, window]
    window_total = np.clip(window_share.sum(axis=1), 1e-15, None)
    table["centroid_5_30hz"] = (window_share * frequency[window]).sum(axis=1) / window_total
    table["peak_frequency_5_30hz"] = frequency[window][window_share.argmax(axis=1)]
    table["peak_share_5_30hz"] = window_share.max(axis=1) / window_total
    return pd.DataFrame(table)


def full_spectrum(signals):
    power, _ = spectral.spectrum(signals)
    total = np.clip(power.sum(axis=1, keepdims=True), 1e-15, None)
    columns = [f"log_share_line_{i + 1}" for i in range(power.shape[1])]
    return pd.DataFrame(np.log10(np.clip(power / total, 1e-15, None)), columns=columns)


def evaluate(X, y, groups, name, rng):
    units = sorted(set(groups))
    position = pd.Series(groups).map({unit: i for i, unit in enumerate(units)}).values
    scores = np.column_stack([previous.out_of_fold(name, X, y, groups, seed) for seed in range(5)]).mean(axis=1)
    samples = []
    for draw in rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=N_BOOT):
        weights = draw[position].astype(float)
        if y[weights > 0].min() != y[weights > 0].max():
            samples.append(roc_auc_score(y, scores, sample_weight=weights))
    real = np.mean([roc_auc_score(y, previous.out_of_fold(name, X, y, groups, 100 + seed)) for seed in range(5)])
    blocks = pd.Series(groups + "|" + y.astype(str))
    unique_blocks = blocks.drop_duplicates()
    null = []
    for permutation in range(N_PERMUTATIONS[name]):
        labels = rng.permutation(unique_blocks.str.split("|").str[1].values)
        shuffled = blocks.map(dict(zip(unique_blocks.values, labels))).astype(int).values
        null.append(roc_auc_score(shuffled, previous.out_of_fold(name, X, shuffled, groups, permutation)))
    return {"model": name, "auc": real, "ci_low": np.quantile(samples, 0.025), "ci_high": np.quantile(samples, 0.975), "null_mean": np.mean(null),
            "null_p97_5": np.quantile(null, 0.975), "p_value": (np.sum(np.array(null) >= real) + 1) / (len(null) + 1), "n_permutations": len(null)}


def main():
    df = pd.read_csv(base.RAIZ / "dados" / "pbit_features_corrigido_v3.csv")
    df = df[df["condicao"].isin(["N01", "F02"])].reset_index(drop=True)
    signals = spectral.load_signals(df["experimento_idx"].tolist())
    y = (df["condicao"].values == "F02").astype(int)
    groups = df["numero_serie"].values
    bands, spectrum_table = band_features(signals), full_spectrum(signals)
    sets = {"physical_bands": bands, "full_spectrum": spectrum_table, "bands_and_spectrum": pd.concat([bands, spectrum_table], axis=1)}
    rng = np.random.default_rng(2026)

    rows = []
    for (set_name, frame), name in product(sets.items(), N_PERMUTATIONS):
        rows.append({"feature_set": set_name, **evaluate(frame.values, y, groups, name, rng)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(base.RAIZ / "resultados" / "f02_100hz.csv", index=False, lineterminator="\n")

    same_unit = bands.assign(numero_serie=groups, condicao=df["condicao"].values)
    same_unit = same_unit[same_unit["numero_serie"].isin(set(groups[y == 1]) & set(groups[y == 0]))]
    medians = same_unit.groupby(["numero_serie", "condicao"])[[c for c in bands if c.startswith("share_") or c.endswith("5_30hz")]].median().round(3)
    medians.to_csv(base.RAIZ / "resultados" / "f02_100hz_same_unit.csv", lineterminator="\n")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(medians.to_string())
    print(bands.assign(condicao=df["condicao"].values).groupby("condicao")[["share_14_18hz", "centroid_5_30hz", "peak_frequency_5_30hz", "peak_share_5_30hz"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
