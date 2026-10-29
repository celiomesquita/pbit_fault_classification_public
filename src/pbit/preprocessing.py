"""Preprocessing steps for the PBIT long/tidy dataset.

Two steps are applied before feature engineering:

1. Linear interpolation for experiments with sparse missing values
   (defined as missing_fraction <= SPARSE_THRESHOLD, default 0.5).
   Interpolation is intra-experiment only — it uses only the signal values
   within the same PBIT cycle, so it is fold-safe.

2. Exclusion of experiments whose signal is mostly absent
   (missing_fraction > SPARSE_THRESHOLD).  These experiments cannot be
   meaningfully represented by any feature extraction and are logged
   separately for traceability.
"""

from __future__ import annotations

import pandas as pd


SPARSE_THRESHOLD = 0.50   # fraction of missing samples below which we interpolate


def classify_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-experiment missing-value summary with a quality label.

    Columns returned
    ----------------
    Experiment_Index, Serial_Number, Fault_Mode, Severity_Index,
    Total_Samples, Missing_Count, Missing_Fraction, Quality
      Quality = 'ok'      — no missing values
      Quality = 'sparse'  — 1 or a few missing (interpolatable)
      Quality = 'corrupt' — majority of signal is absent
    """
    total_per_exp = df.groupby("Experiment_Index")["Measurement"].size().rename("Total_Samples")
    missing_per_exp = df.groupby("Experiment_Index")["Measurement"].apply(
        lambda s: s.isna().sum()
    ).rename("Missing_Count")

    meta = (
        df[["Experiment_Index", "Serial_Number", "Fault_Mode", "Severity_Index"]]
        .drop_duplicates("Experiment_Index")
        .set_index("Experiment_Index")
    )

    summary = meta.join(total_per_exp).join(missing_per_exp)
    summary["Missing_Fraction"] = summary["Missing_Count"] / summary["Total_Samples"]
    summary["Quality"] = "ok"
    summary.loc[
        (summary["Missing_Count"] > 0) & (summary["Missing_Fraction"] <= SPARSE_THRESHOLD),
        "Quality",
    ] = "sparse"
    summary.loc[summary["Missing_Fraction"] > SPARSE_THRESHOLD, "Quality"] = "corrupt"

    return summary.reset_index().sort_values("Experiment_Index")


def interpolate_sparse(df: pd.DataFrame, threshold: float = SPARSE_THRESHOLD) -> pd.DataFrame:
    """Linearly interpolate missing values for experiments below *threshold*.

    Interpolation is applied per experiment, ordered by Sample_Index.
    Experiments at or above *threshold* are left unchanged.

    Returns a copy of *df* with NaNs filled where applicable.
    """
    missing_frac = (
        df.groupby("Experiment_Index")["Measurement"]
        .apply(lambda s: s.isna().sum() / len(s))
    )
    sparse_experiments = missing_frac[
        (missing_frac > 0) & (missing_frac <= threshold)
    ].index

    df_out = df.copy()
    for exp_idx in sparse_experiments:
        mask = df_out["Experiment_Index"] == exp_idx
        df_out.loc[mask, "Measurement"] = (
            df_out.loc[mask]
            .sort_values("Sample_Index")["Measurement"]
            .interpolate(method="linear", limit_direction="both")
            .values
        )
    return df_out


def exclude_corrupt(df: pd.DataFrame, threshold: float = SPARSE_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove experiments whose missing fraction exceeds *threshold*.

    Returns
    -------
    df_clean : pd.DataFrame
        Dataset with corrupt experiments removed.
    excluded : pd.DataFrame
        Metadata of the removed experiments for the audit log.
    """
    missing_frac = (
        df.groupby("Experiment_Index")["Measurement"]
        .apply(lambda s: s.isna().sum() / len(s))
    )
    corrupt_experiments = missing_frac[missing_frac > threshold].index

    excluded_meta = (
        df[df["Experiment_Index"].isin(corrupt_experiments)][
            ["Experiment_Index", "Serial_Number", "Serial_Run_Index",
             "Fault_Mode", "Severity_Index"]
        ]
        .drop_duplicates("Experiment_Index")
        .sort_values("Experiment_Index")
        .copy()
    )
    excluded_meta["Missing_Count"] = (
        df[df["Experiment_Index"].isin(corrupt_experiments)]
        .groupby("Experiment_Index")["Measurement"]
        .apply(lambda s: s.isna().sum())
        .values
    )
    excluded_meta["Missing_Fraction"] = (
        missing_frac[corrupt_experiments].values
    )

    df_clean = df[~df["Experiment_Index"].isin(corrupt_experiments)].copy()
    return df_clean, excluded_meta


def normalize_by_serial_number(
    long_df: pd.DataFrame,
    method: str = "zscore",
) -> pd.DataFrame:
    """Normalise Measurement within each Serial_Number.

    Removes equipment-level DC offset and scale differences so the feature
    extractor learns fault *shape* rather than absolute amplitude.  Because
    each Serial_Number belongs exclusively to either the training or the test
    partition, computing statistics per SN on the full dataset is leak-free.

    method
    ------
    'zscore'  : (x - SN_mean) / SN_std   — default, removes offset and scale
    'demean'  : x - SN_mean              — removes offset only
    """
    import numpy as np

    df = long_df.copy()
    stats = (
        df.groupby("Serial_Number")["Measurement"]
        .agg(sn_mean="mean", sn_std="std")
    )
    df = df.join(stats, on="Serial_Number")
    if method == "zscore":
        df["sn_std"] = df["sn_std"].replace(0, 1.0).fillna(1.0)
        df["Measurement"] = (df["Measurement"] - df["sn_mean"]) / df["sn_std"]
    else:  # demean
        df["Measurement"] = df["Measurement"] - df["sn_mean"]
    return df.drop(columns=["sn_mean", "sn_std"])


def preprocess(df: pd.DataFrame, threshold: float = SPARSE_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply both preprocessing steps in order.

    1. Linearly interpolate sparse-missing experiments.
    2. Exclude corrupt experiments (majority of signal absent).

    Parameters
    ----------
    df : pd.DataFrame
        Raw long dataset as returned by ``load_long_dataset``.
    threshold : float
        Missing-fraction boundary between 'sparse' and 'corrupt'.

    Returns
    -------
    df_clean : pd.DataFrame
        Preprocessed dataset (interpolated, corrupt rows removed).
    excluded : pd.DataFrame
        Metadata of excluded experiments.
    """
    df_interpolated = interpolate_sparse(df, threshold)
    df_clean, excluded = exclude_corrupt(df_interpolated, threshold)
    return df_clean, excluded
