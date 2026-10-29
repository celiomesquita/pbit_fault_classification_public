"""Grouped train/test split for the PBIT feature matrix.

The primary rule (from ML_DEVELOPMENT_PLAN.md): Serial_Number must be used
as the grouping variable so measurements from the same equipment never
appear in both train and test.

The split uses StratifiedGroupKFold from scikit-learn, which tries to keep
all experiments of a given Serial_Number in the same fold while preserving
class proportions as much as possible.

With n_splits=5 the first fold becomes the test set (~20 %), the remaining
four folds become the training set (~80 %).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


META_COLS = ["Experiment_Index", "Serial_Number", "Serial_Run_Index",
             "Fault_Mode", "Severity_Index"]

FAULT_ORDER = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]


def grouped_train_test_split(
    features_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *features_df* into train and test sets, grouped by Serial_Number.

    Uses StratifiedGroupKFold so that:
    - All experiments from the same Serial_Number stay in the same partition.
    - Fault_Mode proportions are approximated in both partitions.

    The first fold is used as the test set (~1/n_splits of the data).

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with metadata columns (Experiment_Index,
        Serial_Number, Fault_Mode, Severity_Index, …).
    n_splits : int
        Number of folds; determines test fraction (~1/n_splits).
    random_state : int

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Both DataFrames preserve the original column order and index.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)

    X      = features_df.index.values
    y      = features_df["Fault_Mode"].values
    groups = features_df["Serial_Number"].values

    for train_idx, test_idx in sgkf.split(X, y, groups):
        break  # take only the first fold as test

    return features_df.iloc[train_idx].copy(), features_df.iloc[test_idx].copy()


def split_diagnostics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-class count table for train and test sets.

    Columns: Fault_Mode, Train_Count, Test_Count, Total, Train_Pct, Test_Pct.
    """
    train_counts = train_df["Fault_Mode"].value_counts().rename("Train")
    test_counts  = test_df["Fault_Mode"].value_counts().rename("Test")

    diag = pd.concat([train_counts, test_counts], axis=1).fillna(0).astype(int)
    diag.index.name = "Fault_Mode"
    diag = diag.reindex([c for c in FAULT_ORDER if c in diag.index])
    diag["Total"]    = diag["Train"] + diag["Test"]
    diag["Train_%"]  = (diag["Train"] / diag["Total"] * 100).round(1)
    diag["Test_%"]   = (diag["Test"]  / diag["Total"] * 100).round(1)
    return diag.reset_index()


def f06_split_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print F06-specific split details: serial numbers and experiment indices."""
    print("F06 in TRAIN")
    f06_train = train_df[train_df["Fault_Mode"] == "F06"][
        ["Experiment_Index", "Serial_Number", "Serial_Run_Index"]
    ]
    print(f06_train.to_string(index=False) if len(f06_train) else "  (none)")
    print()
    print("F06 in TEST")
    f06_test = test_df[test_df["Fault_Mode"] == "F06"][
        ["Experiment_Index", "Serial_Number", "Serial_Run_Index"]
    ]
    print(f06_test.to_string(index=False) if len(f06_test) else "  (none)")
