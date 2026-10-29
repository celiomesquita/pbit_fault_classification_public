from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_LONG_COLUMNS = [
    "Experiment_Index",
    "Serial_Number",
    "Serial_Run_Index",
    "Fault_Mode",
    "Severity_Index",
    "Sample_Index",
    "Measurement",
]


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    columns: int
    experiments: int
    serial_numbers: int
    sample_positions: int
    missing_measurements: int
    rows_per_experiment: list[int]
    fault_counts: dict[str, int]
    severity_counts: dict[int, int]
    f06_records: pd.DataFrame
    bad_records: pd.DataFrame


def load_long_dataset(path: str | Path) -> pd.DataFrame:
    """Load the long/tidy PBIT dataset and enforce core dtypes."""
    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_LONG_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_LONG_COLUMNS].copy()
    integer_columns = ["Experiment_Index", "Serial_Run_Index", "Severity_Index", "Sample_Index"]
    for col in integer_columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    df["Measurement"] = pd.to_numeric(df["Measurement"], errors="coerce")
    df["Serial_Number"] = df["Serial_Number"].astype(str)
    df["Fault_Mode"] = df["Fault_Mode"].astype(str)
    return df


def summarize_long_dataset(df: pd.DataFrame) -> ValidationSummary:
    """Build validation diagnostics for the long/tidy PBIT dataset."""
    rows_per_experiment = sorted(df.groupby("Experiment_Index").size().unique().tolist())
    experiment_meta = (
        df[["Experiment_Index", "Serial_Number", "Serial_Run_Index", "Fault_Mode", "Severity_Index"]]
        .drop_duplicates("Experiment_Index")
        .sort_values("Experiment_Index")
    )

    missing_by_experiment = (
        df["Measurement"]
        .isna()
        .groupby(df["Experiment_Index"])
        .sum()
        .astype(int)
    )
    bad_records = experiment_meta.merge(
        missing_by_experiment[missing_by_experiment > 0].rename("Missing_Measurements"),
        left_on="Experiment_Index",
        right_index=True,
        how="inner",
    )

    f06_records = experiment_meta.loc[experiment_meta["Fault_Mode"] == "F06"].copy()

    return ValidationSummary(
        rows=len(df),
        columns=len(df.columns),
        experiments=df["Experiment_Index"].nunique(),
        serial_numbers=df["Serial_Number"].nunique(),
        sample_positions=df["Sample_Index"].nunique(),
        missing_measurements=int(df["Measurement"].isna().sum()),
        rows_per_experiment=rows_per_experiment,
        fault_counts=experiment_meta["Fault_Mode"].value_counts().sort_index().to_dict(),
        severity_counts=experiment_meta["Severity_Index"].value_counts().sort_index().to_dict(),
        f06_records=f06_records,
        bad_records=bad_records,
    )


def assert_expected_long_dataset(summary: ValidationSummary) -> None:
    """Raise if the known r1 long dataset invariants are violated."""
    expected_fault_counts = {
        "F01": 161,
        "F02": 320,
        "F03": 267,
        "F04": 39,
        "F05": 41,
        "F06": 9,
        "N01": 586,
        "N02": 67,
    }
    expected_severity_counts = {0: 653, 1: 161, 2: 400, 3: 276}

    checks = {
        "rows": summary.rows == 183_270,
        "columns": summary.columns == 7,
        "experiments": summary.experiments == 1_490,
        "serial_numbers": summary.serial_numbers == 65,
        "sample_positions": summary.sample_positions == 123,
        "rows_per_experiment": summary.rows_per_experiment == [123],
        "fault_counts": summary.fault_counts == expected_fault_counts,
        "severity_counts": summary.severity_counts == expected_severity_counts,
        "missing_measurements": summary.missing_measurements == 248,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Dataset validation failed: {failed}")


def print_summary(summary: ValidationSummary) -> None:
    """Print a concise human-readable validation report."""
    print("Long dataset validation")
    print(f"Rows: {summary.rows:,}")
    print(f"Columns: {summary.columns}")
    print(f"Experiments: {summary.experiments:,}")
    print(f"Serial numbers: {summary.serial_numbers}")
    print(f"Sample positions: {summary.sample_positions}")
    print(f"Rows per experiment: {summary.rows_per_experiment}")
    print(f"Missing measurements: {summary.missing_measurements}")
    print()
    print("Fault-mode counts")
    for key, value in summary.fault_counts.items():
        print(f"  {key}: {value}")
    print()
    print("Severity counts")
    for key, value in summary.severity_counts.items():
        print(f"  {key}: {value}")
    print()
    print("F06 records")
    print(summary.f06_records.to_string(index=False))
    print()
    print("Records with missing measurements")
    print(summary.bad_records.to_string(index=False))

