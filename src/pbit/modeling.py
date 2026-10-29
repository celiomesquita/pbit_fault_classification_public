"""Model training utilities for the PBIT fault classification pipeline.

Implements the 6 experimental configurations from ML_DEVELOPMENT_PLAN.md,
grouped cross-validation, class weighting, and evaluation metrics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from pbit.augmentation import augment_training_set
from pbit.features import extract_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_MAP: dict[str, int] = {
    "N01": 0, "N02": 0,
    "F01": 1,
    "F02": 2, "F04": 2, "F05": 2,
    "F03": 3, "F06": 3,
}

FAULT_ORDER = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]

META_COLS = ["Experiment_Index", "Serial_Number", "Serial_Run_Index",
             "Fault_Mode", "Severity_Index"]


# ---------------------------------------------------------------------------
# XGBoost wrapper — avoids stale internal state across CV folds
# ---------------------------------------------------------------------------

class XGBWrapper:
    """Thin sklearn-compatible wrapper around XGBClassifier.

    XGBoost's deepcopy can preserve fitted state (classes_, booster) across
    folds, causing 'Invalid classes' errors when string labels are used.
    This wrapper always creates a fresh XGBClassifier on each fit() call and
    encodes string labels to integers internally, so callers never need to
    worry about label encoding or deepcopy state.
    """

    def __init__(self, **kwargs):
        self._params = kwargs
        self._le: LabelEncoder | None = None

    def get_params(self, deep: bool = True) -> dict:
        return self._params.copy()

    def set_params(self, **params):
        self._params.update(params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        from xgboost import XGBClassifier
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._clf = XGBClassifier(**self._params)
        self._clf.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_enc = self._clf.predict(X)
        return self._le.inverse_transform(y_enc)


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    augment: bool
    weight_mode: str   # "none" | "freq" | "sev" | "freq_sev"


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig("Baseline",         augment=False, weight_mode="none"),
    ExperimentConfig("Freq weights",     augment=False, weight_mode="freq"),
    ExperimentConfig("Sev weights",      augment=False, weight_mode="sev"),
    ExperimentConfig("Freq×Sev weights", augment=False, weight_mode="freq_sev"),
    ExperimentConfig("Augmentation",     augment=True,  weight_mode="none"),
    ExperimentConfig("Aug + Freq×Sev",   augment=True,  weight_mode="freq_sev"),
]


# ---------------------------------------------------------------------------
# Class / sample weighting
# ---------------------------------------------------------------------------

def compute_sample_weights(
    y: np.ndarray,
    mode: str,
    severity_map: dict[str, int] = SEVERITY_MAP,
) -> np.ndarray:
    """Return a per-sample weight array.

    mode
    ----
    "none"      : all weights = 1
    "freq"      : frequency-based (inverse class frequency, normalised)
    "sev"       : severity-based  (1 + severity_index)
    "freq_sev"  : frequency × severity
    """
    if mode == "none":
        return np.ones(len(y), dtype=float)

    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes  = len(classes)
    freq_w = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}

    weights = np.empty(len(y), dtype=float)
    for i, label in enumerate(y):
        fw = freq_w.get(label, 1.0)
        sw = 1.0 + severity_map.get(label, 0)
        if mode == "freq":
            weights[i] = fw
        elif mode == "sev":
            weights[i] = sw
        else:   # freq_sev
            weights[i] = fw * sw
    return weights


# ---------------------------------------------------------------------------
# Cross-validation setup
# ---------------------------------------------------------------------------

def make_cv_folds(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) pairs for grouped stratified CV.

    Grouped by Serial_Number, stratified by Fault_Mode.
    Indices are positional (iloc-compatible).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)
    X      = np.arange(len(train_df))
    y      = train_df["Fault_Mode"].values
    groups = train_df["Serial_Number"].values
    return list(sgkf.split(X, y, groups))


def fold_diagnostics(train_df: pd.DataFrame, folds: list) -> pd.DataFrame:
    """Per-fold class and serial-number counts."""
    rows = []
    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        tr  = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]
        row = {"fold": fold_i}
        row["train_n"]   = len(tr)
        row["val_n"]     = len(val)
        row["train_sns"] = tr["Serial_Number"].nunique()
        row["val_sns"]   = val["Serial_Number"].nunique()
        for fm in FAULT_ORDER:
            row[f"tr_{fm}"]  = int((tr["Fault_Mode"]  == fm).sum())
            row[f"val_{fm}"] = int((val["Fault_Mode"] == fm).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Safety metrics
# ---------------------------------------------------------------------------

_SEV0_CLASSES = {"N01", "N02"}
_SEV3_CLASSES = {"F03", "F06"}


def compute_safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity_map: dict[str, int] = SEVERITY_MAP,
) -> dict:
    """Compute the three safety metrics used in the paper.

    CMR (Critical Miss Rate)
        Rate at which severity-3 faults (F03, F06) are predicted as
        severity-0 (N01, N02).  Lower is safer.

    Downward Risk
        Rate at which any sample is predicted as a strictly lower-severity
        class.  Lower is safer.

    Severity Error Rate
        Rate of any severity mismatch between true and predicted class.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)

    crit_mask  = np.array([c in _SEV3_CLASSES for c in y_true_arr])
    sev0_mask  = np.array([c in _SEV0_CLASSES for c in y_pred_arr])
    cmr_count  = int(np.sum(crit_mask & sev0_mask))
    crit_total = int(crit_mask.sum())
    cmr = cmr_count / crit_total if crit_total > 0 else float("nan")

    sev_true = np.array([severity_map.get(c, 0) for c in y_true_arr])
    sev_pred = np.array([severity_map.get(c, 0) for c in y_pred_arr])
    down_risk = float(np.sum(sev_pred < sev_true) / n)
    sev_err   = float(np.sum(sev_pred != sev_true) / n)

    return {
        "cmr":        float(cmr),
        "down_risk":  down_risk,
        "sev_err":    sev_err,
        "cmr_count":  cmr_count,
        "crit_total": crit_total,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity_map: dict[str, int] = SEVERITY_MAP,
    labels: list[str] = FAULT_ORDER,
) -> dict:
    """Compute all required evaluation metrics."""
    present = sorted(set(y_true) | set(y_pred))
    eval_labels = [l for l in labels if l in present]

    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                               labels=eval_labels, zero_division=0))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))

    report = classification_report(
        y_true, y_pred, labels=eval_labels,
        output_dict=True, zero_division=0,
    )
    per_class_recall = {c: report[c]["recall"]   for c in eval_labels if c in report}
    per_class_f1     = {c: report[c]["f1-score"] for c in eval_labels if c in report}
    per_class_prec   = {c: report[c]["precision"] for c in eval_labels if c in report}

    # Per-severity recall (macro within each severity group)
    sev_groups: dict[int, list[str]] = {}
    for label, sev in severity_map.items():
        if label in eval_labels:
            sev_groups.setdefault(sev, []).append(label)

    per_sev_recall: dict[int, float] = {}
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    for sev, classes in sev_groups.items():
        mask = np.isin(y_true_arr, classes)
        if mask.sum() > 0:
            per_sev_recall[sev] = float(
                f1_score(y_true_arr[mask], y_pred_arr[mask],
                         labels=classes, average="macro", zero_division=0)
            )

    safety = compute_safety_metrics(y_true, y_pred, severity_map)

    return {
        "macro_f1":           macro_f1,
        "balanced_accuracy":  bal_acc,
        "f06_recall":         per_class_recall.get("F06", float("nan")),
        "f05_recall":         per_class_recall.get("F05", float("nan")),
        "f04_recall":         per_class_recall.get("F04", float("nan")),
        "sev0_recall":        per_sev_recall.get(0, float("nan")),
        "sev1_recall":        per_sev_recall.get(1, float("nan")),
        "sev2_recall":        per_sev_recall.get(2, float("nan")),
        "sev3_recall":        per_sev_recall.get(3, float("nan")),
        "cmr":                safety["cmr"],
        "down_risk":          safety["down_risk"],
        "sev_err":            safety["sev_err"],
        "per_class_recall":   per_class_recall,
        "per_class_f1":       per_class_f1,
        "per_class_prec":     per_class_prec,
        "confusion_matrix":   confusion_matrix(y_true, y_pred, labels=eval_labels),
        "cm_labels":          eval_labels,
    }


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def _fit_predict(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    sample_weights: np.ndarray,
    needs_imputation: bool,
) -> np.ndarray:
    """Fit model on training fold and return predictions on validation fold.

    Handles NaN imputation for sklearn models that cannot deal with missing
    values natively (Random Forest, Gradient Boosting).
    """
    if needs_imputation:
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_val   = imp.transform(X_val)

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model.predict(X_val)


def run_cv_experiment(
    model,
    config: ExperimentConfig,
    train_df: pd.DataFrame,
    long_df: pd.DataFrame,
    feat_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    needs_imputation: bool = False,
    aug_target: int = 160,
    aug_noise: float = 0.01,
    random_state: int = 42,
) -> list[dict]:
    """Run one experiment configuration across all CV folds.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must support ``fit(X, y, sample_weight=...)`` and ``predict(X)``.
    config : ExperimentConfig
    train_df : pd.DataFrame
        Real training feature matrix (from train_real.csv).
    long_df : pd.DataFrame
        Clean long/tidy dataset (needed for per-fold augmentation).
    feat_cols : list[str]
        Feature column names (excludes metadata).
    folds : list of (train_idx, val_idx)
        Positional index pairs from ``make_cv_folds``.
    needs_imputation : bool
        True for sklearn models that cannot handle NaN (RF, GB).
    aug_target : int
        Target experiments per minority class after augmentation.
    aug_noise : float
        Gaussian noise fraction of per-signal std.
    random_state : int

    Returns
    -------
    list of dicts — one dict of metrics per fold.
    """
    import copy

    fold_results = []

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        tr_real  = train_df.iloc[tr_idx]
        val_real = train_df.iloc[val_idx]

        if config.augment:
            tr_exp_ids = set(tr_real["Experiment_Index"].tolist())
            aug_long   = augment_training_set(
                long_df, tr_exp_ids,
                target_per_class=aug_target,
                noise_scale=aug_noise,
                random_state=random_state + fold_i,
            )
            if len(aug_long) > 0:
                aug_feat = extract_features(aug_long)
                tr_fold  = pd.concat([tr_real, aug_feat], ignore_index=True)
            else:
                tr_fold = tr_real
        else:
            tr_fold = tr_real

        X_train = tr_fold[feat_cols].values.astype(float)
        y_train = tr_fold["Fault_Mode"].values
        X_val   = val_real[feat_cols].values.astype(float)
        y_val   = val_real["Fault_Mode"].values

        sw = compute_sample_weights(y_train, config.weight_mode)

        m = type(model)(**model.get_params())
        y_pred = _fit_predict(m, X_train, y_train, X_val, sw, needs_imputation)

        metrics = evaluate_predictions(y_val, y_pred)
        metrics["fold"] = fold_i
        fold_results.append(metrics)

    return fold_results


def summarize_cv(fold_results: list[dict]) -> dict:
    """Aggregate fold metrics into mean ± std for scalar metrics."""
    scalar_keys = [
        "macro_f1", "balanced_accuracy",
        "f06_recall", "f05_recall", "f04_recall",
        "sev0_recall", "sev1_recall", "sev2_recall", "sev3_recall",
        "cmr", "down_risk", "sev_err",
    ]
    summary = {}
    for key in scalar_keys:
        vals = [r[key] for r in fold_results if not np.isnan(r.get(key, float("nan")))]
        summary[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        summary[f"{key}_std"]  = float(np.std(vals))  if vals else float("nan")

    # Per-class recall mean across folds
    all_classes = set()
    for r in fold_results:
        all_classes |= set(r["per_class_recall"].keys())
    per_class_mean = {}
    for cls in all_classes:
        vals = [r["per_class_recall"].get(cls, float("nan")) for r in fold_results]
        vals = [v for v in vals if not np.isnan(v)]
        per_class_mean[cls] = float(np.mean(vals)) if vals else float("nan")
    summary["per_class_recall_mean"] = per_class_mean

    return summary


def results_table(all_summaries: dict[str, dict]) -> pd.DataFrame:
    """Maintenance-perspective comparison table (sorted by Macro F1)."""
    rows = []
    for label, s in all_summaries.items():
        rows.append({
            "Experiment":  label,
            "Macro_F1":    round(s.get("macro_f1_mean", float("nan")), 4),
            "F1_std":      round(s.get("macro_f1_std",  float("nan")), 4),
            "Bal_Acc":     round(s.get("balanced_accuracy_mean", float("nan")), 4),
            "F06_Recall":  round(s.get("f06_recall_mean", float("nan")), 4),
            "F05_Recall":  round(s.get("f05_recall_mean", float("nan")), 4),
            "F04_Recall":  round(s.get("f04_recall_mean", float("nan")), 4),
            "Sev3_Recall": round(s.get("sev3_recall_mean", float("nan")), 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Macro_F1", ascending=False)
        .reset_index(drop=True)
    )


def safety_results_table(all_summaries: dict[str, dict]) -> pd.DataFrame:
    """Safety-perspective comparison table (sorted by CMR ascending)."""
    rows = []
    for label, s in all_summaries.items():
        rows.append({
            "Experiment":  label,
            "CMR":         round(s.get("cmr_mean",       float("nan")), 4),
            "Down_Risk":   round(s.get("down_risk_mean", float("nan")), 4),
            "Sev_Err":     round(s.get("sev_err_mean",   float("nan")), 4),
            "Bal_Acc":     round(s.get("balanced_accuracy_mean", float("nan")), 4),
            "Macro_F1":    round(s.get("macro_f1_mean",  float("nan")), 4),
            "F06_Recall":  round(s.get("f06_recall_mean", float("nan")), 4),
            "Sev3_Recall": round(s.get("sev3_recall_mean", float("nan")), 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("CMR", ascending=True)
        .reset_index(drop=True)
    )
