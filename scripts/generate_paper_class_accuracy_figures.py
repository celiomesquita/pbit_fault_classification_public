"""Regenerate per-class accuracy figures used in the paper.

The figures are based on the current notebook configuration from
``notebooks/04_model_training.ipynb``: grouped five-fold out-of-fold
predictions on ``data/train_real.csv`` with 246 features, the
``Aug + Freq×Sev`` configuration, F04/F05/F06 augmentation target 160, and 1%
noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pbit.data_validation import load_long_dataset
from pbit.modeling import (
    FAULT_ORDER,
    META_COLS,
    SEVERITY_MAP,
    ExperimentConfig,
    make_cv_folds,
    run_cv_experiment,
    summarize_cv,
)


FEATURES_PATH = PROJECT_ROOT / "data" / "train_real.csv"
LONG_CLEAN_PATH = PROJECT_ROOT / "data" / "PBIT_atuador_DESCARAC_r1_long_clean.csv"
FIG_DIR = PROJECT_ROOT / "paper" / "supplement" / "figures"

CONFIG = ExperimentConfig("Paper canonical", augment=True, weight_mode="freq_sev")
AUG_TARGET = 160
AUG_NOISE = 0.01
RANDOM_STATE = 42

SEVERITY_COLORS = {
    0: "#1f77b4",
    1: "#7ec8e3",
    2: "#f28e2b",
    3: "#d62728",
}
INCORRECT_COLOR = "#d9d9d9"


def pooled_confusion(fold_results: list[dict]) -> np.ndarray:
    labels = FAULT_ORDER
    out = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_i = {label: i for i, label in enumerate(labels)}

    for fold in fold_results:
        fold_labels = fold["cm_labels"]
        cm = fold["confusion_matrix"]
        for r, true_label in enumerate(fold_labels):
            for c, pred_label in enumerate(fold_labels):
                out[label_to_i[true_label], label_to_i[pred_label]] += int(cm[r, c])
    return out


def plot_class_accuracy(cm: np.ndarray, model_name: str, output_path: Path) -> pd.DataFrame:
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    recall = np.divide(correct, totals, out=np.zeros_like(correct, dtype=float), where=totals > 0)
    incorrect = 1.0 - recall

    x = np.arange(len(FAULT_ORDER))
    colors = [SEVERITY_COLORS[SEVERITY_MAP[label]] for label in FAULT_ORDER]

    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.bar(x, recall * 100.0, color=colors, edgecolor="#222222", linewidth=0.45)
    ax.bar(
        x,
        incorrect * 100.0,
        bottom=recall * 100.0,
        color=INCORRECT_COLOR,
        edgecolor="#222222",
        linewidth=0.45,
    )

    for i, (pct, n) in enumerate(zip(recall * 100.0, totals)):
        if n == 0:
            label = "n=0"
            y = 4
            va = "bottom"
            color = "#222222"
        elif pct >= 12:
            label = f"{pct:.0f}%"
            y = pct / 2.0
            va = "center"
            color = "white"
        else:
            label = f"{pct:.0f}%"
            y = pct + 4.0
            va = "bottom"
            color = "#222222"
        ax.text(i, y, label, ha="center", va=va, fontsize=8, color=color, fontweight="bold")
        ax.text(i, 103.0, f"n={n}", ha="center", va="bottom", fontsize=7, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(FAULT_ORDER, fontsize=9)
    ax.set_ylabel("Classification share (%)", fontsize=9)
    ax.set_ylim(0, 111)
    ax.set_title(model_name, fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.7)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(
        {
            "Fault_Mode": FAULT_ORDER,
            "n": totals,
            "correct": correct,
            "recall": np.round(recall, 4),
        }
    )


def run_model(model, needs_imputation: bool, label: str, output_name: str, train_df: pd.DataFrame, long_df: pd.DataFrame, feat_cols: list[str], folds: list) -> pd.DataFrame:
    print(f"Running {label}...")
    fold_results = run_cv_experiment(
        model,
        CONFIG,
        train_df,
        long_df,
        feat_cols,
        folds,
        needs_imputation=needs_imputation,
        aug_target=AUG_TARGET,
        aug_noise=AUG_NOISE,
        random_state=RANDOM_STATE,
    )
    summary = summarize_cv(fold_results)
    cm = pooled_confusion(fold_results)
    stats = plot_class_accuracy(cm, label, FIG_DIR / output_name)
    print(
        f"{label}: macro-F1={summary['macro_f1_mean']:.4f}, "
        f"balanced accuracy={summary['balanced_accuracy_mean']:.4f}, "
        f"F06 fold recall={summary['f06_recall_mean']:.4f}"
    )
    print(stats.to_string(index=False))
    print()
    return stats


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(FEATURES_PATH)
    long_df = load_long_dataset(LONG_CLEAN_PATH)
    feat_cols = [col for col in train_df.columns if col not in META_COLS]
    folds = make_cv_folds(train_df, n_splits=5, random_state=RANDOM_STATE)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    run_model(
        rf,
        True,
        "Random Forest",
        "fig_class_accuracy_rf.pdf",
        train_df,
        long_df,
        feat_cols,
        folds,
    )
    run_model(
        lgbm,
        False,
        "LightGBM",
        "fig_class_accuracy_lgbm.pdf",
        train_df,
        long_df,
        feat_cols,
        folds,
    )


if __name__ == "__main__":
    main()
