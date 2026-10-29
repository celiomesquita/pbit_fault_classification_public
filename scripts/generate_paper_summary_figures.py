"""Generate summary figures for the paper results section."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "paper" / "manuscript" / "figures"
SUPP_FIG_DIR = PROJECT_ROOT / "paper" / "supplement" / "figures"


def percent_label(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def generate_heldout_tradeoff() -> None:
    df = pd.read_csv(DATA_DIR / "final_test_predictions.csv")
    normal = df["Fault_Mode"].isin(["N01", "N02"])
    critical = df["Fault_Mode"].isin(["F03", "F06"])
    colors = {"F1-tuned": "#4e79a7", "CMR-tuned": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.1, 3.35))

    for model in colors:
        predicted_normal = df[f"pred_main_{model}"].isin(["N01", "N02"])
        misses = int((critical & predicted_normal).sum())
        alarms = int((normal & ~predicted_normal).sum())
        cmr = misses / int(critical.sum())
        fa0 = alarms / int(normal.sum())
        ax.scatter(
            fa0,
            cmr,
            s=65,
            color=colors[model],
            edgecolors="#222222",
            linewidths=0.45,
            zorder=3,
        )
        ax.annotate(
            f"{model}\n"
            f"CMR {percent_label(cmr)} ({misses}/{int(critical.sum())})\n"
            f"FA0 {percent_label(fa0)} ({alarms}/{int(normal.sum())})",
            (fa0, cmr),
            xytext=(12, -10) if model == "F1-tuned" else (-12, 14),
            textcoords="offset points",
            fontsize=8,
            color=colors[model],
            ha="left" if model == "F1-tuned" else "right",
            va="top" if model == "F1-tuned" else "bottom",
        )

    ax.set_xlabel("False-alarm rate on normal cycles (FA0)", fontsize=9)
    ax.set_ylabel("Critical miss rate (CMR)", fontsize=9)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(-0.01, 0.11)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(labelsize=8)
    ax.grid(color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.7)
    fig.savefig(FIG_DIR / "fig_heldout_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_f06_sensitivity() -> None:
    df = pd.read_csv(DATA_DIR / "cv_composite6_sensitivity.csv")
    df["Label"] = df["Config"].replace({"All F06 SNs": "All F06\nSNs"})
    df["Label"] = df["Label"].str.replace("Remove SN ", "Remove\n", regex=False)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    bars = ax.bar(
        x,
        df["F06_Recall"].astype(float),
        color="#4e79a7",
        edgecolor="#222222",
        linewidth=0.45,
        label="F06 recall",
    )

    for bar, val in zip(bars, df["F06_Recall"].astype(float)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            percent_label(val),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold" if val > 0 else "normal",
        )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        df["CMR"].astype(float),
        color="#d62728",
        marker="o",
        linewidth=1.8,
        label="CMR",
    )
    for xi, val in zip(x, df["CMR"].astype(float)):
        ax2.text(xi, val + 0.004, f"{val:.3f}", color="#d62728", fontsize=7, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=8)
    ax.set_ylabel("F06 recall", fontsize=9, color="#4e79a7")
    ax2.set_ylabel("CMR", fontsize=9, color="#d62728")
    ax.set_ylim(0, 0.78)
    ax2.set_ylim(0.05, 0.11)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout(pad=0.7)
    fig.savefig(SUPP_FIG_DIR / "fig_f06_sn_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_heldout_tradeoff()
    generate_f06_sensitivity()
    print("Generated:")
    print(FIG_DIR / "fig_heldout_tradeoff.pdf")
    print(SUPP_FIG_DIR / "fig_f06_sn_sensitivity.pdf")


if __name__ == "__main__":
    main()
