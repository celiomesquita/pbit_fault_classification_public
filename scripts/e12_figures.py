"""Publication/export figures for the verified E12 diagnostic, with no model fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e12_diagnostic"
FIG = OUT / "figures"


def save(fig, name):
    fig.savefig(FIG / (name+".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name+".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    assert json.loads((OUT / "verification.json").read_text())["status"] == "passed"
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    units = pd.read_csv(OUT / "unit_errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [3, 1.2]}, sharey=True)
    for ax, cls in zip(axes, ("F02", "F05")):
        unit_ids = sorted(units[units.Fault_Mode == cls].Serial_Number.unique())
        for j, (arm, label, colour) in enumerate((("B1_complete", "B1 complete", "#2369a0"), ("B3_joint", "B3 joint", "#b24b29"))):
            group = units[(units.arm == arm) & (units.Fault_Mode == cls)].set_index("Serial_Number").loc[unit_ids]
            ax.bar(np.arange(len(unit_ids))+(j-.5)*.36, group.detect_recall, .36, label=label, color=colour)
        ax.axhline(.90, color="black", linestyle="--", linewidth=1, label="Experimental target")
        ax.set_xticks(np.arange(len(unit_ids)), [f"{u.replace('SN ', '')}\n(n={int(group.loc[u, 'n_cycles'])})" for u in unit_ids])
        ax.set_title(f"{cls}: {len(unit_ids)} units")
        ax.set_xlabel("Serial number and unique cycle count")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Detection recall, mean across ten reused seeds")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "unit_recall")

    geometry = pd.read_csv(OUT / "geometry_summary.csv")
    geometry = geometry[geometry.arm == "B1_complete"]
    reps = ["raw_absolute", "raw_shape", "F18_IQR", "F246_IQR"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric, title in zip(axes, ("normal_closer_cycle_fraction", "within_normal_q95_cycle_fraction"),
                 ("Normal neighbour nearer than same-fault neighbour", "Inside training normal distance reference (95th percentile)")):
        for j, (cls, colour) in enumerate((("F02", "#2369a0"), ("F05", "#b24b29"))):
            values = geometry[geometry.Fault_Mode == cls].set_index("representation").loc[reps, metric]
            ax.bar(np.arange(4)+(j-.5)*.36, values, .36, label=cls, color=colour)
        ax.set_xticks(np.arange(4), ["Raw level", "Shape", "F18/IQR", "F246/IQR"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Share of cycle–seed comparisons (descriptive)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "training_reference_geometry")

    long = pd.read_csv(ROOT / "data/ress_revision/freeze_v1/long_primary.csv")
    raw = long.pivot(index="Experiment_Index", columns="Sample_Index", values="Measurement")
    meta = long.groupby("Experiment_Index").first()
    examples = pd.read_csv(OUT / "illustrative_examples.csv")
    fig, axes = plt.subplots(4, 2, figsize=(13, 12))
    for row_index, example in enumerate(examples.itertuples()):
        candidates = [(example.Experiment_Index, "Query", "#242424", "-"),
                      (example.normal_neighbor_id, "Training normal", "#2369a0", "--"),
                      (example.fault_neighbor_id, "Training same fault", "#b24b29", ":")]
        for column in range(2):
            ax = axes[row_index, column]
            for exp, label, colour, style in candidates:
                if pd.isna(exp):
                    continue
                exp = int(exp)
                signal = raw.loc[exp].to_numpy(float)
                if column:
                    std = signal.std()
                    signal = (signal-signal.mean())/std if std > 1e-9 else np.zeros_like(signal)
                text = f"{label}: {meta.loc[exp, 'Fault_Mode']}, {meta.loc[exp, 'Serial_Number']}, ID {exp}"
                ax.plot(raw.columns, signal, color=colour, linestyle=style, linewidth=1.25, label=text)
            ax.set_title(f"{example.Fault_Mode}, ID {example.Experiment_Index}: {example.miss_fraction:.0%} B1 misses; " + ("shape" if column else "raw signal"), fontsize=10)
            ax.set_ylabel("Standardised within cycle" if column else "Measurement units")
            ax.set_xlabel("Ordered measurement position (not elapsed time)")
            ax.legend(fontsize=7, loc="best", frameon=False)
            ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, "illustrative_signal_pairs")
    # Derived summaries for inspection, not a new statistical model.
    scores = pd.read_csv(OUT / "score_diagnostics.csv")
    scores["infinite_threshold"] = ~np.isfinite(scores.threshold)
    scores["finite_threshold"] = scores.threshold.where(np.isfinite(scores.threshold))
    scores.groupby(["arm", "Fault_Mode", "Serial_Number"]).agg(
        q_median=("score_q", "median"), threshold_median_finite=("finite_threshold", "median"),
        infinite_threshold_share=("infinite_threshold", "mean"),
        inner_class_recall_mean=("inner_class_recall", "mean"),
        inner_FA0_mean=("inner_FA0", "mean")).reset_index().to_csv(OUT / "score_by_unit.csv", index=False)
    print("E12: three figures exported as PDF and PNG; no classifier fits.")


if __name__ == "__main__":
    main()
