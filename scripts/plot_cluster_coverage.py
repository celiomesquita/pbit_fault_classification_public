"""Generate an exportable E3 coverage figure from completed simulation artefacts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = pd.read_csv(ROOT / "data/cluster_coverage/run/summary.csv")
    rows = rows[rows.event.eq("coverage95") & rows.rho.eq(0.8)]
    labels = {"percentile": "Percentile", "pairs_t": "Pairs bootstrap-t", "wild_rademacher": "Wild-t Rademacher",
              "wild_webb": "Wild-t Webb", "sign_swap": "Sign swap", "bca": "BCa"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True, layout="constrained")
    for i, profile in enumerate(["historical", "audited"]):
        for j, metric in enumerate(["CMR", "FA0"]):
            ax = axes[i, j]
            ax.axhline(0.95, color="0.35", linestyle="--", linewidth=0.8)
            ax.axhline(0.93, color="0.6", linestyle=":", linewidth=0.8)
            for method, label in labels.items():
                d = rows[rows.profile.eq(profile) & rows.metric.eq(metric) & rows.method.eq(method)].sort_values("delta")
                x = d.delta/d.margin
                ax.errorbar(x, d.rate, yerr=[d.rate-d.wilson_low, d.wilson_high-d.rate],
                            label=label, marker="o", markersize=3, capsize=2, linewidth=1)
            ax.set_title(f"{profile.capitalize()} — {metric}")
            ax.set_ylim(0.5, 1.01)
            ax.grid(axis="y", alpha=0.2)
            if i == 1:
                ax.set_xlabel(r"True effect / equivalence margin ($\delta/\epsilon$)")
            if j == 0:
                ax.set_ylabel("95% interval coverage")
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside lower center", ncol=3, frameon=False)
    destination = ROOT / "paper/supplement/figures/fig_cluster_coverage"
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
