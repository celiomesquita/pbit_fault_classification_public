"""Plot E6 partition distributions; whiskers are descriptive, not confidence bounds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data/crc_repeated_partitions/results.csv")
assert len(data) == 3000
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), sharey="row", layout="constrained")
alphas = [.10, .125, .15, .20]
for column, n in enumerate([7, 9, 11]):
    selected = data[data.n_cal.eq(n)]
    groups = [selected[selected.alpha.isna()]] + [selected[selected.alpha.eq(a)] for a in alphas]
    for row, metric in enumerate(["mean_unit_loss", "FA0"]):
        ax = axes[row, column]
        arrays = [g[metric].dropna().to_numpy() for g in groups]
        bp = ax.boxplot(arrays, positions=np.arange(5), whis=(5, 95), showfliers=False,
                        patch_artist=True, widths=.48,
                        medianprops={"color": "#333333", "linewidth": 1.1})
        for box in bp["boxes"]:
            box.set(facecolor="#dbe9f2", edgecolor="#56788c")
        ax.scatter(np.arange(5), [v.mean() for v in arrays], color="#14669a", s=23, zorder=4)
        if row == 0:
            ax.scatter(np.arange(1, 5), alphas, marker="D", color="#b24b35", s=21, zorder=5)
            ax.set_title(f"{n} calibration / {14-n} critical training units")
        ax.set_xticks(np.arange(5), ["Base", ".100", ".125", ".150", ".200"])
        ax.set_ylim(-.025, 1.025)
        ax.grid(axis="y", color="#e4e4e4", linewidth=.6)
        if row == 1:
            ax.set_xlabel("Target unit loss")
axes[0, 0].set_ylabel("Realised unit-average critical loss")
axes[1, 0].set_ylabel("False-alarm rate on normal cycles")
fig.legend(handles=[Line2D([], [], color="#14669a", marker="o", linestyle="none", label="Mean over partitions"),
                    Line2D([], [], color="#b24b35", marker="D", linestyle="none", label="Target expected loss")],
           loc="outside lower center", ncol=2, frameon=False)
destination = ROOT / "paper/supplement/figures/fig_crc_repeated"
fig.savefig(destination.with_suffix(".pdf"))
fig.savefig(destination.with_suffix(".png"), dpi=180)
plt.close(fig)
