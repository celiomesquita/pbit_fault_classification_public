"""Summarise unit coverage and signal structure; export reproducible exploratory figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]
COLORS = ["#237a55", "#59905b", "#b38326", "#bb5c32", "#a53949", "#7a5ca2", "#546aab", "#273e62"]


def main():
    data = pd.read_csv(ROOT / "dados/pbit_features_v3.csv")
    spectral = pd.read_csv(ROOT / "dados/pbit_spectral_features_v3.csv")
    merged = data.merge(spectral, on="experimento_idx", validate="one_to_one")
    results, figures = ROOT / "resultados", ROOT / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    n01 = data[data.condicao == "N01"].groupby("numero_serie").media.agg(
        cycles="size", median="median", mean="mean", sd="std",
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    n01["coefficient_of_variation_percent"] = 100 * n01.sd / n01["mean"].abs()
    n01.to_csv(results / "exploratory_n01_by_unit.csv", lineterminator="\n")
    unit_condition = merged.groupby(["numero_serie", "condicao"]).agg(
        cycles=("experimento_idx", "size"), signal_mean_median=("media", "median"),
        low_frequency_share_median=("spectral_low_share", "median"))
    unit_condition.to_csv(results / "exploratory_unit_condition.csv", lineterminator="\n")
    # Fault level against N01 in the same unit; F06 is excluded because its definition rests on the signal level.
    levels = unit_condition.signal_mean_median.unstack()
    shift = levels[["F01", "F02", "F03", "F04", "F05"]].stack().rename("fault_median").reset_index()
    shift["n01_median"] = shift.numero_serie.map(levels["N01"])
    shift = shift.dropna(subset=["n01_median"])
    shift["fault_lower"] = shift.fault_median < shift.n01_median
    shift.to_csv(results / "exploratory_level_shift.csv", index=False, lineterminator="\n")
    class_spectral = unit_condition.low_frequency_share_median.groupby("condicao").median()
    n02_counts = data[data.condicao == "N02"].groupby("numero_serie").size()
    unit_counts = data.groupby("numero_serie").size()
    summaries = {
        "cycles": len(data), "units": data.numero_serie.nunique(),
        "units_with_multiple_conditions": int((data.groupby("numero_serie").condicao.nunique() > 1).sum()),
        "cycles_per_unit_min": unit_counts.min(), "cycles_per_unit_median": unit_counts.median(),
        "cycles_per_unit_max": unit_counts.max(),
        "n02_dominant_unit_share": n02_counts.max() / n02_counts.sum(),
        "n01_unit_median_level_min": n01["median"].min(),
        "n01_unit_median_level_max": n01["median"].max(),
        "n01_coefficient_of_variation_percent_median": n01.coefficient_of_variation_percent.median(),
        "n01_coefficient_of_variation_percent_max": n01.coefficient_of_variation_percent.max(),
        **{f"{condition.lower()}_low_share_median_across_units": class_spectral[condition] for condition in CONDITIONS},
        "within_unit_fault_lower_than_n01": int(shift.fault_lower.sum()), "within_unit_fault_comparisons": len(shift),
        "peak_to_peak_identity_max_error": (data.pico_a_pico - (data.maximo - data.minimo)).abs().max(),
        "start_end_identity_max_error": (data.delta_inicio_fim - (data.valor_final - data.valor_inicial)).abs().max(),
        "energy_identity_max_error": (data.energia - 123 * data.rms**2).abs().max(),
    }
    pd.DataFrame(summaries.items(), columns=["quantity", "value"]).to_csv(
        results / "exploratory_summary.csv", index=False, lineterminator="\n")

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(6.1, 5.2), layout="constrained")
    ordered = n01.sort_values("median")
    ranks = np.arange(1, len(ordered) + 1)
    axes[0].errorbar(ranks, ordered["median"],
                    yerr=[ordered["median"] - ordered.q25, ordered.q75 - ordered["median"]],
                    fmt="o", markersize=3, color=COLORS[0], ecolor="#61766c", capsize=2)
    axes[0].set(title="(a) Normal-condition level varies between units",
                xlabel="Unit rank by median N01 level (not time)", ylabel="Signal level")
    axes[0].set_xticks([1, 5, 10, 15, 20, 25, 30, 34])

    values = [unit_condition.xs(condition, level="condicao").low_frequency_share_median.to_numpy()
              for condition in CONDITIONS]
    boxes = axes[1].boxplot(values, tick_labels=CONDITIONS, patch_artist=True,
                            widths=.5, showfliers=False, medianprops={"color": "black"})
    for i, (box, samples, color) in enumerate(zip(boxes["boxes"], values, COLORS), start=1):
        box.set_facecolor(color)
        box.set_alpha(.22)
        offset = np.linspace(-.13, .13, len(samples))
        axes[1].scatter(i + offset, np.sort(samples), color=color, s=11, zorder=3)
    axes[1].set(title="(b) Low-frequency content by unit and condition",
                xlabel="Recorded condition", ylabel="Low-frequency power share", ylim=(-.03, 1.03))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.savefig(figures / "fig_exploratory_analysis.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(figures / "fig_exploratory_analysis.png", dpi=200)
    plt.close(fig)
    print("Exploratory summaries and figures exported")


if __name__ == "__main__":
    main()
