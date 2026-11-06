"""Compare the no-F02 procedure with and without frequency descriptors, paired by unit, and repeat the within-unit check.
Usage, from scripts/: py compare_spectral_gain.py. Outputs: resultados/spectral_gain.csv, resultados/binary_no_f02_spectral_within_unit.csv."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import experimentos_base as base
import train_binary_model as model

R = base.RAIZ / "resultados"
N_BOOT = 2000


def procedure(prefix):
    predictions = pd.read_csv(R / f"{prefix}_predictions.csv")
    inner = pd.read_csv(R / f"{prefix}_inner_selection.csv")
    return predictions, model.procedure_predictions(predictions, inner)[0]


def main():
    _, without = procedure("binary_no_f02")
    predictions, with_frequency = procedure("binary_no_f02_spectral")
    units = sorted(without["numero_serie"].unique())
    position = {unit: i for i, unit in enumerate(units)}
    draws = np.random.default_rng(2026).multinomial(len(units), np.full(len(units), 1 / len(units)), size=N_BOOT)

    def values(frame, point):
        alarm = (frame["score"] >= 0.5) if point == "default" else (frame["score"] > frame[f"threshold_{point}"])
        unit = frame["numero_serie"].map(position).values
        out = {}
        for metric, mask, sign in (("recall", frame["y"].values == 1, alarm), ("false_alarm", frame["y"].values == 0, alarm),
                                   ("critical_miss", frame["condicao"].isin(model.CRITICAL).values, ~alarm)):
            hits, total = np.zeros(len(units)), np.zeros(len(units))
            np.add.at(hits, unit[mask], sign.values[mask].astype(float))
            np.add.at(total, unit[mask], 1.0)
            out[metric] = (hits, total)
        return out

    rows = []
    for point in ("default", "10", "20"):
        first, second = values(without, point), values(with_frequency, point)
        for metric in first:
            a_hits, a_total = first[metric]
            b_hits, b_total = second[metric]
            delta = (draws @ b_hits) / (draws @ b_total) - (draws @ a_hits) / (draws @ a_total)
            rows.append({"point": point, "metric": metric, "without_frequency": a_hits.sum() / a_total.sum(), "with_frequency": b_hits.sum() / b_total.sum(),
                         "difference": b_hits.sum() / b_total.sum() - a_hits.sum() / a_total.sum(), "ci_low": np.nanquantile(delta, 0.025), "ci_high": np.nanquantile(delta, 0.975)})
    gain = pd.DataFrame(rows)
    gain.to_csv(R / "spectral_gain.csv", index=False, lineterminator="\n")
    print(gain.round(3).to_string(index=False))

    final = predictions[(predictions.feature_set == "E") & (predictions.model == "logistic")]
    mean_score = final.groupby(["experimento_idx", "numero_serie", "condicao", "y"])[["score"]].mean().reset_index()
    rows = []
    for unit, frame in mean_score[mean_score.condicao != "N02"].groupby("numero_serie"):
        if frame.y.nunique() == 2:
            rows.append({"numero_serie": unit, "failures": ",".join(sorted(frame[frame.y == 1].condicao.unique())), "n_failure": int(frame.y.sum()),
                         "auc_within_unit": roc_auc_score(frame.y, frame.score)})
    within = pd.DataFrame(rows)
    within.to_csv(R / "binary_no_f02_spectral_within_unit.csv", index=False, lineterminator="\n")
    print(within.round(3).to_string(index=False))
    sustained = within[within.n_failure >= 5]
    above = int((sustained.auc_within_unit > 0.5).sum())
    print("sustained units", len(sustained), "median", round(sustained.auc_within_unit.median(), 3), "above 0.5:", above, "p", round(stats.binomtest(above, len(sustained), 0.5).pvalue, 4))
    print("sum of alarms normal:", final[final.y == 0].assign(a=final.score > final.threshold_10).groupby("condicao").a.agg(["sum", "size"]).to_dict())


if __name__ == "__main__":
    main()
