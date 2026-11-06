"""Check whether the no-F02 model separates failure from N01 inside the same unit, and where critical misses concentrate.
Output: resultados/binary_no_f02_within_unit.csv."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

R = "../resultados/"
predictions = pd.read_csv(R + "binary_no_f02_predictions.csv")
final = predictions[(predictions.feature_set == "D") & (predictions.model == "logistic")]
mean_score = final.groupby(["experimento_idx", "numero_serie", "condicao", "y"])[["score"]].mean().reset_index()

rows = []
for unit, frame in mean_score[mean_score.condicao != "N02"].groupby("numero_serie"):
    if frame.y.nunique() == 2:
        rows.append({"numero_serie": unit, "failures": ",".join(sorted(frame[frame.y == 1].condicao.unique())), "n_failure": int(frame.y.sum()),
                     "n_n01": int((frame.y == 0).sum()), "auc_within_unit": roc_auc_score(frame.y, frame.score),
                     "median_score_failure": frame[frame.y == 1].score.median(), "median_score_n01": frame[frame.y == 0].score.median()})
within = pd.DataFrame(rows)
within.to_csv(R + "binary_no_f02_within_unit.csv", index=False, lineterminator="\n")
pd.set_option("display.width", 200)
print(within.round(3).to_string(index=False))
sustained = within[within.n_failure >= 5]
print("units", len(within), "median within-unit AUC", round(within.auc_within_unit.median(), 3), "| sustained-failure units", len(sustained), "median", round(sustained.auc_within_unit.median(), 3),
      "| above 0.5:", int((sustained.auc_within_unit > 0.5).sum()), "sign test p", round(stats.binomtest(int((sustained.auc_within_unit > 0.5).sum()), len(sustained), 0.5).pvalue, 4))

alarm = final.assign(alarm=final.score > final.threshold_10)
critical = alarm[alarm.condicao.isin(["F03", "F06"])].groupby(["condicao", "numero_serie"]).alarm.agg(["size", "mean"])
critical["missed_share"] = (1 - critical["mean"]) * critical["size"] / ((1 - critical["mean"]) * critical["size"]).sum()
print(critical.round(3).to_string())
print(alarm[alarm.condicao == "F05"].groupby("numero_serie").alarm.mean().round(3).to_dict())
print(alarm[alarm.condicao == "N01"].groupby("numero_serie").alarm.mean().sort_values(ascending=False).head(6).round(3).to_dict())
