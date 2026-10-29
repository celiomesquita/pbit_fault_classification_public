from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

DATA = Path(__file__).resolve().parents[1] / "data"
SEV = {"N01": 0, "N02": 0, "F01": 1, "F02": 2, "F04": 2, "F05": 2, "F03": 3, "F06": 3}
MODELS = {"F1-tuned": "pred_main_F1-tuned", "CMR-tuned": "pred_main_CMR-tuned"}
B = 5000


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    d = pd.read_csv(DATA / "final_test_predictions.csv")
    d["sev_true"] = d["Fault_Mode"].map(SEV)
    out = ["# Incerteza no nível da unidade — teste final (Ação 9)", "",
           f"Teste retido: {len(d)} ciclos, {d['Serial_Number'].nunique()} SN. Previsões regeneradas e idênticas às do artigo "
           "em CMR, risco de rebaixamento, erro de severidade e macro-F1 (Ação 4).", ""]

    comp = d.groupby(["Serial_Number", "Fault_Mode"]).size().unstack(fill_value=0)
    out += ["## Composição do teste por SN e classe (ciclos)", "", md(comp), ""]
    sev3 = d[d["sev_true"] == 3]
    out += [f"Ciclos de Sev3: {len(sev3)}, em {sev3['Serial_Number'].nunique()} SN "
            f"({', '.join(f'{k}: {v} ciclos' for k, v in sev3.groupby('Serial_Number').size().items())}).", ""]

    rows = []
    for name, col in MODELS.items():
        d[f"sev_{name}"] = d[col].map(SEV)
        s3 = d[d["sev_true"] == 3]
        miss = s3[s3[f"sev_{name}"] == 0]
        k, n = len(miss), len(s3)
        lo, hi = cp(k, n)
        units = s3.groupby("Serial_Number").apply(lambda g: int((g[f"sev_{name}"] == 0).any()), include_groups=False)
        ku, nu = int(units.sum()), len(units)
        lou, hiu = cp(ku, nu)
        rows.append({
            "modelo": name, "ciclos_criticos_perdidos": f"{k}/{n}",
            "CMR_por_ciclo_IC95_exato": f"{k / n:.3f} [{lo:.3f}; {hi:.3f}]",
            "unidades_Sev3_com_perda": f"{ku}/{nu}",
            "taxa_por_unidade_IC95_exato": f"{ku / nu:.2f} [{lou:.3f}; {hiu:.3f}]",
        })
    out += ["## CMR: por ciclo e por unidade (intervalos exatos de Clopper–Pearson)", "", md(pd.DataFrame(rows).set_index("modelo")), "",
            "O intervalo por ciclo trata os 74 ciclos como independentes. As perdas e os acertos estão concentrados em duas unidades; "
            "o intervalo por unidade (2 unidades de Sev3) é a medida adequada da incerteza entre unidades.", ""]

    rows = []
    for name in MODELS:
        s3 = d[d["sev_true"] == 3]
        for sn, g in s3.groupby("Serial_Number"):
            rows.append({"modelo": name, "SN": sn, "classe": ", ".join(sorted(g["Fault_Mode"].unique())),
                         "ciclos": len(g), "perdidos_para_Sev0": int((g[f"sev_{name}"] == 0).sum()),
                         "mantidos_em_Sev3": int((g[f"sev_{name}"] == 3).sum()),
                         "rebaixados_Sev1_2": int(((g[f"sev_{name}"] > 0) & (g[f"sev_{name}"] < 3)).sum())})
    out += ["## Ciclos de Sev3 por unidade", "", md(pd.DataFrame(rows).set_index("modelo")), ""]

    rows = []
    n01 = d[d["Fault_Mode"] == "N01"]
    for name, col in MODELS.items():
        per = n01.groupby("Serial_Number").apply(
            lambda g: pd.Series({"n": len(g), "correct": int((g[col] == "N01").sum()),
                                 "alarm": int((g[f"sev_{name}"] >= 1).sum())}), include_groups=False)
        per["recall_N01"] = (per["correct"] / per["n"]).round(2)
        per["FA0"] = (per["alarm"] / per["n"]).round(2)
        rng = np.random.default_rng(42)
        recs, fas = [], []
        for _ in range(B):
            idx = rng.integers(0, len(per), len(per))
            sel = per.iloc[idx]
            recs.append(sel["correct"].sum() / sel["n"].sum())
            fas.append(sel["alarm"].sum() / sel["n"].sum())
        rows.append({"modelo": name, "unidades_N01": len(per), "ciclos_N01": int(per["n"].sum()),
                     "recall_N01_agregado": f"{per['correct'].sum() / per['n'].sum():.3f}",
                     "IC95_bootstrap_SN": f"[{np.percentile(recs, 2.5):.3f}; {np.percentile(recs, 97.5):.3f}]",
                     "FA0_N01_agregado": f"{per['alarm'].sum() / per['n'].sum():.3f}",
                     "IC95_FA0": f"[{np.percentile(fas, 2.5):.3f}; {np.percentile(fas, 97.5):.3f}]",
                     "recall_por_SN_min_max": f"{per['recall_N01'].min():.2f}–{per['recall_N01'].max():.2f}"})
        out += [f"## Ciclos de N01 por unidade — {name}", "", md(per.reset_index().set_index("Serial_Number")), ""]
    out += ["## Resumo de N01 (bootstrap por SN, 5.000 reamostragens)", "", md(pd.DataFrame(rows).set_index("modelo")), ""]

    (DATA / "final_test_uncertainty.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
