from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SEV = {"N01": 0, "N02": 0, "F01": 1, "F02": 2, "F04": 2, "F05": 2, "F03": 3, "F06": 3}
SN_OF = {917: "SN 045", 920: "SN 045", 950: "SN 045", 951: "SN 045", 1178: "SN 052", 1227: "SN 054", 1258: "SN 056"}


def md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(DATA / "loso_f06" / "preds_*.csv")))], ignore_index=True)
    d["sev_true"] = d["Fault_Mode"].map(SEV)
    d["sev_pred"] = d["pred"].map(SEV)
    f6 = d[(d["Fault_Mode"] == "F06")]
    out = ["# Leave-one-SN-out para F06 — resumo (Ação 8)", "",
           "5 sementes (42, 0, 7, 13, 21). 'F06' = fração das sementes que preveem F06; 'Sev3' = fração que prevê F03 ou F06; "
           "'Sev0' = fração que prevê N01 ou N02.", ""]

    def cyc_table(df):
        g = df.groupby(["dataset", "target", "Experiment_Index"])
        t = pd.DataFrame({
            "F06": g["pred"].apply(lambda s: (s == "F06").mean()).round(2),
            "Sev3": g["sev_pred"].apply(lambda s: (s == 3).mean()).round(2),
            "Sev0": g["sev_pred"].apply(lambda s: (s == 0).mean()).round(2),
            "classe_modal": g["pred"].agg(lambda s: s.value_counts().idxmax()),
            "distribuicao": g["pred"].agg(lambda s: ", ".join(f"{k} = {v}" for k, v in s.value_counts().items())),
        })
        return t

    main_ = f6[(f6["variant"] == "V0") & (f6["model"] == "M1_lgbm_F246")]
    out += ["## M1 (LightGBM F246, ampliação), V0: todas as demais SN de F06 no treino — por ciclo de F06", "",
            md(cyc_table(main_)), ""]

    rows = []
    for (ds, tgt, var, mdl), g in f6[f6["variant"] != "V0"].groupby(["dataset", "target", "variant", "model"]):
        if mdl != "M1_lgbm_F246" or ds != "D_primary":
            continue
        rows.append({"alvo": tgt, "fonte_do_F06_no_treino": var.replace("V1-", ""),
                     "ciclos_F06": g["Experiment_Index"].nunique(),
                     "F06": round((g["pred"] == "F06").mean(), 2), "Sev3": round((g["sev_pred"] == 3).mean(), 2),
                     "Sev0": round((g["sev_pred"] == 0).mean(), 2)})
    out += ["## M1, D_primary: uma única SN-fonte de F06 no treino (fração pooled sobre ciclos e sementes)", "",
            md(pd.DataFrame(rows).set_index("alvo")), ""]

    rows = []
    for (ds, tgt, mdl), g in f6[(f6["variant"] == "V0") & (~f6["model"].str.startswith("BIN"))].groupby(["dataset", "target", "model"]):
        if ds != "D_primary":
            continue
        rows.append({"alvo": tgt, "modelo": mdl, "F06": round((g["pred"] == "F06").mean(), 2),
                     "Sev3": round((g["sev_pred"] == 3).mean(), 2), "Sev0": round((g["sev_pred"] == 0).mean(), 2)})
    out += ["## Modelos M1–M4, V0, D_primary (fração pooled sobre ciclos de F06 e sementes)", "",
            md(pd.DataFrame(rows).set_index("alvo")), ""]

    rows = []
    for (ds, tgt, mdl), g in f6[f6["model"].str.startswith("BIN")].groupby(["dataset", "target", "model"]):
        rows.append({"dataset": ds, "alvo": tgt, "modelo_binario": mdl, "ciclos_F06": g["Experiment_Index"].nunique(),
                     "recall_F06": round((g["pred"] == "F06").mean(), 2)})
    out += ["## Variante binária F03 × F06 (reconciliação com a Seção 5.5)", "",
            md(pd.DataFrame(rows).set_index("dataset")), ""]

    rows = []
    others = d[(d["Fault_Mode"] != "F06") & (d["variant"] == "V0") & (d["model"] == "M1_lgbm_F246") & (d["dataset"] == "D_primary")]
    for tgt, g in others.groupby("target"):
        for fm, h in g.groupby("Fault_Mode"):
            rows.append({"alvo": tgt, "classe": fm, "ciclos": h["Experiment_Index"].nunique(),
                         "acuracia_de_classe": round((h["pred"] == h["Fault_Mode"]).mean(), 2),
                         "previsto_Sev_ge1": round((h["sev_pred"] >= 1).mean(), 2),
                         "previsto_como_F06": round((h["pred"] == "F06").mean(), 2)})
    out += ["## Demais ciclos da unidade deixada fora (M1, V0, D_primary)", "", md(pd.DataFrame(rows).set_index("alvo")), ""]

    (DATA / "loso_f06_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
