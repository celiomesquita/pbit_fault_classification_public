from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_cv_summary import metrics_from_cm  # noqa: E402
from pbit.modeling import FAULT_ORDER, SEVERITY_MAP  # noqa: E402

DATA = PROJECT_ROOT / "data"
B = 2000
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
CIDX = {c: i for i, c in enumerate(FAULT_ORDER)}
KEYS = ["CMR", "DR", "macro_F1"]
BASE_MARGIN = {"CMR": 0.02, "DR": 0.02, "macro_F1": 0.03}
FACTORS = [0.5, 1.0, 1.5, 2.5]

# (group, label, arm A, arm B, metric, source of A, source of B)
COMPARISONS = [
    ("Famílias", "RF - LightGBM", "rf_F246", "lgbm_F246", "CMR"),
    ("Famílias", "RF - XGBoost", "rf_F246", "xgb_F246", "CMR"),
    ("Atributos (LightGBM)", "F246 - F115", "lgbm_F246", "lgbm_F115", "CMR"),
    ("Atributos (LightGBM)", "F246 - F115", "lgbm_F246", "lgbm_F115", "DR"),
    ("Atributos (LightGBM)", "F246 - F115", "lgbm_F246", "lgbm_F115", "macro_F1"),
    ("Atributos (LightGBM)", "F246 - F18", "lgbm_F246", "lgbm_F18", "CMR"),
    ("Atributos (LightGBM)", "F246 - F18", "lgbm_F246", "lgbm_F18", "DR"),
    ("Atributos (LightGBM)", "F246 - F18", "lgbm_F246", "lgbm_F18", "macro_F1"),
    ("Atributos (RF)", "F246 - F115", "rf_F246", "rf_F115", "CMR"),
    ("Atributos (RF)", "F246 - F18", "rf_F246", "rf_F18", "CMR"),
    ("Desbalanceamento (LightGBM)", "P1 pesos freq_sev - freq (gaussiana)", "lgbm_freq_sev_gaussian", "lgbm_freq_gaussian", "CMR"),
    ("Desbalanceamento (LightGBM)", "P2 gaussiana - sem ampliação (freq_sev)", "lgbm_freq_sev_gaussian", "lgbm_freq_sev_none", "macro_F1"),
    ("Desbalanceamento (LightGBM)", "P3 gaussiana - SMOTE (freq_sev)", "lgbm_freq_sev_gaussian", "lgbm_freq_sev_smote", "macro_F1"),
]


def load_all() -> pd.DataFrame:
    abl = pd.concat([pd.read_csv(DATA / "ablation" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    fam = pd.concat([pd.read_csv(DATA / "families" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    imb = pd.concat([pd.read_csv(DATA / "imbalance" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    key = ["seed", "Experiment_Index"]
    df = abl[key + ["Serial_Number", "Fault_Mode", "rf_F246", "lgbm_F246", "rf_F115", "lgbm_F115", "rf_F18", "lgbm_F18"]]
    df = df.merge(fam[key + ["xgb_F246"]], on=key, how="inner")
    imb_cols = ["lgbm_freq_sev_gaussian", "lgbm_freq_gaussian", "lgbm_freq_sev_none", "lgbm_freq_sev_smote"]
    df = df.merge(imb[key + imb_cols], on=key, how="inner")
    assert len(df) == len(abl) == len(fam) == len(imb), (len(df), len(abl), len(fam), len(imb))
    return df


def main() -> None:
    df = load_all()
    sns = sorted(df["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    arms = sorted({c[2] for c in COMPARISONS} | {c[3] for c in COMPARISONS})
    cms = {a: np.zeros((len(SEEDS), len(sns), 8, 8)) for a in arms}
    for si, seed in enumerate(SEEDS):
        d = df[df["seed"] == seed]
        ti, ni = d["Fault_Mode"].map(CIDX).values, d["Serial_Number"].map(sidx).values
        for a in arms:
            np.add.at(cms[a][si], (ni, ti, d[a].map(CIDX).values), 1)

    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    point, boot = {}, {}
    for a in arms:
        point[a] = dict(zip(KEYS, np.mean([[metrics_from_cm(cms[a][si].sum(axis=0))[k] for k in KEYS]
                                           for si in range(len(SEEDS))], axis=0)))
        arr = np.zeros((B, len(KEYS)))
        for b in range(B):
            acc = np.zeros(len(KEYS))
            for si in range(len(SEEDS)):
                m = metrics_from_cm(np.tensordot(W[b], cms[a][si], axes=(0, 0)))
                acc += np.array([m[k] for k in KEYS])
            arr[b] = acc / len(SEEDS)
        boot[a] = pd.DataFrame(arr, columns=KEYS)

    train = pd.read_csv(DATA / "train_real.csv")
    n3 = int(train["Fault_Mode"].isin(["F03", "F06"]).sum())
    n3_units = int(train.loc[train["Fault_Mode"].isin(["F03", "F06"]), "Serial_Number"].nunique())

    rows = []
    for group, label, A, Bf, m in COMPARISONS:
        d = boot[A][m] - boot[Bf][m]
        lo90, hi90 = np.percentile(d, [5, 95])
        lo95, hi95 = np.percentile(d, [2.5, 97.5])
        dmin = max(-lo90, hi90)
        base = BASE_MARGIN[m]
        row = {"grupo": group, "comparacao": label, "metrica": m,
               "dif_pontual": point[A][m] - point[Bf][m],
               "IC95_inf": lo95, "IC95_sup": hi95, "IC90_inf": lo90, "IC90_sup": hi90,
               "delta_min": dmin, "margem_adotada": base,
               "classe": ("robusta" if dmin <= base else
                          "dependente da margem (nao equivalente)" if dmin <= 2.5 * base else
                          "nao equivalente em nenhuma margem da grade")}
        for f in FACTORS:
            row[f"equiv_{f*base:.3f}"] = bool(dmin < f * base)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "margin_sensitivity.csv", index=False)

    def md(frame: pd.DataFrame) -> str:
        head = "| " + " | ".join(map(str, frame.columns)) + " |"
        sep = "|" + "|".join("---" for _ in frame.columns) + "|"
        body = ["| " + " | ".join(str(v) for v in r) + " |" for r in frame.itertuples(index=False)]
        return "\n".join([head, sep, *body])

    tab = []
    for r in rows:
        base = r["margem_adotada"]
        cells = {f"{f}x": ("sim" if r[f"equiv_{f*base:.3f}"] else "não") + f" ({f*base:.3f})" for f in FACTORS}
        tab.append({"grupo": r["grupo"], "comparacao": r["comparacao"], "metrica": r["metrica"],
                    "dif": f"{r['dif_pontual']:+.4f}", "IC90": f"[{r['IC90_inf']:+.5f}; {r['IC90_sup']:+.5f}]",
                    "delta_min": f"{r['delta_min']:.5f}", "margem_adotada": f"{base:.2f}",
                    **cells, "classe": r["classe"]})
    TEX_LABEL = {"Famílias": "Families", "Atributos (LightGBM)": "Features, LightGBM", "Atributos (RF)": "Features, RF",
                 "Desbalanceamento (LightGBM)": "Imbalance, LightGBM"}
    TEX_CMP = {"RF - LightGBM": "RF $-$ LightGBM", "RF - XGBoost": "RF $-$ XGBoost", "F246 - F115": "F246 $-$ F115",
               "F246 - F18": "F246 $-$ F18", "P1 pesos freq_sev - freq (gaussiana)": "P1: severity-frequency $-$ frequency weights",
               "P2 gaussiana - sem ampliação (freq_sev)": "P2: Gaussian $-$ no augmentation",
               "P3 gaussiana - SMOTE (freq_sev)": "P3: Gaussian $-$ SMOTE"}
    TEX_MET = {"CMR": "CMR", "DR": "DR", "macro_F1": "Macro-F1"}

    def sgn(x):
        return f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")

    lines = ["% Source: data/margin_sensitivity_summary.md (scripts/margin_sensitivity.py); IC90 of the paired difference, ten paired seeds, 54 serial numbers.",
             "\\begin{tabular}{llcccccccc}", "\\toprule",
             "\\textbf{Group} & \\textbf{Comparison} & \\textbf{Metric} & \\textbf{Difference} & \\textbf{90\\% interval} & "
             "$\\delta_{\\min}$ & \\makecell{\\textbf{Adopted}\\\\\\textbf{margin}} & \\makecell{\\textbf{Equivalent at}\\\\\\textbf{0.5$\\times$ / 1.5$\\times$ / 2.5$\\times$}} \\\\",
             "\\midrule"]
    prev = None
    for r in rows:
        base = r["margem_adotada"]
        yn = " / ".join("yes" if r[f"equiv_{f*base:.3f}"] else "no" for f in (0.5, 1.5, 2.5))
        grp = TEX_LABEL[r["grupo"]]
        lines.append(("\\midrule\n\t" if prev is not None and prev != grp else "\t") +
                     f"{grp if grp != prev else ''} & {TEX_CMP[r['comparacao']]} & {TEX_MET[r['metrica']]} & {sgn(r['dif_pontual'])} & "
                     f"[{sgn(r['IC90_inf'])}; {sgn(r['IC90_sup'])}] & {r['delta_min']:.3f} & {base:.2f} & {yn} \\\\")
        prev = grp
    lines += ["\t\\bottomrule", "\\end{tabular}", ""]
    (PROJECT_ROOT / "paper" / "supplement" / "tables" / "tab_margin_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")

    out = ["# Sensibilidade das conclusões de equivalência à margem (Ação 16)", "",
           f"Predições salvas das Ações 6, 7 e 14; 10 sementes pareadas, 54 SN, {B} reamostragens de SN (semente 42). "
           "δ_min é a menor margem simétrica que contém o IC90 da diferença pareada; a equivalência vale em uma margem δ "
           "quando δ_min < δ. Os fatores 0,5, 1, 1,5 e 2,5 multiplicam as margens adotadas "
           "(0,02 em CMR e risco de rebaixamento; 0,03 em macro-F1).", "",
           f"Contexto operacional: a partição de treino tem {n3} ciclos de severidade 3 (F03 e F06) em {n3_units} SN; "
           f"a margem de 0,02 em CMR equivale a {0.02 * n3:.1f} ciclos desses {n3} (cerca de {0.02 * n3_units:.2f} unidade em {n3_units}).", "",
           md(pd.DataFrame(tab)), ""]
    (DATA / "margin_sensitivity_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
