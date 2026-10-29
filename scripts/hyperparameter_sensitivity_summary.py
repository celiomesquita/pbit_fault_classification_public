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
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
KEYS = ["CMR", "FA0", "DR", "R3", "macro_F1"]
MARGIN = {"CMR": 0.02, "DR": 0.02, "macro_F1": 0.03}
LOWER_BETTER = {"CMR": True, "FA0": True, "DR": True, "R3": False, "macro_F1": False}


def metrics(cm):
    m = metrics_from_cm(cm)
    s0 = SEV == 0
    out = {k: m[k] for k in ("CMR", "DR", "R3", "macro_F1")}
    out["FA0"] = cm[np.ix_(s0, SEV >= 1)].sum() / cm[s0, :].sum()
    return out


def md(df):
    df = df.reset_index(drop=True)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    d = pd.concat([pd.read_csv(DATA / "hpsens" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    abl = pd.concat([pd.read_csv(DATA / "ablation" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    chk = d.merge(abl[["seed", "Experiment_Index", "lgbm_F246", "rf_F246"]], on=["seed", "Experiment_Index"])
    assert len(chk) == len(d)
    same_l, same_r = (chk.lgbm_base == chk.lgbm_F246).mean(), (chk.rf_base == chk.rf_F246).mean()
    arms = [c for c in d.columns if c.startswith(("lgbm_base", "rf_base", "L", "R")) and c not in ("Fault_Mode",)]
    arms = [c for c in arms if c in d.columns and c not in ("Experiment_Index",)]
    sns = sorted(d["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    cms = {a: np.zeros((len(SEEDS), len(sns), 8, 8)) for a in arms}
    for si, seed in enumerate(SEEDS):
        g = d[d["seed"] == seed]
        ti, ni = g["Fault_Mode"].map(CIDX).values, g["Serial_Number"].map(sidx).values
        for a in arms:
            np.add.at(cms[a][si], (ni, ti, g[a].map(CIDX).values), 1)
    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    point, boot = {}, {}
    for a in arms:
        point[a] = dict(zip(KEYS, np.mean([[metrics(cms[a][si].sum(axis=0))[k] for k in KEYS]
                                           for si in range(len(SEEDS))], axis=0)))
        arr = np.zeros((B, len(KEYS)))
        for b in range(B):
            acc = np.zeros(len(KEYS))
            for si in range(len(SEEDS)):
                m = metrics(np.tensordot(W[b], cms[a][si], axes=(0, 0)))
                acc += np.array([m[k] for k in KEYS])
            arr[b] = acc / len(SEEDS)
        boot[a] = pd.DataFrame(arr, columns=KEYS)

    def ci(x, lo=2.5, hi=97.5):
        return f"[{np.percentile(x, lo):+.5f}; {np.percentile(x, hi):+.5f}]"

    def verdict(metric, x):
        lo95, hi95 = np.percentile(x, [2.5, 97.5])
        lo90, hi90 = np.percentile(x, [5, 95])
        sup = ""
        if lo95 > 0 or hi95 < 0:
            sup = "variação superior" if ((hi95 < 0) == LOWER_BETTER[metric]) else "base superior"
        eq = metric in MARGIN and lo90 > -MARGIN[metric] and hi90 < MARGIN[metric]
        if sup and eq:
            return f"{sup} (distinta de zero, mas dentro da margem: equivalente)"
        return sup or ("equivalente" if eq else "inconclusivo")

    def pval(x):
        x = np.asarray(x)
        return min(1.0, 2 * min((np.sum(x <= 0) + 1) / (len(x) + 1), (np.sum(x >= 0) + 1) / (len(x) + 1)))

    out = ["# Sensibilidade dos hiperparâmetros fixos (Ação 17)", "",
           f"F246, partição de treino (54 SN), 10 sementes pareadas, {B} reamostragens de SN (semente 42). "
           f"Verificação do braço-base contra as predições salvas da Ação 6: concordância de {same_l:.4f} (LightGBM) e {same_r:.4f} (RF) "
           f"em {len(chk)} predições.", ""]
    rows = []
    for fam, base in (("LightGBM", "lgbm_base"), ("RF", "rf_base")):
        for a in [x for x in arms if (x.startswith("L") or x == "lgbm_base")] if fam == "LightGBM" else [x for x in arms if (x.startswith("R") or x == "rf_base")]:
            p, b = point[a], boot[a]
            rows.append({"familia": fam, "braco": a, "CMR": f"{p['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]).replace('+', ''),
                         "FA0": f"{p['FA0']:.3f}", "DR": f"{p['DR']:.3f}", "R3": f"{p['R3']:.3f}",
                         "macro_F1": f"{p['macro_F1']:.3f}"})
    out += ["## Métricas por braço", "", md(pd.DataFrame(rows)), ""]

    def compare(base, variants, holm: bool):
        crows, pv = [], []
        for a in variants:
            for m in ("CMR", "FA0", "DR", "macro_F1"):
                x = boot[a][m] - boot[base][m]
                crows.append({"variacao": a, "metrica": m, "dif_pontual": f"{point[a][m] - point[base][m]:+.4f}",
                              "IC95": ci(x), "IC90": ci(x, 5, 95), "p_bootstrap": f"{pval(x):.4f}",
                              "veredito": verdict(m, x)})
                if holm and m == "CMR":
                    pv.append((len(crows) - 1, pval(x)))
        if holm:
            run = 0.0
            for rank, (ri, p_) in enumerate(sorted(pv, key=lambda z: z[1])):
                run = max(run, min(1.0, (len(pv) - rank) * p_))
                crows[ri]["p_Holm"] = f"{run:.4f}"
        return crows

    lvars = [a for a in arms if a.startswith("L")]
    rvars = [a for a in arms if a.startswith("R")]
    lrows = compare("lgbm_base", lvars, True)
    out += ["## LightGBM: variação − base (família primária: CMR, 13 testes, Holm)", "",
            "Diferença negativa em CMR e FA0 é favorável à variação; positiva em macro-F1 é favorável à variação.", "",
            md(pd.DataFrame(lrows).fillna("")), ""]
    rrows = compare("rf_base", rvars, False)
    out += ["## Random Forest: variação − base (secundárias, exploratórias)", "", md(pd.DataFrame(rrows)), ""]
    LABEL = {"L01_trees100": "100 trees", "L02_trees200": "200 trees", "L03_trees800": "800 trees",
             "L04_lr0.02": "learning rate 0.02", "L05_lr0.10": "learning rate 0.10",
             "L06_leaves15": "15 leaves", "L07_leaves31": "31 leaves", "L08_leaves127": "127 leaves",
             "L09_mcs2": "minimum child samples 2", "L10_mcs20": "minimum child samples 20",
             "L11_colsample0.5": "column subsampling 0.5", "L12_colsample1.0": "column subsampling 1.0",
             "L13_bagging_freq1": "row bagging enabled (0.8)",
             "R01_trees100": "100 trees", "R02_trees1000": "1000 trees", "R03_msl1": "minimum leaf size 1",
             "R04_msl5": "minimum leaf size 5", "R05_maxfeat_log2": r"features per split $\log_2 p$",
             "R06_maxfeat0.3": "features per split $0.3p$"}

    def sg(v):
        return f"{v:+.3f}".replace("-", "$-$").replace("+", "$+$")

    tex = ["% Source: data/hpsens_summary.md (scripts/hyperparameter_sensitivity.py, scripts/hyperparameter_sensitivity_summary.py); ten paired seeds, 54 serial numbers.",
           r"\begin{tabular}{llcccc}", r"\toprule",
           r"\textbf{Model} & \textbf{Setting} & \textbf{CMR} & \textbf{FA0} & \textbf{Macro-F1} & "
           r"\makecell{\textbf{CMR minus base}\\\textbf{[95\% CI]}} \\",
           r"\midrule"]
    for fam, base, names in (("LightGBM", "lgbm_base", lvars), ("Random Forest", "rf_base", rvars)):
        pb = point[base]
        tex.append(f"\t{fam} & base setting & {pb['CMR']:.3f} & {pb['FA0']:.3f} & {pb['macro_F1']:.3f} & --- \\\\")
        for a in names:
            x = boot[a]["CMR"] - boot[base]["CMR"]
            lo, hi = np.percentile(x, [2.5, 97.5])
            pa = point[a]
            tex.append(f"\t & {LABEL[a]} & {pa['CMR']:.3f} & {pa['FA0']:.3f} & {pa['macro_F1']:.3f} & "
                       f"{sg(pa['CMR'] - pb['CMR'])} [{sg(lo)}; {sg(hi)}] \\\\")
        if fam == "LightGBM":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    (PROJECT_ROOT / "paper" / "supplement" / "tables" / "tab_hpsens.tex").write_text("\n".join(tex), encoding="utf-8")
    pd.DataFrame(lrows + rrows).to_csv(DATA / "hpsens_comparisons.csv", index=False)
    (DATA / "hpsens_summary.md").write_text("\n".join(out), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
