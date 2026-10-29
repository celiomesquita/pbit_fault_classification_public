from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_cv_summary import metrics_from_cm  # noqa: E402
from pbit.modeling import FAULT_ORDER, SEVERITY_MAP  # noqa: E402

DATA = PROJECT_ROOT / "data"
B = 2000
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
CIDX = {c: i for i, c in enumerate(FAULT_ORDER)}
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
MARGIN = {"CMR": 0.02, "DR": 0.02, "macro_F1": 0.03}
LOWER_BETTER = {"CMR": True, "DR": True, "macro_F1": False, "FA0": True, "R3": False}
ARMS = {"RF": "rf_F246", "LightGBM": "lgbm_F246", "XGBoost": "xgb_F246", "GradientBoosting": "gb_F246"}
KEYS = ["CMR", "DR", "R3", "macro_F1", "FA0"]


def metrics(cm):
    m = metrics_from_cm(cm)
    s0 = SEV == 0
    m["FA0"] = cm[np.ix_(s0, SEV >= 1)].sum() / cm[s0, :].sum()
    return m


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    abl = pd.concat([pd.read_csv(DATA / "ablation" / f"preds_seed{s}.csv") for s in SEEDS])
    fam = pd.concat([pd.read_csv(DATA / "families" / f"preds_seed{s}.csv") for s in SEEDS])
    df = abl[["seed", "Experiment_Index", "Serial_Number", "Fault_Mode", "rf_F246", "lgbm_F246"]].merge(
        fam[["seed", "Experiment_Index", "xgb_F246", "gb_F246"]], on=["seed", "Experiment_Index"], how="inner")
    assert len(df) == len(abl) == len(fam), (len(df), len(abl), len(fam))
    sns = sorted(df["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    cms = {a: np.zeros((len(SEEDS), len(sns), 8, 8)) for a in ARMS}
    for si, seed in enumerate(SEEDS):
        d = df[df["seed"] == seed]
        ti, ni = d["Fault_Mode"].map(CIDX).values, d["Serial_Number"].map(sidx).values
        for a, col in ARMS.items():
            np.add.at(cms[a][si], (ni, ti, d[col].map(CIDX).values), 1)

    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    point, boot = {}, {}
    per_seed_cmr = {}
    for a in ARMS:
        point[a] = dict(zip(KEYS, np.mean([[metrics(cms[a][si].sum(axis=0))[k] for k in KEYS]
                                           for si in range(len(SEEDS))], axis=0)))
        per_seed_cmr[a] = [metrics(cms[a][si].sum(axis=0))["CMR"] for si in range(len(SEEDS))]
        arr = np.zeros((B, len(KEYS)))
        for b in range(B):
            acc = np.zeros(len(KEYS))
            for si in range(len(SEEDS)):
                m = metrics(np.tensordot(W[b], cms[a][si], axes=(0, 0)))
                acc += np.array([m[k] for k in KEYS])
            arr[b] = acc / len(SEEDS)
        boot[a] = pd.DataFrame(arr, columns=KEYS)

    def ci(x, lo=2.5, hi=97.5):
        return f"[{np.percentile(x, lo):.4f}; {np.percentile(x, hi):.4f}]"

    def verdict(metric, d):
        lo95, hi95 = np.percentile(d, [2.5, 97.5])
        lo90, hi90 = np.percentile(d, [5, 95])
        sup = ""
        if lo95 > 0 or hi95 < 0:
            a_lower = hi95 < 0
            sup = "A superior" if (a_lower if LOWER_BETTER[metric] else not a_lower) else "B superior"
        eq = metric in MARGIN and lo90 > -MARGIN[metric] and hi90 < MARGIN[metric]
        if sup and eq:
            return f"{sup} (distinta de zero, mas dentro da margem: equivalente)"
        return sup or ("equivalente" if eq else "inconclusivo")

    def pval(d):
        d = np.asarray(d)
        return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))

    out = ["# Famílias de classificadores — resumo (Ação 7)", "",
           f"F246, 10 sementes pareadas, 54 SN; {B} reamostragens de SN; hiperparâmetros padrão dos cadernos.", ""]
    rows = []
    for a in ARMS:
        p, b = point[a], boot[a]
        rows.append({"familia": a, "CMR": f"{p['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]),
                     "FA0": f"{p['FA0']:.3f}", "FA0_IC95": ci(b["FA0"]), "DR": f"{p['DR']:.3f}",
                     "R3": f"{p['R3']:.3f}", "macro_F1": f"{p['macro_F1']:.3f}"})
    out += ["## Métricas por família", "", md(pd.DataFrame(rows).set_index("familia")), ""]

    ranks = pd.DataFrame(per_seed_cmr)
    lowest = ranks.idxmin(axis=1).value_counts().to_dict()
    out += ["## Menor CMR por semente (descritivo)", "", f"Número de sementes (de 10) em que cada família tem a menor CMR agregada: {lowest}.", ""]

    primary = [("RF", "LightGBM"), ("RF", "XGBoost")]
    rows, pv = [], []
    for A, Bf in primary:
        for m in ("CMR", "FA0"):
            d = boot[A][m] - boot[Bf][m]
            pt = point[A][m] - point[Bf][m]
            rows.append({"comparacao": f"{A} (A) - {Bf} (B)", "metrica": m, "dif_pontual": f"{pt:+.4f}",
                         "IC95": ci(d), "IC90": ci(d, 5, 95), "p_bootstrap": f"{pval(d):.4f}",
                         "prob_A_melhor": f"{(d < 0).mean():.3f}", "veredito": verdict(m, d)})
            pv.append((len(rows) - 1, pval(d)))
    run = 0.0
    for rank, (ri, p) in enumerate(sorted(pv, key=lambda x: x[1])):
        run = max(run, min(1.0, (len(pv) - rank) * p))
        rows[ri]["p_Holm"] = f"{run:.4f}"
    out += ["## Família primária (4 testes, Holm)", "", md(pd.DataFrame(rows).set_index("comparacao")), ""]

    rows = []
    for A, Bf in itertools.combinations(ARMS, 2):
        for m in ("CMR", "DR", "macro_F1", "FA0"):
            d = boot[A][m] - boot[Bf][m]
            pt = point[A][m] - point[Bf][m]
            rows.append({"comparacao": f"{A} (A) - {Bf} (B)", "metrica": m, "dif_pontual": f"{pt:+.4f}",
                         "IC95": ci(d), "IC90": ci(d, 5, 95), "p_bootstrap": f"{pval(d):.4f}",
                         "veredito": verdict(m, d)})
    out += ["## Todas as comparações (secundárias, exploratórias, sem correção)", "",
            md(pd.DataFrame(rows).set_index("comparacao")), ""]
    (DATA / "families_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
