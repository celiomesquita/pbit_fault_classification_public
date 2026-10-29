from __future__ import annotations

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
CIDX = {c: i for i, c in enumerate(FAULT_ORDER)}
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
MARGIN = {"CMR": 0.02, "DR": 0.02, "macro_F1": 0.03}
LOWER_BETTER = {"CMR": True, "DR": True, "macro_F1": False, "FA0": True}
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]


def fa0(cm):
    s0 = SEV == 0
    tot = cm[s0, :].sum()
    return cm[np.ix_(s0, SEV >= 1)].sum() / tot if tot else np.nan


def metrics(cm):
    m = metrics_from_cm(cm)
    m["FA0"] = fa0(cm)
    return m


def md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    frames = [pd.read_csv(DATA / "ablation" / f"preds_seed{s}.csv") for s in SEEDS]
    df = pd.concat(frames, ignore_index=True)
    arms = [c for c in df.columns if c.startswith(("lgbm_", "rf_", "B"))]
    sns = sorted(df["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    n_sn = len(sns)
    cms = {a: np.zeros((len(SEEDS), n_sn, 8, 8)) for a in arms}
    for si, seed in enumerate(SEEDS):
        d = df[df["seed"] == seed]
        ti = d["Fault_Mode"].map(CIDX).values
        ni = d["Serial_Number"].map(sidx).values
        for a in arms:
            np.add.at(cms[a][si], (ni, ti, d[a].map(CIDX).values), 1)

    rng = np.random.default_rng(42)
    W = rng.multinomial(n_sn, np.ones(n_sn) / n_sn, size=B)
    keys = ["CMR", "DR", "SE", "R3", "macro_F1", "FA0"]

    def arm_boot(a):
        out = np.zeros((B, len(keys)))
        for b in range(B):
            acc = np.zeros(len(keys))
            for si in range(len(SEEDS)):
                m = metrics(np.tensordot(W[b], cms[a][si], axes=(0, 0)))
                acc += np.array([m[k] for k in keys])
            out[b] = acc / len(SEEDS)
        return out

    point, boot = {}, {}
    for a in arms:
        pts = np.mean([[metrics(cms[a][si].sum(axis=0))[k] for k in keys] for si in range(len(SEEDS))], axis=0)
        point[a] = dict(zip(keys, pts))
        boot[a] = pd.DataFrame(arm_boot(a), columns=keys)

    def ci(x, lo=2.5, hi=97.5):
        return f"[{np.percentile(x, lo):.3f}; {np.percentile(x, hi):.3f}]"

    out = ["# Ablação de atributos e linhas de base — resumo", "",
           f"10 sementes pareadas (partições dos SN), 5 folds, 54 SN; {B} reamostragens de SN; estimativa = média das 10 sementes.", ""]
    rows = []
    for a in arms:
        p, b = point[a], boot[a]
        rows.append({"braco": a, "CMR": f"{p['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]),
                     "DR": f"{p['DR']:.3f}", "DR_IC95": ci(b["DR"]),
                     "macro_F1": f"{p['macro_F1']:.3f}", "macro_F1_IC95": ci(b["macro_F1"]),
                     "FA0": f"{p['FA0']:.3f}", "R3": f"{p['R3']:.3f}"})
    out += ["## Métricas por braço", "", md(pd.DataFrame(rows).set_index("braco")), ""]

    def verdict(metric, d):
        lo95, hi95 = np.percentile(d, [2.5, 97.5])
        lo90, hi90 = np.percentile(d, [5, 95])
        sup = ""
        if lo95 > 0 or hi95 < 0:
            a_lower = hi95 < 0
            a_better = a_lower if LOWER_BETTER[metric] else not a_lower
            sup = "A superior" if a_better else "B superior"
        eq = False
        if metric in MARGIN:
            eq = (lo90 > -MARGIN[metric]) and (hi90 < MARGIN[metric])
        if sup and eq:
            return f"{sup} (distinta de zero, mas dentro da margem: equivalente)"
        if sup:
            return sup
        if eq:
            return "equivalente"
        return "inconclusivo"

    def pval(d):
        d = np.asarray(d)
        return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))

    pairs = [("F246", "F18"), ("F246", "F115"), ("F115", "F18")]
    for model, label in (("lgbm", "LightGBM (família primária: CMR)"), ("rf", "Random Forest (secundária)")):
        rows = []
        pv = []
        for A, Bf in pairs:
            for m in ("CMR", "DR", "macro_F1", "FA0"):
                d = boot[f"{model}_{A}"][m] - boot[f"{model}_{Bf}"][m]
                pt = point[f"{model}_{A}"][m] - point[f"{model}_{Bf}"][m]
                rows.append({"comparacao": f"{A} (A) - {Bf} (B)", "metrica": m,
                             "dif_pontual": f"{pt:+.4f}", "IC95": ci(d), "IC90": ci(d, 5, 95),
                             "p_bootstrap": f"{pval(d):.4f}", "veredito": verdict(m, d),
                             "familia": "primaria" if (model == "lgbm" and m == "CMR") else "secundaria"})
                if model == "lgbm" and m == "CMR":
                    pv.append((len(rows) - 1, pval(d)))
        if pv:
            order = sorted(pv, key=lambda x: x[1])
            m_ = len(order)
            run = 0.0
            for rank, (ri, p) in enumerate(order):
                run = max(run, min(1.0, (m_ - rank) * p))
                rows[ri]["p_Holm"] = f"{run:.4f}"
        out += [f"## Comparações pareadas — {label}", "", md(pd.DataFrame(rows).set_index("comparacao")), ""]

    rows = []
    for model in ("lgbm", "rf"):
        for fs in ("F18", "F115", "F246"):
            a = f"{model}_{fs}"
            for ref in ("B1_always_N01", "B3_threshold_std", "B4_threshold_safety", "B5_always_F03"):
                d = boot[a]["CMR"] - boot[ref]["CMR"]
                rows.append({"braco": a, "referencia": ref, "dif_CMR": f"{point[a]['CMR'] - point[ref]['CMR']:+.3f}",
                             "IC95": ci(d), "dif_FA0": f"{point[a]['FA0'] - point[ref]['FA0']:+.3f}",
                             "dif_macroF1": f"{point[a]['macro_F1'] - point[ref]['macro_F1']:+.3f}"})
    out += ["## Modelos contra linhas de base triviais (diferença = modelo − base)", "", md(pd.DataFrame(rows).set_index("braco")), ""]

    (DATA / "ablation_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
