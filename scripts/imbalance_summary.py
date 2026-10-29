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
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
CIDX = {c: i for i, c in enumerate(FAULT_ORDER)}
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
WEIGHTS = ["none", "freq", "sev", "freq_sev"]
AUGS = ["none", "gaussian", "smote"]
KEYS = ["CMR", "FA0", "DR", "R3", "macro_F1", "rec_F04", "rec_F05", "rec_F06"]
MARGIN = {"macro_F1": 0.03, "CMR": 0.02, "DR": 0.02}
LOWER_BETTER = {"CMR": True, "FA0": True, "DR": True, "R3": False, "macro_F1": False}


def metrics(cm):
    m = metrics_from_cm(cm)
    s0 = SEV == 0
    out = {k: m[k] for k in ("CMR", "DR", "R3", "macro_F1")}
    out["FA0"] = cm[np.ix_(s0, SEV >= 1)].sum() / cm[s0, :].sum()
    for c in ("F04", "F05", "F06"):
        i = CIDX[c]
        out[f"rec_{c}"] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() else np.nan
    return out


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    d = pd.concat([pd.read_csv(DATA / "imbalance" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    sns = sorted(d["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    arms = [f"{m}_{w}_{a}" for m in ("lgbm", "rf") for w in WEIGHTS for a in AUGS]
    cms = {}
    for arm in arms:
        cm = np.zeros((len(SEEDS), len(sns), 8, 8))
        for si, seed in enumerate(SEEDS):
            g = d[d["seed"] == seed]
            np.add.at(cm[si], (g["Serial_Number"].map(sidx).values, g["Fault_Mode"].map(CIDX).values,
                               g[arm].map(CIDX).values), 1)
        cms[arm] = cm
    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    point, boot = {}, {}
    for arm, cm in cms.items():
        point[arm] = dict(zip(KEYS, np.nanmean([[metrics(cm[si].sum(axis=0))[k] for k in KEYS] for si in range(len(SEEDS))], axis=0)))
        arr = np.zeros((B, len(KEYS)))
        for b in range(B):
            acc = np.zeros(len(KEYS))
            for si in range(len(SEEDS)):
                m = metrics(np.tensordot(W[b], cm[si], axes=(0, 0)))
                acc += np.nan_to_num(np.array([m[k] for k in KEYS]))
            arr[b] = acc / len(SEEDS)
        boot[arm] = pd.DataFrame(arr, columns=KEYS)

    def ci(x, lo=2.5, hi=97.5):
        return f"[{np.percentile(x, lo):.3f}; {np.percentile(x, hi):.3f}]"

    def verdict(metric, x):
        lo95, hi95 = np.percentile(x, [2.5, 97.5])
        lo90, hi90 = np.percentile(x, [5, 95])
        sup = ""
        if lo95 > 0 or hi95 < 0:
            a_lower = hi95 < 0
            sup = "A superior" if (a_lower if LOWER_BETTER[metric] else not a_lower) else "B superior"
        eq = metric in MARGIN and lo90 > -MARGIN[metric] and hi90 < MARGIN[metric]
        if sup and eq:
            return f"{sup} (dentro da margem: equivalente)"
        return sup or ("equivalente" if eq else "inconclusivo")

    def pval(x):
        x = np.asarray(x)
        return min(1.0, 2 * min((np.sum(x <= 0) + 1) / (len(x) + 1), (np.sum(x >= 0) + 1) / (len(x) + 1)))

    out = ["# Ablação do tratamento de desbalanceamento — resumo (Ação 14)", "",
           f"F246, partição de treino (54 SN), 10 sementes pareadas, {B} reamostragens de SN. Todos os braços imputam ausentes pela mediana.", ""]
    for mdl in ("lgbm", "rf"):
        rows = []
        for w in WEIGHTS:
            for a in AUGS:
                arm = f"{mdl}_{w}_{a}"
                p, b = point[arm], boot[arm]
                rows.append({"pesos": w, "ampliacao": a, "CMR": f"{p['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]),
                             "FA0": f"{p['FA0']:.3f}", "DR": f"{p['DR']:.3f}", "R3": f"{p['R3']:.3f}",
                             "macro_F1": f"{p['macro_F1']:.3f}", "macro_F1_IC95": ci(b["macro_F1"]),
                             "rec_F04": f"{p['rec_F04']:.3f}", "rec_F05": f"{p['rec_F05']:.3f}", "rec_F06": f"{p['rec_F06']:.3f}"})
        out += [f"## Métricas por combinação — {mdl}", "", md(pd.DataFrame(rows).set_index("pesos")), ""]

    prim = [("P1: freq_sev - freq (gaussian), CMR", "lgbm_freq_sev_gaussian", "lgbm_freq_gaussian", "CMR"),
            ("P2: gaussian - none (freq_sev), macro-F1", "lgbm_freq_sev_gaussian", "lgbm_freq_sev_none", "macro_F1"),
            ("P3: gaussian - smote (freq_sev), macro-F1", "lgbm_freq_sev_gaussian", "lgbm_freq_sev_smote", "macro_F1")]
    rows, pv = [], []
    for lab, A, Bf, m in prim:
        x = boot[A][m] - boot[Bf][m]
        rows.append({"teste": lab, "dif_pontual": f"{point[A][m] - point[Bf][m]:+.4f}", "IC95": ci(x),
                     "IC90": ci(x, 5, 95), "p_bootstrap": f"{pval(x):.4f}", "veredito": verdict(m, x)})
        pv.append((len(rows) - 1, pval(x)))
    run = 0.0
    for rank, (ri, p_) in enumerate(sorted(pv, key=lambda z: z[1])):
        run = max(run, min(1.0, (len(pv) - rank) * p_))
        rows[ri]["p_Holm"] = f"{run:.4f}"
    out += ["## Família primária (LightGBM, 3 testes, Holm)", "", md(pd.DataFrame(rows).set_index("teste")), ""]

    rows = []
    for mdl in ("lgbm", "rf"):
        for w in WEIGHTS:
            if w == "freq_sev":
                continue
            A, Bf = f"{mdl}_freq_sev_gaussian", f"{mdl}_{w}_gaussian"
            for m in ("CMR", "FA0", "macro_F1"):
                x = boot[A][m] - boot[Bf][m]
                rows.append({"modelo": mdl, "contraste": f"freq_sev - {w} (gaussian)", "metrica": m,
                             "dif": f"{point[A][m] - point[Bf][m]:+.4f}", "IC95": ci(x), "veredito": verdict(m, x)})
        for a in ("none", "smote"):
            A, Bf = f"{mdl}_freq_sev_gaussian", f"{mdl}_freq_sev_{a}"
            for m in ("CMR", "FA0", "macro_F1", "rec_F06"):
                x = boot[A][m] - boot[Bf][m]
                rows.append({"modelo": mdl, "contraste": f"gaussian - {a} (freq_sev)", "metrica": m,
                             "dif": f"{point[A][m] - point[Bf][m]:+.4f}", "IC95": ci(x),
                             "veredito": verdict(m, x) if m in LOWER_BETTER else "descritivo"})
    out += ["## Contrastes exploratórios", "", md(pd.DataFrame(rows).set_index("modelo")), ""]
    (DATA / "imbalance_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
