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
ARMS = ["grouped_faultmode", "grouped_severity", "random_stratified"]
MODELS = ["rf", "lgbm"]
KEYS = ["accuracy", "macro_F1", "CMR", "DR", "SE", "FA0", "R3"]


def metrics(cm):
    m = metrics_from_cm(cm)
    s0 = SEV == 0
    m["FA0"] = cm[np.ix_(s0, SEV >= 1)].sum() / cm[s0, :].sum()
    m["accuracy"] = np.trace(cm) / cm.sum()
    return m


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    d = pd.concat([pd.read_csv(DATA / "leakage" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    sns = sorted(d["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    cms = {}
    for arm in ARMS:
        for mdl in MODELS:
            cm = np.zeros((len(SEEDS), len(sns), 8, 8))
            for si, seed in enumerate(SEEDS):
                g = d[(d["arm"] == arm) & (d["seed"] == seed)]
                np.add.at(cm[si], (g["Serial_Number"].map(sidx).values, g["Fault_Mode"].map(CIDX).values,
                                   g[mdl].map(CIDX).values), 1)
            cms[(arm, mdl)] = cm
    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    point, boot = {}, {}
    for key, cm in cms.items():
        point[key] = dict(zip(KEYS, np.mean([[metrics(cm[si].sum(axis=0))[k] for k in KEYS]
                                             for si in range(len(SEEDS))], axis=0)))
        arr = np.zeros((B, len(KEYS)))
        for b in range(B):
            acc = np.zeros(len(KEYS))
            for si in range(len(SEEDS)):
                m = metrics(np.tensordot(W[b], cm[si], axes=(0, 0)))
                acc += np.array([m[k] for k in KEYS])
            arr[b] = acc / len(SEEDS)
        boot[key] = pd.DataFrame(arr, columns=KEYS)

    def ci(x):
        return f"[{np.percentile(x, 2.5):.3f}; {np.percentile(x, 97.5):.3f}]"

    out = ["# Vazamento por SN e estratificação — resumo", "",
           f"F246, ampliação 160, partição de treino (1.196 ciclos, 54 SN), {len(SEEDS)} sementes pareadas, {B} reamostragens de SN.", ""]
    rows = []
    for (arm, mdl), p in point.items():
        b = boot[(arm, mdl)]
        rows.append({"braco": arm, "modelo": mdl, **{k: f"{p[k]:.3f}" for k in KEYS},
                     "accuracy_IC95": ci(b["accuracy"]), "macro_F1_IC95": ci(b["macro_F1"]), "CMR_IC95": ci(b["CMR"])})
    out += ["## Métricas por braço (média das sementes)", "", md(pd.DataFrame(rows).set_index("braco")), ""]
    rows = []
    for mdl in MODELS:
        for a, b_ in (("random_stratified", "grouped_faultmode"), ("grouped_severity", "grouped_faultmode")):
            for k in KEYS:
                d_ = boot[(a, mdl)][k] - boot[(b_, mdl)][k]
                rows.append({"modelo": mdl, "comparacao": f"{a} - {b_}", "metrica": k,
                             "dif": f"{point[(a, mdl)][k] - point[(b_, mdl)][k]:+.3f}", "IC95": ci(d_)})
    out += ["## Diferenças pareadas (bootstrap por SN)", "", md(pd.DataFrame(rows).set_index("modelo")), ""]
    (DATA / "leakage_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
