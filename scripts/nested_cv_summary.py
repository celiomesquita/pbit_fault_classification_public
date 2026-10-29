from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from pbit.modeling import FAULT_ORDER, SEVERITY_MAP  # noqa: E402

DATA = PROJECT_ROOT / "data"
NCV = DATA / "nested_cv"
B = 2000
RNG = np.random.default_rng(42)
CLASS_IDX = {c: i for i, c in enumerate(FAULT_ORDER)}
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])


def load(obj: str) -> pd.DataFrame:
    frames = [pd.read_csv(NCV / f"preds_{obj}_outer{k}.csv").assign(outer=k) for k in range(5)]
    return pd.concat(frames, ignore_index=True)


def per_sn_stats(df: pd.DataFrame, arm: str):
    sns = sorted(df["Serial_Number"].unique())
    idx = {s: i for i, s in enumerate(sns)}
    n = len(sns)
    cm = np.zeros((n, 8, 8))
    for s, t, p in zip(df["Serial_Number"], df["Fault_Mode"], df[f"pred_{arm}"]):
        cm[idx[s], CLASS_IDX[t], CLASS_IDX[p]] += 1
    return sns, cm


def metrics_from_cm(cm: np.ndarray) -> dict:
    n = cm.sum()
    st = SEV[:, None] * np.ones((1, 8))
    sp = np.ones((8, 1)) * SEV[None, :]
    s3 = (SEV == 3)
    sev3_total = cm[s3, :].sum()
    miss = cm[np.ix_(s3, SEV == 0)].sum()
    r3 = cm[np.ix_(s3, SEV == 3)].sum()
    down = (cm * (sp < st)).sum()
    err = (cm * (sp != st)).sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    present = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    recall = np.where(cm.sum(axis=1) > 0, tp / np.maximum(cm.sum(axis=1), 1), np.nan)
    return {
        "CMR": miss / sev3_total if sev3_total else np.nan,
        "DR": down / n, "SE": err / n, "R3": r3 / sev3_total if sev3_total else np.nan,
        "macro_F1": f1[present].mean(),
        "bal_acc": np.nanmean(recall),
        "n_sev3": sev3_total, "miss": miss,
        "FA0": cm[np.ix_(SEV == 0, SEV >= 1)].sum() / cm[SEV == 0, :].sum() if cm[SEV == 0, :].sum() else np.nan,
    }


def ci(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return f"[{np.percentile(x, 2.5):.4f}; {np.percentile(x, 97.5):.4f}]"


def md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    out = ["# Validação aninhada — resumo (5 folds externos × 5 internos, SN agrupados)", ""]
    arms_all = {}
    sn_list = None
    for obj in ("cmr", "cmr_rel", "f1"):
        df = load(obj)
        studies = [json.loads((NCV / f"study_{obj}_outer{k}.json").read_text()) for k in range(5)]
        inner = [s["inner_best_value"] for s in studies]
        label = {"cmr": "CMR-tuned, restrição original (acurácia balanceada ≥ 0,50; inviável em 3 de 5 folds)",
                 "cmr_rel": "CMR-tuned, restrição relativa (emenda 5.1)", "f1": "F1-tuned"}[obj]
        out += [f"## Objetivo: {label} ({studies[0]['n_trials']} tentativas por fold externo)", "",
                f"Melhor valor interno por fold externo: {', '.join(f'{v:.4f}' for v in inner)}; média {np.mean(inner):.4f}"
                " (estimativa otimista, vista pela seleção).", ""]
        for arm in ("tuned", "untuned", "paper_params"):
            sns, cm = per_sn_stats(df, arm)
            arms_all[(obj, arm)] = cm
            sn_list = sns
    n_sn = len(sn_list)
    W = RNG.multinomial(n_sn, np.ones(n_sn) / n_sn, size=B)

    def boot(cm):
        return pd.DataFrame([metrics_from_cm(np.tensordot(w, cm, axes=(0, 0))) for w in W])

    point, bootd = {}, {}
    for key, cm in arms_all.items():
        point[key] = metrics_from_cm(cm.sum(axis=0))
        bootd[key] = boot(cm)

    rows = []
    for (obj, arm), pt in point.items():
        b = bootd[(obj, arm)]
        rows.append({
            "objetivo": obj, "braco": arm,
            "CMR": f"{pt['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]),
            "CMR_contagem": f"{int(pt['miss'])}/{int(pt['n_sev3'])}",
            "DR": f"{pt['DR']:.3f}", "DR_IC95": ci(b["DR"]),
            "R3": f"{pt['R3']:.3f}", "FA0": f"{pt['FA0']:.3f}", "macro_F1": f"{pt['macro_F1']:.3f}", "macro_F1_IC95": ci(b["macro_F1"]),
        })
    out += ["## Métricas externas agregadas (ciclos reais; 54 SN)", "", md(pd.DataFrame(rows).set_index("objetivo")), ""]

    comps = [
        ("cmr_rel", "untuned", "cmr_rel", "tuned", "reducao CMR (untuned - tuned CMR-tuned relativo)", "CMR"),
        ("f1", "untuned", "f1", "tuned", "reducao CMR (untuned - tuned F1-tuned)", "CMR"),
        ("f1", "tuned", "cmr_rel", "tuned", "F1-tuned - CMR-tuned relativo (CMR)", "CMR"),
        ("cmr_rel", "paper_params", "cmr_rel", "tuned", "otimismo: paper_params - tuned aninhado (CMR)", "CMR"),
        ("f1", "tuned", "cmr_rel", "tuned", "F1-tuned - CMR-tuned relativo (DR)", "DR"),
        ("f1", "tuned", "cmr_rel", "tuned", "F1-tuned - CMR-tuned relativo (macro-F1)", "macro_F1"),
        ("cmr", "untuned", "cmr", "tuned", "restricao original: untuned - tuned (CMR)", "CMR"),
    ]
    rows = []
    for o1, a1, o2, a2, label, m in comps:
        d = bootd[(o1, a1)][m] - bootd[(o2, a2)][m]
        pt = point[(o1, a1)][m] - point[(o2, a2)][m]
        sign = (d.dropna() > 0).mean()
        rows.append({"comparacao": label, "diferenca_pontual": f"{pt:+.3f}", "IC95": ci(d),
                     "proporcao_reamostragens_positivas": f"{sign:.3f}"})
    out += ["## Diferenças pareadas (bootstrap por SN, 2.000 reamostragens)", "", md(pd.DataFrame(rows).set_index("comparacao")), ""]
    valid = int(bootd[("cmr_rel", "tuned")]["CMR"].notna().sum())
    out += [f"Reamostragens com ciclos de Sev3: {valid} de {B}.", ""]

    (DATA / "nested_cv_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
