from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_cv_summary import metrics_from_cm, per_sn_stats  # noqa: E402
from pbit.modeling import SEVERITY_MAP, FAULT_ORDER  # noqa: E402

DATA = PROJECT_ROOT / "data"
NCV = DATA / "nested_cv"
B = 2000
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
MARGIN = {"CMR": 0.02, "DR": 0.02, "macro_F1": 0.03}
LOWER_BETTER = {"CMR": True, "DR": True, "macro_F1": False, "FA0": True}
FILES = {"F246": "cmr_rel_outer{k}", "F115": "cmr_rel_F115_outer{k}", "F18": "cmr_rel_F18_outer{k}"}


def fa0(cm):
    s0 = SEV == 0
    tot = cm[s0, :].sum()
    return cm[np.ix_(s0, SEV >= 1)].sum() / tot if tot else np.nan


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    rng = np.random.default_rng(42)
    cms, studies, sns = {}, {}, None
    for fs, pat in FILES.items():
        df = pd.concat([pd.read_csv(NCV / f"preds_{pat.format(k=k)}.csv") for k in range(5)], ignore_index=True)
        studies[fs] = [json.loads((NCV / f"study_{pat.format(k=k)}.json").read_text()) for k in range(5)]
        for arm in ("tuned", "untuned"):
            sns, cm = per_sn_stats(df, arm)
            cms[(fs, arm)] = cm
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    keys = ["CMR", "DR", "SE", "R3", "macro_F1", "FA0"]

    def met(cm):
        m = metrics_from_cm(cm)
        m["FA0"] = fa0(cm)
        return m

    point, boot = {}, {}
    for key, cm in cms.items():
        point[key] = met(cm.sum(axis=0))
        boot[key] = pd.DataFrame([met(np.tensordot(w, cm, axes=(0, 0))) for w in W])[keys]

    def ci(x, lo=2.5, hi=97.5):
        return f"[{np.percentile(x, lo):.4f}; {np.percentile(x, hi):.4f}]"

    out = ["# Ablação de atributos com ajuste aninhado (T2) — resumo", "",
           "Objetivo cmr_rel (restrição relativa), 100 tentativas por fold externo, 5 folds externos (semente 42), 54 SN.", ""]
    for fs in FILES:
        inner = [s["inner_best_value"] for s in studies[fs]]
        floors = [round(s["bal_floor"], 3) for s in studies[fs]]
        out.append(f"- {fs}: melhor valor interno por fold {', '.join(f'{v:.4f}' for v in inner)} (média {np.mean(inner):.4f}); limiares {floors}.")
    out.append("")
    rows = []
    for (fs, arm), p in point.items():
        b = boot[(fs, arm)]
        rows.append({"atributos": fs, "braco": arm, "CMR": f"{p['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]),
                     "contagem": f"{int(p['miss'])}/{int(p['n_sev3'])}", "DR": f"{p['DR']:.3f}",
                     "macro_F1": f"{p['macro_F1']:.3f}", "FA0": f"{p['FA0']:.3f}", "R3": f"{p['R3']:.3f}"})
    out += ["## Métricas externas agregadas", "", md(pd.DataFrame(rows).set_index("atributos")), ""]

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

    rows, pv = [], []
    for A, Bf in (("F246", "F18"), ("F246", "F115"), ("F115", "F18")):
        for m in ("CMR", "DR", "macro_F1", "FA0"):
            d = boot[(A, "tuned")][m] - boot[(Bf, "tuned")][m]
            pt = point[(A, "tuned")][m] - point[(Bf, "tuned")][m]
            rows.append({"comparacao": f"{A} (A) - {Bf} (B)", "metrica": m, "dif_pontual": f"{pt:+.4f}",
                         "IC95": ci(d), "IC90": ci(d, 5, 95), "p_bootstrap": f"{pval(d):.4f}",
                         "veredito": verdict(m, d), "familia": "primaria" if m == "CMR" else "secundaria"})
            if m == "CMR":
                pv.append((len(rows) - 1, pval(d)))
    order = sorted(pv, key=lambda x: x[1])
    run = 0.0
    for rank, (ri, p) in enumerate(order):
        run = max(run, min(1.0, (len(order) - rank) * p))
        rows[ri]["p_Holm"] = f"{run:.4f}"
    out += ["## Comparações pareadas entre conjuntos de atributos (braço ajustado)", "", md(pd.DataFrame(rows).set_index("comparacao")), ""]

    rows = []
    for fs in FILES:
        d = boot[(fs, "untuned")]["CMR"] - boot[(fs, "tuned")]["CMR"]
        rows.append({"atributos": fs, "reducao_CMR_untuned_menos_tuned": f"{point[(fs, 'untuned')]['CMR'] - point[(fs, 'tuned')]['CMR']:+.4f}",
                     "IC95": ci(d), "veredito": verdict("CMR", -d)})
    out += ["## Redução de CMR pelo ajuste, por conjunto de atributos (sem ajuste − ajustado)", "", md(pd.DataFrame(rows).set_index("atributos")), ""]
    rows = []
    for fs in FILES:
        pu, pt = point[(fs, "untuned")], point[(fs, "tuned")]
        q = (pt["FA0"] - pu["FA0"]) / (1 - pu["FA0"])
        chord = (1 - q) * pu["CMR"]
        bu, bt = boot[(fs, "untuned")], boot[(fs, "tuned")]
        qb = (bt["FA0"] - bu["FA0"]) / (1 - bu["FA0"])
        cb = (1 - qb) * bu["CMR"]
        d = bt["CMR"] - cb
        rows.append({"atributos": fs, "FA0_sem_ajuste": f"{pu['FA0']:.3f}", "FA0_ajustado": f"{pt['FA0']:.3f}",
                     "CMR_sem_ajuste": f"{pu['CMR']:.3f}", "CMR_ajustado": f"{pt['CMR']:.3f}",
                     "fracao_alarme_equivalente_q": f"{q:.3f}", "CMR_na_corda": f"{chord:.3f}",
                     "CMR_ajustado_menos_corda": f"{pt['CMR'] - chord:+.4f}", "IC95": ci(d)})
    out += ["## Ponto de operação: CMR contra falsos alarmes (FA0) e a política de mistura com 'alarme sempre'", "",
            "A corda liga o modelo sem ajuste (FA0_u, CMR_u) à política trivial 'prever F03 em tudo' (FA0 = 1, CMR = 0). "
            "Misturar as duas com probabilidade q de alarme gera FA0 = FA0_u + q(1 - FA0_u) e CMR = (1 - q) CMR_u. "
            "Um modelo ajustado só supera a mistura se o CMR ajustado ficar abaixo da corda no mesmo FA0. Análise exploratória, "
            "definida após ver os valores de FA0.", "",
            md(pd.DataFrame(rows).set_index("atributos")), ""]
    (DATA / "nested_cv_features_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
