from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import risk_control as rc  # noqa: E402
from pbit.modeling import SEVERITY_MAP  # noqa: E402

DATA = PROJECT_ROOT / "data"
B = 2000
KAPPAS = [2, 3, 5, 10, 20]


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    pa = pd.read_csv(DATA / "risk_control" / "partA_scores.csv")
    cs = pd.read_csv(DATA / "cost_sensitive" / "preds.csv")
    d = pa.merge(cs.drop(columns=["Serial_Number", "Fault_Mode"]), on="Experiment_Index", how="inner")
    assert len(d) == len(pa) == len(cs), (len(d), len(pa), len(cs))
    sns, n3, n0, miss_u, fa_u = rc.counts(d, "s_u", "base_u")
    sidx = {s: i for i, s in enumerate(sns)}
    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    CU = (W @ miss_u) / (W @ n3)[:, None]
    FU = (W @ fa_u) / (W @ n0)[:, None]
    cmr_u, fa0_u = miss_u.sum(0) / n3.sum(), fa_u.sum(0) / n0.sum()

    sev_true = d["Fault_Mode"].map(SEVERITY_MAP).values
    arms = {f"{kind}_k{k}": (kind, k) for kind in ("mec", "mc") for k in KAPPAS}
    rows, pv = [], []
    for name, (kind, k) in arms.items():
        sp = d[name].map(SEVERITY_MAP).values
        miss = np.zeros(len(sns))
        fa = np.zeros(len(sns))
        for sn, st, sv in zip(d["Serial_Number"], sev_true, sp):
            i = sidx[sn]
            if st == 3 and sv == 0:
                miss[i] += 1
            if st == 0 and sv >= 1:
                fa[i] += 1
        cmr_a, fa0_a = miss.sum() / n3.sum(), fa.sum() / n0.sum()
        Ca, Fa = (W @ miss) / (W @ n3), (W @ fa) / (W @ n0)
        diffs = np.array([Ca[b] - CU[b][FU[b] <= Fa[b] + 1e-12].min() if (FU[b] <= Fa[b] + 1e-12).any() else np.nan
                          for b in range(B)])
        ok = fa0_u <= fa0_a + 1e-12
        best = cmr_u[ok].min() if ok.any() else np.nan
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        verdict = "supera o limiar" if hi < 0 else ("inferior ao limiar" if lo > 0 else "não distinguível")
        x = diffs[~np.isnan(diffs)]
        p = min(1.0, 2 * min((np.sum(x <= 0) + 1) / (len(x) + 1), (np.sum(x >= 0) + 1) / (len(x) + 1)))
        rows.append({"metodo": "MetaCost" if kind == "mc" else "decisão de custo mínimo", "kappa": k,
                     "CMR": f"{cmr_a:.3f}", "FA0": f"{fa0_a:.3f}", "melhor_CMR_limiar_FA0<=": f"{best:.3f}",
                     "dif_pontual": f"{cmr_a - best:+.4f}", "IC95": f"[{lo:.4f}; {hi:.4f}]", "p": f"{p:.4f}",
                     "veredito": verdict})
        if k == 10:
            pv.append((len(rows) - 1, p))
    run = 0.0
    for rank, (ri, p_) in enumerate(sorted(pv, key=lambda z: z[1])):
        run = max(run, min(1.0, (len(pv) - rank) * p_))
        rows[ri]["p_Holm"] = f"{run:.4f}"
    reling = {k: float(cs[f"mc_relabelled_k{k}"].mean()) for k in KAPPAS}
    out = ["# Comparador sensível a custo (Ação 15)", "",
           "Folds externos da Ação 13 (semente 42), partição de treino, F246, ampliação 160. Comparador: curva de limiar do modelo sem ajuste.", "",
           f"Ponto do modelo sem ajuste: CMR {cmr_u[0]:.3f}, FA0 {fa0_u[0]:.3f}.", "",
           md(pd.DataFrame(rows).set_index("metodo")), "",
           "Fração média de exemplos de treino reclassificados pelo MetaCost: " +
           ", ".join(f"κ = {k}: {v:.3f}" for k, v in reling.items()) + ".", ""]
    (DATA / "cost_sensitive_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
