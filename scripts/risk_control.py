from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nested_cv_tuning as nt  # noqa: E402
from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, SEVERITY_MAP, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = DATA / "risk_control"
GRID = np.concatenate([[np.inf], np.linspace(1.0, 0.0, 401)])
B = 2000
THREADS = 8


def sev(labels):
    return np.array([SEVERITY_MAP[c] for c in labels])


def fit_predict(X, y, Xt, params, f05=None, s3=None):
    sw = nt.weights(y, f05, s3)
    m = lgb.LGBMClassifier(**{**params, "n_jobs": THREADS}).fit(X, y, sample_weight=sw)
    proba = m.predict_proba(Xt)
    cls = list(m.classes_)
    s = proba[:, cls.index("F03")] + proba[:, cls.index("F06")]
    base = sev(np.array(cls)[proba.argmax(axis=1)])
    return s, base


def counts(df, s_col, base_col):
    """Per-SN counts over the threshold grid."""
    sns = sorted(df["Serial_Number"].unique())
    n3 = np.zeros(len(sns))
    n0 = np.zeros(len(sns))
    miss = np.zeros((len(sns), len(GRID)))
    fa = np.zeros((len(sns), len(GRID)))
    for i, sn in enumerate(sns):
        g = df[df["Serial_Number"] == sn]
        s, b, st = g[s_col].values, g[base_col].values, g["sev_true"].values
        alarm = s[:, None] >= GRID[None, :]
        final = np.where(alarm, 3, b[:, None])
        n3[i] = (st == 3).sum()
        n0[i] = (st == 0).sum()
        miss[i] = ((st[:, None] == 3) & (final == 0)).sum(axis=0)
        fa[i] = ((st[:, None] == 0) & (final >= 1)).sum(axis=0)
    return sns, n3, n0, miss, fa


def md(df):
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def part_a(out):
    train = pd.read_csv(DATA / "train_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in train.columns if c not in META_COLS]
    rows = []
    for k, (otr, ote) in enumerate(nt.folds_for(train, 5)):
        otrain, otest = train.iloc[otr].reset_index(drop=True), train.iloc[ote].reset_index(drop=True)
        aug = extract_features(augment_training_set(long_df, set(otrain["Experiment_Index"]), target_per_class=160,
                                                    noise_scale=0.01, random_state=42)).reset_index(drop=True)
        X = np.vstack([otrain[cols].values, aug[cols].values]).astype(float)
        y = np.concatenate([otrain["Fault_Mode"].values, aug["Fault_Mode"].values])
        Xt = otest[cols].values.astype(float)
        su, bu = fit_predict(X, y, Xt, nt.UNTUNED)
        st = json.loads((DATA / "nested_cv" / f"study_cmr_rel_outer{k}.json").read_text())["best_params"]
        st = dict(st)
        f05, s3 = st.pop("f05_boost"), st.pop("sev3_boost")
        stt, bt = fit_predict(X, y, Xt, {**st, "random_state": 42, "verbose": -1}, f05, s3)
        d = otest[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        d["s_u"], d["base_u"], d["s_t"], d["base_t"] = su, bu, stt, bt
        rows.append(d)
        print(f"A fold {k} done", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d["sev_true"] = sev(d["Fault_Mode"].values)
    OUT.mkdir(exist_ok=True)
    d.to_csv(OUT / "partA_scores.csv", index=False)

    sns, n3, n0, miss_u, fa_u = counts(d, "s_u", "base_u")
    _, _, _, miss_t, fa_t = counts(d, "s_t", "base_t")
    cmr_u, fa0_u = miss_u.sum(0) / n3.sum(), fa_u.sum(0) / n0.sum()
    cmr_t, fa0_t = miss_t.sum(0) / n3.sum(), fa_t.sum(0) / n0.sum()

    rng = np.random.default_rng(42)
    W = rng.multinomial(len(sns), np.ones(len(sns)) / len(sns), size=B)
    CU, FU = (W @ miss_u) / (W @ n3)[:, None], (W @ fa_u) / (W @ n0)[:, None]
    CT, FT = (W @ miss_t) / (W @ n3)[:, None], (W @ fa_t) / (W @ n0)[:, None]
    # tuned model's own operating point (no threshold = grid index 0)
    diffs = []
    for b in range(B):
        ok = FU[b] <= FT[b, 0] + 1e-12
        diffs.append(CT[b, 0] - CU[b][ok].min() if ok.any() else np.nan)
    diffs = np.array(diffs)
    point_ok = fa0_u <= fa0_t[0] + 1e-12
    best_point = cmr_u[point_ok].min() if point_ok.any() else np.nan

    out += ["## Parte A — curva de operação (partição de treino, folds externos da Ação 5)", "",
            f"Ciclos de Sev3: {int(n3.sum())} em {int((n3 > 0).sum())} SN; ciclos de Sev0: {int(n0.sum())}. Escore s = P(F03) + P(F06).", "",
            md(pd.DataFrame([
                {"modelo": "sem ajuste (sem limiar)", "CMR": f"{cmr_u[0]:.3f}", "FA0": f"{fa0_u[0]:.3f}"},
                {"modelo": "CMR-tuned (sem limiar)", "CMR": f"{cmr_t[0]:.3f}", "FA0": f"{fa0_t[0]:.3f}"},
            ]).set_index("modelo")), ""]
    out += [f"**Comparação principal.** Menor CMR da curva do modelo sem ajuste com FA0 ≤ {fa0_t[0]:.3f}: {best_point:.3f}. "
            f"CMR do CMR-tuned menos esse valor: {cmr_t[0] - best_point:+.4f} (IC95 por SN "
            f"[{np.nanpercentile(diffs, 2.5):.4f}; {np.nanpercentile(diffs, 97.5):.4f}]).", ""]

    rows = []
    for tgt in (0.10, 0.05, 0.02):
        for name, c, f in (("sem ajuste + limiar", cmr_u, fa0_u), ("CMR-tuned + limiar", cmr_t, fa0_t)):
            ok = c <= tgt
            rows.append({"CMR_alvo": tgt, "curva": name, "FA0_minima": f"{f[ok].min():.3f}" if ok.any() else "inatingível"})
    out += ["FA0 mínima necessária para atingir uma CMR-alvo (pontos da grade, estimativa pontual):", "",
            md(pd.DataFrame(rows).set_index("CMR_alvo")), ""]
    return d


def part_b(out):
    audit = pd.read_csv(DATA / "audit_dataset_r1.csv")
    keep = set(audit.loc[audit["in_D_primary"], "Experiment_Index"])
    feats = pd.read_csv(DATA / "PBIT_features_r1.csv")
    feats = feats[feats["Experiment_Index"].isin(keep)].reset_index(drop=True)
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in feats.columns if c not in META_COLS]
    res = {}
    for seed in (42, 0, 7):
        rows = []
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
            np.arange(len(feats)), feats["Fault_Mode"].values, feats["Serial_Number"].values))
        for k, (tr_i, te_i) in enumerate(folds):
            tr, te = feats.iloc[tr_i], feats.iloc[te_i]
            aug = extract_features(augment_training_set(long_df, set(tr["Experiment_Index"]), target_per_class=160,
                                                        noise_scale=0.01, random_state=seed + k)).reset_index(drop=True)
            X = np.vstack([tr[cols].values, aug[cols].values]).astype(float)
            y = np.concatenate([tr["Fault_Mode"].values, aug["Fault_Mode"].values])
            s, b = fit_predict(X, y, te[cols].values.astype(float), nt.UNTUNED)
            d = te[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
            d["s_u"], d["base_u"] = s, b
            rows.append(d)
        d = pd.concat(rows, ignore_index=True)
        d["sev_true"] = sev(d["Fault_Mode"].values)
        d.to_csv(OUT / f"partB_scores_seed{seed}.csv", index=False)
        res[seed] = d
        print(f"B seed {seed} done", flush=True)

    alphas = [0.0625, 0.10, 0.15, 0.20, 0.30]
    rows = []
    for seed, d in res.items():
        sns, n3, n0, miss, fa = counts(d, "s_u", "base_u")
        units = [i for i in range(len(sns)) if n3[i] > 0]
        L = miss[units] / n3[units][:, None]
        FA = fa.sum(0) / n0.sum()
        for alpha in alphas:
            realised, fa_cost, naive_real, naive_fa, viol = [], [], [], [], 0
            for j in range(len(units)):
                cal = np.delete(np.arange(len(units)), j)
                n = len(cal)
                R = L[cal].mean(axis=0)
                crc = (n / (n + 1)) * R + 1.0 / (n + 1)
                okc = np.where(crc <= alpha)[0]
                gi = okc[0] if len(okc) else len(GRID) - 1
                realised.append(L[j, gi])
                fa_cost.append(FA[gi])
                viol += int(L[j, gi] > alpha)
                okn = np.where(R <= alpha)[0]
                gn = okn[0] if len(okn) else len(GRID) - 1
                naive_real.append(L[j, gn])
                naive_fa.append(FA[gn])
            rows.append({"semente": seed, "alpha": alpha, "perda_realizada_media": np.mean(realised),
                         "unidades_com_perda>alpha": viol, "FA0_media": np.mean(fa_cost),
                         "sem_correcao_perda": np.mean(naive_real), "sem_correcao_FA0": np.mean(naive_fa),
                         "sem_controle_perda": L[:, 0].mean(), "sem_controle_FA0": FA[0]})
    r = pd.DataFrame(rows)
    agg = r.groupby("alpha").mean(numeric_only=True).drop(columns="semente").round(3).reset_index().set_index("alpha")
    out += ["## Parte B — controle de risco no nível da unidade (conjunto completo, 16 unidades de Sev3, 3 sementes)", "",
            "Perda por unidade = fração dos ciclos de Sev3 da unidade com severidade final 0. Calibração com as 15 outras unidades "
            "(leave-one-unit-out); média dos 16 resultados e das 3 sementes. 'sem correção' = limiar escolhido sem o termo 1/(n+1); "
            "'sem controle' = modelo sem ajuste, sem limiar.", "", md(agg), ""]
    need = pd.DataFrame([{"alpha": a, "unidades_de_Sev3_necessarias": int(np.ceil(1 / a - 1))} for a in (0.10, 0.05, 0.02, 0.01)])
    out += ["Unidades de Sev3 necessárias para que a garantia seja viável (n ≥ 1/α − 1):", "", md(need.set_index("alpha")), ""]
    return r


def main():
    OUT.mkdir(exist_ok=True)
    out = ["# Ação 13 — curva CMR–FA0 e controle de risco (resumo)", ""]
    part_a(out)
    part_b(out)
    (DATA / "risk_control_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
