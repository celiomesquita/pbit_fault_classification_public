"""Testa se o ganho da variante C de S2 equivale a um detector simples de mudanca de nivel e se depende do rotulo N01.
Escore: |media / referencia - 1|. Sem treinamento. Saida: resultados/s2_regra_nivel.csv."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import experimentos_base as b
from experimento_s2_ablacao import K_REF

d = pd.read_csv(b.RAIZ / "dados" / "pbit_features_corrigido_v3.csv")
FALHAS = b.CLASSES[2:]
LIMIARES = (0.005, 0.01, 0.02, 0.03, 0.05)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def avaliar(ordenado_por_rotulo):
    base = d[d["condicao"] == "N01"] if ordenado_por_rotulo else d
    elegiveis = d[d["condicao"] == "N01"]["numero_serie"].unique()
    primeiros = base[base["numero_serie"].isin(elegiveis)].sort_values("experimento_idx").groupby("numero_serie").head(K_REF)
    ref = primeiros.groupby("numero_serie")["media"].median()
    s = d[d["numero_serie"].isin(ref.index) & ~d["experimento_idx"].isin(primeiros["experimento_idx"])].copy()
    s["escore"] = (s["media"] / s["numero_serie"].map(ref) - 1).abs()
    return s[s["condicao"] != "N02"]


def auc_ic(s, rng, n=2000):
    y = s["condicao"].isin(FALHAS).values
    unidades = sorted(s["numero_serie"].unique())
    pos = s["numero_serie"].map({u: i for i, u in enumerate(unidades)}).values
    vals = []
    for w in rng.multinomial(len(unidades), np.full(len(unidades), 1 / len(unidades)), size=n):
        p = w[pos]
        if y[p > 0].min() != y[p > 0].max():
            vals.append(roc_auc_score(y, s["escore"].values, sample_weight=p))
    return roc_auc_score(y, s["escore"].values), np.quantile(vals, 0.025), np.quantile(vals, 0.975)


linhas = []
rng = np.random.default_rng(2026)
for nome, s in (("referencia_N01_rotulado", avaliar(True)), ("referencia_primeiros_ciclos_sem_rotulo", avaliar(False))):
    a, i, h = auc_ic(s, rng)
    fal, n01 = s["condicao"].isin(FALHAS), s["condicao"] == "N01"
    print(f"\n{nome}: linhas {len(s)}, SN {s.numero_serie.nunique()}, SN com falha {s[fal].numero_serie.nunique()}, AUC {a:.3f} [{i:.3f}; {h:.3f}]")
    for t in LIMIARES:
        al = s["escore"] > t
        linhas.append({"referencia": nome, "limiar": t, "auc": a, "auc_inf": i, "auc_sup": h, "recall_falha": al[fal].mean(), "falso_alarme_N01": al[n01].mean(),
                       **{f"recall_{c}": al[s["condicao"] == c].mean() for c in FALHAS}})
    print(pd.DataFrame(linhas).query("referencia == @nome").drop(columns=["referencia", "auc", "auc_inf", "auc_sup"]).round(3).to_string(index=False))
    if nome == "referencia_N01_rotulado":
        print(s[fal].groupby(["condicao", "numero_serie"])["escore"].agg(["size", "min", "median", "max"]).round(3).to_string())
        print("N01 acima de 2% por SN:", s[n01 & (s.escore > 0.02)].numero_serie.value_counts().to_dict())
        print("quantis do escore em N01:", s[n01].escore.quantile([.5, .9, .95, .99]).round(4).to_dict())
pd.DataFrame(linhas).to_csv(b.RAIZ / "resultados" / "s2_regra_nivel.csv", index=False, lineterminator="\n")
