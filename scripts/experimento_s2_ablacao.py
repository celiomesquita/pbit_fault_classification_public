"""Executa P6 do plano: ablacao de features (S2) na v3, em E1 e E3b.
A: 18 features. B: sem nivel absoluto. C: nivel relativo ao N01 da propria unidade, so nas SN com N01.
Referencia de C: mediana dos 3 primeiros N01 da SN por experimento_idx; esses ciclos saem do treino e do teste, porque
usar N01 avaliados na referencia vazaria o rotulo.
Uso, a partir de scripts/: py experimento_s2_ablacao.py [--rapido]. Requer resultados/predicoes_fora_da_amostra.csv."""
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import experimentos_base as b

NIVEL = ["media", "mediana", "minimo", "maximo", "rms", "valor_inicial", "valor_final", "energia"]
RELATIVAS = ["media", "mediana", "minimo", "maximo", "rms", "valor_inicial", "valor_final"]
K_REF = 3
CRITICAS = [b.POS["F03"], b.POS["F06"]]
SEM_EFEITO = [b.POS["N01"], b.POS["N02"]]
ESTS = ["E1", "E3b"]
MODELOS = list(b.GRADES)
COMPARADAS = ["recall_falha", "recall_falha_sem_F06", "macro_recall", "macro_recall_sem_N02", "falso_alarme_N01", "cmr"]


def matriz(d, colunas):
    x = d[colunas].copy()
    for c in b.LOG:
        if c in x:
            x[c] = np.log10(x[c])
    return x.values


def preparar_subconjunto(d):
    com_n01 = d[d["condicao"] == "N01"].sort_values("experimento_idx").groupby("numero_serie")
    ref_linhas = com_n01.head(K_REF)
    ref = ref_linhas.groupby("numero_serie")["media"].median()
    s = d[d["numero_serie"].isin(ref.index) & ~d["experimento_idx"].isin(ref_linhas["experimento_idx"])].reset_index(drop=True).copy()
    r = s["numero_serie"].map(ref)
    rel = s.copy()
    for c in RELATIVAS:
        rel[c] = s[c] / r
    return s, rel


def executar(d, X, conjunto, populacao):
    y, g = d["condicao"].values, d["numero_serie"].values
    saida = []
    inicio = time.time()
    for rep, est in product(range(b.N_REP), ESTS):
        f = b.dobras(d, rep)[est]
        for nome, k in product(MODELOS, range(b.N_FOLD)):
            tr, te = f != k, f == k
            sem = 1000 * rep + k
            params, _, p_in = b.ajustar_interno(nome, X[tr], y[tr], g[tr], sem)
            fal_in = p_in[:, b.FALHAS].sum(1)[y[tr] == "N01"]
            p = b.proba_alinhada(b.novo_modelo(nome, params, sem).fit(X[tr], y[tr]), X[te])
            lin = pd.DataFrame({"populacao": populacao, "conjunto": conjunto, "estrategia": est, "modelo": nome, "repeticao": rep, "fold": k,
                                "experimento_idx": d["experimento_idx"].values[te], "numero_serie": g[te], "condicao": y[te],
                                "pred": np.array(b.CLASSES)[p.argmax(1)], "p_falha": p[:, b.FALHAS].sum(1)})
            for a in b.ALFAS:
                lin[f"tau_{int(a * 100):02d}"] = np.quantile(fal_in, 1 - a)
            saida.append(lin)
        print(f"{populacao} {conjunto} rep {rep} {est} {time.time() - inicio:.0f}s", flush=True)
    return pd.concat(saida, ignore_index=True)


def metricas_cmr(c):
    m = b.metricas(c)
    m["cmr"] = b.razao(c[np.ix_(CRITICAS, SEM_EFEITO)].sum(), c[CRITICAS].sum())
    return m


def bootstrap(tensores, n_unidades, rng):
    pesos = rng.multinomial(n_unidades, np.full(n_unidades, 1 / n_unidades), size=b.N_BOOT)
    return {k: pd.DataFrame([metricas_cmr(np.tensordot(w, t, axes=1)) for w in pesos]) for k, t in tensores.items()}


def resumir(todos, populacao, rng):
    x = todos[todos.populacao == populacao]
    unidades = sorted(x["numero_serie"].unique())
    tens = {k: b.tensor(v, unidades) for k, v in x.groupby(["conjunto", "estrategia", "modelo"])}
    boots = bootstrap(tens, len(unidades), rng)
    ic = []
    for k, bt in boots.items():
        ponto = metricas_cmr(tens[k].sum(0))
        for m in bt.columns:
            ic.append({"populacao": populacao, "conjunto": k[0], "estrategia": k[1], "modelo": k[2], "metrica": m, "estimativa": ponto[m],
                       "ic95_inf": bt[m].quantile(0.025), "ic95_sup": bt[m].quantile(0.975)})
    conj = sorted({k[0] for k in tens})
    dif = []
    for c, est, mod, m in product([c for c in conj if c != "A"], ESTS, MODELOS, COMPARADAS):
        a, r = (c, est, mod), ("A", est, mod)
        delta = boots[a][m] - boots[r][m]
        dif.append({"populacao": populacao, "comparacao": f"{c}-A", "estrategia": est, "modelo": mod, "metrica": m,
                    "diferenca": metricas_cmr(tens[a].sum(0))[m] - metricas_cmr(tens[r].sum(0))[m],
                    "ic95_inf": delta.quantile(0.025), "ic95_sup": delta.quantile(0.975)})
    # H5: a perda ao retirar o nivel e maior com vazamento do que em SN novos?
    for mod, m in product(MODELOS, ("recall_falha", "macro_recall", "falso_alarme_N01")):
        if "B" not in conj:
            continue
        pe = lambda c, est: metricas_cmr(tens[(c, est, mod)].sum(0))[m]
        delta = (boots[("A", "E1", mod)][m] - boots[("B", "E1", mod)][m]) - (boots[("A", "E3b", mod)][m] - boots[("B", "E3b", mod)][m])
        dif.append({"populacao": populacao, "comparacao": "perda_E1-perda_E3b", "estrategia": "E1,E3b", "modelo": mod, "metrica": m,
                    "diferenca": (pe("A", "E1") - pe("B", "E1")) - (pe("A", "E3b") - pe("B", "E3b")),
                    "ic95_inf": delta.quantile(0.025), "ic95_sup": delta.quantile(0.975)})
    return pd.DataFrame(ic), pd.DataFrame(dif)


def op_auc(todos):
    linhas = []
    for (pop, c, est, mod), x in todos.groupby(["populacao", "conjunto", "estrategia", "modelo"]):
        fal, n01 = x["condicao"].isin(b.CLASSES[2:]), x["condicao"] == "N01"
        vals = []
        for _, r in x.groupby("repeticao"):
            s = r[r["condicao"] != "N02"]
            vals.append(roc_auc_score(s["condicao"].isin(b.CLASSES[2:]), s["p_falha"]))
        reg = {"populacao": pop, "conjunto": c, "estrategia": est, "modelo": mod, "auc": np.mean(vals)}
        for a in b.ALFAS:
            al = x["p_falha"] > x[f"tau_{int(a * 100):02d}"]
            reg[f"recall_alvo_{int(a * 100):02d}"] = al[fal].mean()
            reg[f"fa_real_{int(a * 100):02d}"] = al[n01].mean()
        linhas.append(reg)
    return pd.DataFrame(linhas)


def identificar_sn(X, g):
    acc = []
    for rep in range(b.N_REP):
        pred = np.empty(len(g), dtype=object)
        for tr, te in KFold(b.N_FOLD, shuffle=True, random_state=rep).split(X):
            pred[te] = RandomForestClassifier(200, random_state=rep, n_jobs=-1).fit(X[tr], g[tr]).predict(X[te])
        acc.append((pred == g).mean())
    return np.mean(acc)


if __name__ == "__main__":
    b.SAIDA.mkdir(exist_ok=True)
    d = pd.read_csv(b.RAIZ / "dados" / "pbit_features_corrigido_v3.csv")
    F = list(d.columns[3:])
    B_COL = [c for c in F if c not in NIVEL]
    C_COL = B_COL + RELATIVAS
    s, rel = preparar_subconjunto(d)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 40)
    print("features B:", B_COL)
    print("subconjunto:", len(s), "linhas,", s["numero_serie"].nunique(), "SN")
    print(s.groupby("condicao").agg(linhas=("condicao", "size"), sn=("numero_serie", "nunique")).to_string(), flush=True)

    sn = pd.DataFrame([{"conjunto": c, "acuracia_identificacao_sn": identificar_sn(matriz(d, col), d["numero_serie"].values)}
                       for c, col in (("A", F), ("B", B_COL))])
    sn.to_csv(b.SAIDA / "s2_identificacao_sn.csv", index=False, lineterminator="\n")
    print(sn.round(3).to_string(index=False), flush=True)

    base = pd.read_csv(b.RAIZ / "resultados" / "predicoes_fora_da_amostra.csv")
    base = base[base.estrategia.isin(ESTS) & (base.repeticao < b.N_REP) & base.modelo.isin(MODELOS)].drop(columns="p_n01").assign(populacao="completa", conjunto="A")
    partes = [base, executar(d, matriz(d, B_COL), "B", "completa"), executar(s, matriz(s, F), "A", "com_N01"),
              executar(s, matriz(s, B_COL), "B", "com_N01"), executar(rel, matriz(rel, C_COL), "C", "com_N01")]
    todos = pd.concat(partes, ignore_index=True)
    todos[todos.populacao != "completa"].to_csv(b.SAIDA / "s2_predicoes_com_n01.csv", index=False, lineterminator="\n")
    todos[(todos.populacao == "completa") & (todos.conjunto == "B")].to_csv(b.SAIDA / "s2_predicoes_completa_B.csv", index=False, lineterminator="\n")

    rng = np.random.default_rng(2026)
    ics, difs = zip(*(resumir(todos, p, rng) for p in ("completa", "com_N01")))
    ic, dif = pd.concat(ics), pd.concat(difs)
    ic.to_csv(b.SAIDA / "s2_metricas_ic95_por_sn.csv", index=False, lineterminator="\n")
    dif.to_csv(b.SAIDA / "s2_diferencas_ic95.csv", index=False, lineterminator="\n")
    op = op_auc(todos)
    op.to_csv(b.SAIDA / "s2_auc_pontos_de_operacao.csv", index=False, lineterminator="\n")

    ic["v"] = ic.estimativa.round(3).astype(str) + " [" + ic.ic95_inf.round(3).astype(str) + "; " + ic.ic95_sup.round(3).astype(str) + "]"
    print(ic[ic.metrica.isin(COMPARADAS)].pivot_table(index=["populacao", "conjunto", "estrategia", "modelo"], columns="metrica", values="v", aggfunc="first")[COMPARADAS].to_string())
    print(ic[ic.metrica.str.match("recall_[NF]0")].pivot_table(index=["populacao", "conjunto", "estrategia", "modelo"], columns="metrica", values="estimativa").round(3).to_string())
    print(dif.round(3).to_string(index=False))
    print(op.round(3).to_string(index=False))
