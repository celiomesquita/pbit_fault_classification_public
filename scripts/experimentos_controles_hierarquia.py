"""Executa P3 e P4 do plano: controles C1 e C2 e hierarquia E4 sobre a v3, com as divisoes de experimentos_base.py.
Uso: py scripts/experimentos_controles_hierarquia.py [--rapido]. Requer resultados/predicoes_fora_da_amostra.csv."""
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold

import experimentos_base as b

N_PERM = 5 if b.RAPIDO else 200
PARAMS_FIXOS = {"min_samples_leaf": 3, "max_features": 0.5}
ALFAS = b.ALFAS
MODELOS = list(b.GRADES)
REF = "plano"
CONFIGS = ["plano_regra_F06", "dois_niveis", "tres_niveis", "tres_niveis_F04F05_reunidas"]
COMPARADAS = ["recall_falha", "recall_falha_sem_F06", "macro_recall", "macro_recall_sem_N02", "macro_recall_F04F05_reunidas",
              "recall_anormalidade", "falso_alarme_N01", "precisao_falha"]


def carregar():
    d = pd.read_csv(b.RAIZ / "dados" / "pbit_features_v3.csv")
    Xdf = d[list(d.columns[3:])].copy()
    for c in b.LOG:
        Xdf[c] = np.log10(Xdf[c])
    return d, Xdf.values, d["condicao"].values, d["numero_serie"].values, d["minimo"].values


def rf_fixa(semente):
    return b.novo_modelo("RF", PARAMS_FIXOS, semente)


# ---------------------------------------------------------------- C2
def controle_c2(X, g):
    linhas = []
    for rep in range(b.N_REP):
        pred = np.empty(len(g), dtype=object)
        for tr, te in KFold(b.N_FOLD, shuffle=True, random_state=rep).split(X):
            pred[te] = RandomForestClassifier(200, random_state=rep, n_jobs=-1, min_samples_leaf=1).fit(X[tr], g[tr]).predict(X[te])
        recs = [(pred[g == u] == u).mean() for u in np.unique(g)]
        linhas.append({"repeticao": rep, "acuracia": (pred == g).mean(), "macro_recall_sn": np.mean(recs),
                       "acuracia_maioria": pd.Series(g).value_counts(normalize=True).iloc[0], "acaso_macro": 1 / len(set(g))})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- C1
def validar_fixa(X, y, g, est, semente):
    if est == "E1":
        ger = KFold(b.N_FOLD, shuffle=True, random_state=semente).split(X)
    else:
        ger = StratifiedGroupKFold(b.N_FOLD, shuffle=True, random_state=semente).split(X, y, g)
    pred = np.empty(len(y), dtype=object)
    for k, (tr, te) in enumerate(ger):
        pred[te] = rf_fixa(1000 * semente + k).fit(X[tr], y[tr]).predict(X[te])
    return b.metricas(b.confusao(y, pred))


def controle_c1(y, g, X):
    blocos = pd.Series(g + "|" + y)
    unicos = blocos.drop_duplicates()
    rotulo = unicos.str.split("|").str[1].values
    linhas = []
    for est in ("E1", "E3b"):
        for s in range(b.N_REP):
            linhas.append({"estrategia": est, "tipo": "real", "semente": s, **validar_fixa(X, y, g, est, s)})
    rng = np.random.default_rng(2026)
    inicio = time.time()
    for p in range(N_PERM):
        mapa = dict(zip(unicos.values, rng.permutation(rotulo)))
        yp = blocos.map(mapa).values
        for est in ("E1", "E3b"):
            linhas.append({"estrategia": est, "tipo": "permutado", "semente": p, **validar_fixa(X, yp, g, est, p)})
        if p % 20 == 0:
            print(f"C1 perm {p} {time.time() - inicio:.0f}s", flush=True)
    return pd.DataFrame(linhas)


def resumo_c1(c1):
    linhas = []
    for est, m in product(("E1", "E3b"), [c for c in c1.columns if c.startswith(("macro_recall", "recall_", "falso_"))]):
        x = c1[c1.estrategia == est]
        real, nulo = x[x.tipo == "real"][m].mean(), x[x.tipo == "permutado"][m].dropna()
        cauda = (nulo <= real).sum() if m == "falso_alarme_N01" else (nulo >= real).sum()
        linhas.append({"estrategia": est, "metrica": m, "real": real, "nulo_media": nulo.mean(), "nulo_p2.5": nulo.quantile(0.025),
                       "nulo_p97.5": nulo.quantile(0.975), "p": (cauda + 1) / (len(nulo) + 1), "n_perm": len(nulo)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- E4
def ajustar(nome, X, y, g, semente):
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=semente).split(X, y, g))
    melhor, melhor_score = None, -1.0
    for params in b.GRADES[nome]:
        pred = np.empty(len(y), dtype=object)
        for tr, te in cv:
            pred[te] = b.novo_modelo(nome, params, semente).fit(X[tr], y[tr]).predict(X[te])
        score = np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])
        if score > melhor_score:
            melhor, melhor_score = params, score
    return melhor


def treinar(config, nome, X, y, g, minimo, semente, params=None):
    regra = config != "dois_niveis"
    m = (minimo > b.LIMITE_F06) if regra else np.ones(len(y), bool)
    Xa, ya, ga = X[m], y[m], g[m]
    h = {"config": config, "regra": regra}
    if config == "plano_regra_F06":
        p = params["plano"] if params else ajustar(nome, Xa, ya, ga, semente)
        h.update(plano=b.novo_modelo(nome, p, semente).fit(Xa, ya), params={"plano": p})
        return h
    yb = np.where(ya == "N01", "N01", "ANOM")
    resto = ya != "N01"
    y3 = np.where(ya == "F05", "F04", ya) if config == "tres_niveis_F04F05_reunidas" else ya
    p2 = params["n2"] if params else ajustar(nome, Xa, yb, ga, semente)
    p3 = params["n3"] if params else ajustar(nome, Xa[resto], y3[resto], ga[resto], semente + 1)
    h.update(n2=b.novo_modelo(nome, p2, semente).fit(Xa, yb), n3=b.novo_modelo(nome, p3, semente).fit(Xa[resto], y3[resto]), params={"n2": p2, "n3": p3})
    return h


def prever(h, X, minimo):
    nomes = np.array(b.CLASSES)
    if "plano" in h:
        p = b.proba_alinhada(h["plano"], X)
        pred, falha = nomes[p.argmax(1)], p[:, b.FALHAS].sum(1)
        p_anom, p3 = 1 - p[:, 0], None
    else:
        col = list(h["n2"].classes_).index("ANOM")
        p_anom = h["n2"].predict_proba(X)[:, col]
        p3 = b.proba_alinhada(h["n3"], X)
        pred = np.where(p_anom >= 0.5, nomes[p3.argmax(1)], "N01")
        falha = p_anom * (1 - p3[:, b.POS["N02"]])
    if h["regra"]:
        hit = minimo <= b.LIMITE_F06
        pred = np.where(hit, "F06", pred)
        falha = np.where(hit, 1.0, falha)
    return pred, falha, p_anom, p3


def limiares(config, nome, X, y, g, minimo, semente, params):
    """Limiar do escore de falha pela distribuicao interna em N01 fora da amostra, sem reajustar hiperparametros."""
    esc = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=semente + 2).split(X, y, g):
        h = treinar(config, nome, X[tr], y[tr], g[tr], minimo[tr], semente, params)
        esc[te] = prever(h, X[te], minimo[te])[1]
    return {a: np.quantile(esc[y == "N01"], 1 - a) for a in ALFAS}


def executar_e4(d, X, y, g, minimo):
    pred, niveis, hip = [], [], []
    inicio = time.time()
    for rep in range(b.N_REP):
        f = b.dobras(d, rep)["E3b"]
        for nome, config, k in product(MODELOS, CONFIGS, range(b.N_FOLD)):
            tr, te = f != k, f == k
            sem = 1000 * rep + k
            h = treinar(config, nome, X[tr], y[tr], g[tr], minimo[tr], sem)
            tau = limiares(config, nome, X[tr], y[tr], g[tr], minimo[tr], sem, h["params"])
            pr, falha, p_anom, p3 = prever(h, X[te], minimo[te])
            lin = pd.DataFrame({"config": config, "modelo": nome, "repeticao": rep, "fold": k, "experimento_idx": d["experimento_idx"].values[te],
                                "numero_serie": g[te], "condicao": y[te], "pred": pr, "p_falha": falha})
            for a in ALFAS:
                lin[f"tau_{int(a * 100):02d}"] = tau[a]
            pred.append(lin)
            hip.append({"config": config, "modelo": nome, "repeticao": rep, "fold": k, "params": str(h["params"])})
            if config == "tres_niveis":
                yt, mt = y[te], minimo[te] > b.LIMITE_F06
                n = pd.DataFrame({"modelo": nome, "repeticao": rep, "condicao": yt[mt], "p_anom": p_anom[mt],
                                  "pred_n3": np.array(b.CLASSES)[p3[mt].argmax(1)]})
                niveis.append(n)
        print(f"E4 rep {rep} {time.time() - inicio:.0f}s", flush=True)
    return pd.concat(pred, ignore_index=True), pd.concat(niveis, ignore_index=True), pd.DataFrame(hip)


def resumo_niveis(n):
    linhas = []
    for (nome, rep), x in n.groupby(["modelo", "repeticao"]):
        anom = x["condicao"] != "N01"
        al = x["p_anom"] >= 0.5
        r = {"modelo": nome, "repeticao": rep, "n2_recall_not_N01": al[anom].mean(), "n2_falso_alarme": al[~anom].mean(), "n2_auc": roc_auc_score(anom, x["p_anom"])}
        s = x[anom]
        for c in b.CLASSES[1:7]:
            r[f"n3_recall_{c}"] = (s[s.condicao == c].pred_n3 == c).mean()
        linhas.append(r)
    return pd.DataFrame(linhas).groupby("modelo").mean(numeric_only=True).drop(columns="repeticao").round(4).reset_index()


def alarme(x, a):
    return x["p_falha"] > x[f"tau_{int(a * 100):02d}"]


def op_e_auc(todos):
    op, au = [], []
    for (nome, cfg), x in todos.groupby(["modelo", "config"]):
        fal, sem, n01 = x["condicao"].isin(b.CLASSES[2:]), x["condicao"].isin(b.CLASSES[2:7]), x["condicao"] == "N01"
        for a in ALFAS:
            al = alarme(x, a)
            op.append({"modelo": nome, "config": cfg, "falso_alarme_alvo": a, "recall_falha": al[fal].mean(), "recall_falha_sem_F06": al[sem].mean(), "falso_alarme_real": al[n01].mean()})
        vals = []
        for _, r in x.groupby("repeticao"):
            s = r[r["condicao"] != "N02"]
            vals.append(roc_auc_score(s["condicao"].isin(b.CLASSES[2:]), s["p_falha"]))
        au.append({"modelo": nome, "config": cfg, "auc_falha_vs_N01": np.mean(vals), "min": np.min(vals), "max": np.max(vals)})
    return pd.DataFrame(op), pd.DataFrame(au)


def dif_op(todos, unidades, rng):
    idx = {u: i for i, u in enumerate(unidades)}
    pesos = rng.multinomial(len(unidades), np.full(len(unidades), 1 / len(unidades)), size=b.N_BOOT)

    def contagem(x, a):
        s = x[x["condicao"].isin(b.CLASSES[2:])]
        u = s["numero_serie"].map(idx).values
        al, n = np.zeros(len(unidades)), np.zeros(len(unidades))
        np.add.at(al, u, alarme(s, a).values.astype(float))
        np.add.at(n, u, 1.0)
        return al, n

    linhas = []
    for nome, cfg, a in product(MODELOS, CONFIGS, ALFAS):
        pa, na = contagem(todos[(todos.modelo == nome) & (todos.config == cfg)], a)
        pb, nb = contagem(todos[(todos.modelo == nome) & (todos.config == REF)], a)
        delta = (pesos @ pa) / (pesos @ na) - (pesos @ pb) / (pesos @ nb)
        linhas.append({"modelo": nome, "config": cfg, "falso_alarme_alvo": a, "diferenca_recall_falha": pa.sum() / na.sum() - pb.sum() / nb.sum(),
                       "ic95_inf": np.nanquantile(delta, 0.025), "ic95_sup": np.nanquantile(delta, 0.975)})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    b.SAIDA.mkdir(exist_ok=True)
    d, X, y, g, minimo = carregar()
    unidades = sorted(set(g))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 500)

    c2 = controle_c2(X, g)
    c2.to_csv(b.SAIDA / "c2_identificacao_sn.csv", index=False, lineterminator="\n")
    print(c2.round(3).to_string(index=False), flush=True)

    c1 = controle_c1(y, g, X)
    c1.to_csv(b.SAIDA / "c1_permutacoes.csv", index=False, lineterminator="\n")
    rc1 = resumo_c1(c1)
    rc1.to_csv(b.SAIDA / "c1_resumo.csv", index=False, lineterminator="\n")
    print(rc1.round(3).to_string(index=False), flush=True)

    pred, niveis, hip = executar_e4(d, X, y, g, minimo)
    base = pd.read_csv(b.RAIZ / "resultados" / "predicoes_fora_da_amostra.csv")
    base = base[(base.estrategia == "E3b") & (base.repeticao < b.N_REP) & (base.modelo.isin(MODELOS))].drop(columns=["estrategia", "p_n01"]).assign(config=REF)
    todos = pd.concat([base, pred], ignore_index=True)
    pred.to_csv(b.SAIDA / "e4_predicoes.csv", index=False, lineterminator="\n")
    hip.to_csv(b.SAIDA / "e4_hiperparametros.csv", index=False, lineterminator="\n")

    rn = resumo_niveis(niveis)
    rn.to_csv(b.SAIDA / "e4_metricas_por_nivel.csv", index=False, lineterminator="\n")

    tensores = {(m, c): b.tensor(x, unidades) for (m, c), x in todos.groupby(["modelo", "config"])}
    boots = b.bootstrap(tensores, unidades, np.random.default_rng(2026))
    ic, dif = [], []
    for k, bt in boots.items():
        ponto = b.metricas(tensores[k].sum(0))
        for m in bt.columns:
            ic.append({"modelo": k[0], "config": k[1], "metrica": m, "estimativa": ponto[m], "ic95_inf": bt[m].quantile(0.025), "ic95_sup": bt[m].quantile(0.975)})
    for nome, cfg, m in product(MODELOS, CONFIGS, COMPARADAS):
        a, r = (nome, cfg), (nome, REF)
        delta = boots[a][m] - boots[r][m]
        dif.append({"modelo": nome, "config": cfg, "metrica": m, "diferenca": b.metricas(tensores[a].sum(0))[m] - b.metricas(tensores[r].sum(0))[m],
                    "ic95_inf": delta.quantile(0.025), "ic95_sup": delta.quantile(0.975)})
    ic, dif = pd.DataFrame(ic), pd.DataFrame(dif)
    ic.to_csv(b.SAIDA / "e4_metricas_ic95_por_sn.csv", index=False, lineterminator="\n")
    dif.to_csv(b.SAIDA / "e4_diferencas_ic95.csv", index=False, lineterminator="\n")

    op, au = op_e_auc(todos)
    dop = dif_op(todos, unidades, np.random.default_rng(2027))
    op.to_csv(b.SAIDA / "e4_pontos_de_operacao.csv", index=False, lineterminator="\n")
    au.to_csv(b.SAIDA / "e4_auc_falha_vs_n01.csv", index=False, lineterminator="\n")
    dop.to_csv(b.SAIDA / "e4_diferencas_pontos_de_operacao.csv", index=False, lineterminator="\n")

    print(rn.to_string(index=False))
    print(ic[ic.metrica.isin(COMPARADAS)].pivot_table(index=["modelo", "config"], columns="metrica", values="estimativa").round(3).to_string())
    print(ic[ic.metrica.str.match("recall_[NF]0")].pivot_table(index=["modelo", "config"], columns="metrica", values="estimativa").round(3).to_string())
    print(dif.round(3).to_string(index=False))
    print(op.round(3).to_string(index=False))
    print(dop.round(3).to_string(index=False))
    print(au.round(3).to_string(index=False))
