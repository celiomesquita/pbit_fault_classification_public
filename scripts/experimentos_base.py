"""Executa P1 e P2 do plano: divisoes versionadas e experimentos E0, E1, E2, E3a e E3b sobre a v3.
Uso: py scripts/experimentos_base.py [--rapido]. Saidas em resultados/ (ou resultados_teste/ com --rapido)."""
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent.parent
RAPIDO = "--rapido" in sys.argv
SAIDA = RAIZ / ("resultados_teste" if RAPIDO else "resultados")
N_REP = 1 if RAPIDO else 5
N_FOLD = 5
N_BOOT = 200 if RAPIDO else 2000
ALFAS = (0.05, 0.10, 0.20)
CLASSES = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]
POS = {c: i for i, c in enumerate(CLASSES)}
FALHAS = [POS[c] for c in CLASSES[2:]]
LIMITE_F06 = 0.05
LOG = ["desvio_padrao", "pico_a_pico", "maior_salto", "desvio_primeira_metade", "desvio_segunda_metade"]
ESTRATEGIAS = ["E1", "E2", "E3a", "E3b"]

GRADES = {
    "RF": [{"min_samples_leaf": a, "max_features": b} for a, b in product((1, 3, 5), ("sqrt", 0.5))],
    "LR": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
}
if RAPIDO:
    GRADES = {k: v[:1] for k, v in GRADES.items()}


def novo_modelo(nome, params, semente):
    if nome == "RF":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=semente, n_jobs=-1, **params)
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000, **params))


def proba_alinhada(modelo, X):
    p = modelo.predict_proba(X)
    saida = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(modelo.classes_):
        saida[:, POS[c]] = p[:, j]
    return saida


def dobras(d, rep):
    y, g = d["condicao"].values, d["numero_serie"].values
    chave = pd.Series(g + "|" + y)
    n = chave.map(chave.value_counts())
    estratos = np.where(n < N_FOLD, "raro|" + d["condicao"], chave)
    geradores = {
        "E1": KFold(N_FOLD, shuffle=True, random_state=rep).split(y),
        "E2": StratifiedKFold(N_FOLD, shuffle=True, random_state=rep).split(y, y),
        "E3a": StratifiedKFold(N_FOLD, shuffle=True, random_state=rep).split(y, estratos),
        "E3b": StratifiedGroupKFold(N_FOLD, shuffle=True, random_state=rep).split(y, y, g),
    }
    saida = {}
    for nome, ger in geradores.items():
        f = np.full(len(y), -1)
        for k, (_, te) in enumerate(ger):
            f[te] = k
        assert (f >= 0).all()
        saida[nome] = f
    return saida


def ajustar_interno(nome, X, y, g, semente):
    """Escolhe hiperparametros por macro-recall em validacao interna agrupada por SN e devolve as probabilidades internas."""
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=semente).split(X, y, g))
    melhor, melhor_score, melhor_p = None, -1.0, None
    for params in GRADES[nome]:
        p = np.zeros((len(y), len(CLASSES)))
        for tr, te in cv:
            p[te] = proba_alinhada(novo_modelo(nome, params, semente).fit(X[tr], y[tr]), X[te])
        pred = np.array(CLASSES)[p.argmax(1)]
        score = np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])
        if score > melhor_score:
            melhor, melhor_score, melhor_p = params, score, p
    return melhor, melhor_score, melhor_p


def confusao(y, pred):
    c = np.zeros((len(CLASSES), len(CLASSES)))
    np.add.at(c, (pd.Series(y).map(POS).values, pd.Series(pred).map(POS).values), 1)
    return c


def razao(a, b):
    return a / b if b > 0 else np.nan


def metricas(c):
    linhas = c.sum(1)
    colunas = c.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(c) / linhas
        prec = np.diag(c) / colunas
        f1 = 2 * prec * rec / (prec + rec)
    f1 = np.where(linhas > 0, np.nan_to_num(f1), np.nan)
    sem_f06 = [POS[x] for x in CLASSES[2:7]]
    fundido = c.copy()
    a, b = POS["F04"], POS["F05"]
    fundido[a, :] += fundido[b, :]
    fundido[:, a] += fundido[:, b]
    fundido = np.delete(np.delete(fundido, b, 0), b, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec_f = np.diag(fundido) / fundido.sum(1)
    n = c.sum()
    t, p = linhas, colunas
    s = np.trace(c)
    den = np.sqrt((n**2 - (p**2).sum()) * (n**2 - (t**2).sum()))
    pred_falha = c[:, FALHAS].sum(1)
    m = {
        "recall_falha": razao(pred_falha[FALHAS].sum(), linhas[FALHAS].sum()),
        "recall_falha_sem_F06": razao(pred_falha[sem_f06].sum(), linhas[sem_f06].sum()),
        "macro_recall": np.nanmean(rec),
        "macro_recall_sem_N02": np.nanmean(np.delete(rec, POS["N02"])),
        "macro_recall_F04F05_reunidas": np.nanmean(rec_f),
        "recall_anormalidade": razao(linhas[1:].sum() - c[1:, 0].sum(), linhas[1:].sum()),
        "falso_alarme_N01": razao(linhas[0] - c[0, 0], linhas[0]),
        "precisao_falha": razao(pred_falha[FALHAS].sum(), pred_falha.sum()),
        "macro_f1": np.nanmean(f1),
        "mcc": (s * n - (t * p).sum()) / den if den > 0 else np.nan,
    }
    for cl in CLASSES:
        m[f"recall_{cl}"] = rec[POS[cl]]
    return m


def bootstrap(tensores, unidades, rng):
    """IC por SN. Os mesmos pesos valem para todas as configuracoes, o que torna as diferencas pareadas."""
    pesos = rng.multinomial(len(unidades), np.full(len(unidades), 1 / len(unidades)), size=N_BOOT)
    amostras = {k: [metricas(np.tensordot(w, t, axes=1)) for w in pesos] for k, t in tensores.items()}
    return {k: pd.DataFrame(v) for k, v in amostras.items()}


def tensor(pred, unidades):
    idx = {u: i for i, u in enumerate(unidades)}
    t = np.zeros((len(unidades), len(CLASSES), len(CLASSES)))
    np.add.at(t, (pred["numero_serie"].map(idx).values, pred["condicao"].map(POS).values, pred["pred"].map(POS).values), 1)
    return t


def executar_modelos(d, X, y, g):
    pred, escolhas, div = [], [], []
    inicio = time.time()
    for rep in range(N_REP):
        folds = dobras(d, rep)
        for est in ESTRATEGIAS:
            f = folds[est]
            div.append(pd.DataFrame({"experimento_idx": d["experimento_idx"], "estrategia": est, "repeticao": rep, "fold": f}))
            if est == "E3b":
                for k in range(N_FOLD):
                    assert not set(g[f == k]) & set(g[f != k])
            for nome, k in product(GRADES, range(N_FOLD)):
                tr, te = f != k, f == k
                sem = 1000 * rep + k
                params, score, p_in = ajustar_interno(nome, X[tr], y[tr], g[tr], sem)
                fal_in = p_in[:, FALHAS].sum(1)[y[tr] == "N01"]
                modelo = novo_modelo(nome, params, sem).fit(X[tr], y[tr])
                p = proba_alinhada(modelo, X[te])
                linha = pd.DataFrame({"estrategia": est, "modelo": nome, "repeticao": rep, "fold": k,
                                      "experimento_idx": d["experimento_idx"].values[te], "numero_serie": g[te], "condicao": y[te],
                                      "pred": np.array(CLASSES)[p.argmax(1)], "p_n01": p[:, 0], "p_falha": p[:, FALHAS].sum(1)})
                for a in ALFAS:
                    linha[f"tau_{int(a * 100):02d}"] = np.quantile(fal_in, 1 - a)
                pred.append(linha)
                escolhas.append({"estrategia": est, "modelo": nome, "repeticao": rep, "fold": k, "params": str(params), "macro_recall_interno": score})
            print(f"rep {rep} {est} {time.time() - inicio:.0f}s", flush=True)
    return pd.concat(pred, ignore_index=True), pd.DataFrame(escolhas), pd.concat(div, ignore_index=True)


def executar_baselines(d, X, y, g):
    saida = []
    for rep in range(N_REP):
        f = dobras(d, rep)["E3b"]
        for k in range(N_FOLD):
            tr, te = f != k, f == k
            base = {"estrategia": "E0", "repeticao": rep, "fold": k, "experimento_idx": d["experimento_idx"].values[te],
                    "numero_serie": g[te], "condicao": y[te]}
            maioria = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr]).predict(X[te])
            acaso = DummyClassifier(strategy="stratified", random_state=1000 * rep + k).fit(X[tr], y[tr]).predict(X[te])
            regra = np.where(d["minimo"].values[te] <= LIMITE_F06, "F06", maioria)
            for nome, pr in (("maioria", maioria), ("acaso_estratificado", acaso), ("regra_F06", regra)):
                saida.append(pd.DataFrame({**base, "modelo": nome, "pred": pr}))
    return pd.concat(saida, ignore_index=True)


def pontos_operacao(pred):
    linhas = []
    for (est, mod), g in pred.groupby(["estrategia", "modelo"]):
        for a in ALFAS:
            tau = f"tau_{int(a * 100):02d}"
            al = g["p_falha"] > g[tau]
            fal = g["condicao"].isin(CLASSES[2:])
            sem = g["condicao"].isin(CLASSES[2:7])
            n01 = g["condicao"] == "N01"
            porrep = pd.DataFrame({"r": g["repeticao"], "al": al, "fal": fal, "sem": sem, "n01": n01})
            rr = porrep.groupby("r").apply(lambda x: pd.Series({
                "recall_falha": x.al[x.fal].mean(), "recall_sem_F06": x.al[x["sem"]].mean(), "falso_alarme_real": x.al[x.n01].mean()}))
            linhas.append({"estrategia": est, "modelo": mod, "falso_alarme_alvo": a, **rr.mean().round(4).to_dict(),
                           "recall_falha_min": rr.recall_falha.min(), "recall_falha_max": rr.recall_falha.max()})
    return pd.DataFrame(linhas)


def auc(pred):
    linhas = []
    for (est, mod, r), g in pred.groupby(["estrategia", "modelo", "repeticao"]):
        s = g[g["condicao"] != "N02"]
        linhas.append({"estrategia": est, "modelo": mod, "repeticao": r,
                       "auc_falha_vs_N01": roc_auc_score(s["condicao"].isin(CLASSES[2:]), s["p_falha"])})
    return pd.DataFrame(linhas).groupby(["estrategia", "modelo"])["auc_falha_vs_N01"].agg(["mean", "min", "max"]).round(4).reset_index()


if __name__ == "__main__":
    SAIDA.mkdir(exist_ok=True)
    d = pd.read_csv(RAIZ / "dados" / "pbit_features_corrigido_v3.csv")
    F = list(d.columns[3:])
    Xdf = d[F].copy()
    for c in LOG:
        Xdf[c] = np.log10(Xdf[c])
    X, y, g = Xdf.values, d["condicao"].values, d["numero_serie"].values
    unidades = sorted(set(g))

    pred, escolhas, div = executar_modelos(d, X, y, g)
    base = executar_baselines(d, X, y, g)
    todos = pd.concat([pred, base], ignore_index=True)

    div.to_csv(SAIDA / "divisoes.csv", index=False, lineterminator="\n")
    pred.to_csv(SAIDA / "predicoes_fora_da_amostra.csv", index=False, lineterminator="\n")
    base.to_csv(SAIDA / "predicoes_baselines.csv", index=False, lineterminator="\n")
    escolhas.to_csv(SAIDA / "hiperparametros.csv", index=False, lineterminator="\n")

    porrep = []
    for (est, mod, r), x in todos.groupby(["estrategia", "modelo", "repeticao"]):
        porrep.append({"estrategia": est, "modelo": mod, "repeticao": r, **metricas(confusao(x["condicao"], x["pred"]))})
    porrep = pd.DataFrame(porrep)
    porrep.to_csv(SAIDA / "metricas_por_repeticao.csv", index=False, lineterminator="\n")

    tensores = {k: tensor(x, unidades) for k, x in todos.groupby(["estrategia", "modelo"])}
    boots = bootstrap(tensores, unidades, np.random.default_rng(2026))
    ic = []
    for (est, mod), b in boots.items():
        ponto = metricas(tensores[(est, mod)].sum(0))
        for m in b.columns:
            ic.append({"estrategia": est, "modelo": mod, "metrica": m, "estimativa": ponto[m],
                       "ic95_inf": b[m].quantile(0.025), "ic95_sup": b[m].quantile(0.975)})
    ic = pd.DataFrame(ic)
    ic.to_csv(SAIDA / "metricas_ic95_por_sn.csv", index=False, lineterminator="\n")

    dif = []
    for mod, est, m in product(GRADES, ("E1", "E2", "E3a"), ("recall_falha", "recall_falha_sem_F06", "macro_recall", "macro_recall_sem_N02", "falso_alarme_N01")):
        a, b = (est, mod), ("E3b", mod)
        ponto = metricas(tensores[a].sum(0))[m] - metricas(tensores[b].sum(0))[m]
        delta = boots[a][m] - boots[b][m]
        dif.append({"modelo": mod, "comparacao": f"{est}-E3b", "metrica": m, "diferenca": ponto,
                    "ic95_inf": delta.quantile(0.025), "ic95_sup": delta.quantile(0.975)})
    dif = pd.DataFrame(dif)
    dif.to_csv(SAIDA / "diferencas_vazamento_ic95.csv", index=False, lineterminator="\n")

    op = pontos_operacao(pred)
    op.to_csv(SAIDA / "pontos_de_operacao.csv", index=False, lineterminator="\n")
    au = auc(pred)
    au.to_csv(SAIDA / "auc_falha_vs_n01.csv", index=False, lineterminator="\n")

    confs = []
    for (est, mod), t in tensores.items():
        c = pd.DataFrame(t.sum(0) / N_REP, index=CLASSES, columns=CLASSES).round(1)
        c.insert(0, "condicao", CLASSES)
        c.insert(0, "modelo", mod)
        c.insert(0, "estrategia", est)
        confs.append(c)
    pd.concat(confs).to_csv(SAIDA / "confusao_media.csv", index=False, lineterminator="\n")

    x = todos[todos["modelo"].isin(GRADES)]
    un = x.assign(ok=x["pred"] == x["condicao"]).groupby(["estrategia", "modelo", "condicao", "numero_serie"])["ok"].mean().reset_index()
    un.to_csv(SAIDA / "recall_por_sn.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(ic.round(3).to_string(index=False))
    print(dif.round(3).to_string(index=False))
    print(op.round(3).to_string(index=False))
    print(au.to_string(index=False))
    print(un.groupby(["estrategia", "modelo", "condicao"])["ok"].mean().unstack().round(3).to_string())
    print(pd.read_csv(SAIDA / "confusao_media.csv").query("estrategia=='E3b' and modelo=='RF'").to_string(index=False))
    print(escolhas.groupby(["estrategia", "modelo", "params"]).size().to_string())
    ov = [(np.isin(g[div_f == k], g[div_f != k]).mean()) for est in ESTRATEGIAS for div_f in [div[(div.estrategia == est) & (div.repeticao == 0)].fold.values] for k in range(N_FOLD)]
    print("linhas de teste com SN no treino, por estrategia:", dict(zip(ESTRATEGIAS, np.round(np.array(ov).reshape(4, N_FOLD).mean(1), 3))))
