"""Audita a continuidade do nivel de corrente entre ciclos vizinhos para localizar atribuicoes suspeitas de numero de serie
e condicao nas fronteiras de bloco. Nao altera rotulos; grava dados/pbit_auditoria_continuidade.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"
PP_REL_MAX = 0.15
TOL_SALTO = 0.02
TOL_MATCH = 0.01
TOL_FORTE = 0.005
K_CURTO = 2
N_PERM = 2000


def carregar():
    d = pd.read_csv(DADOS / "pbit_features_limpo.csv").sort_values("experimento_idx").reset_index(drop=True)
    d["confiavel"] = (d["pico_a_pico"] / d["media"].abs() < PP_REL_MAX) & (d["media"] > 0)
    return d


def montar_corridas(d):
    corridas = []
    for _, g in d.groupby((d["numero_serie"] != d["numero_serie"].shift()).cumsum()):
        segs, atual, prev = [], [], None
        for r in g.itertuples():
            if r.confiavel and prev is not None and atual and abs(r.media / prev - 1) > TOL_SALTO:
                segs.append(atual)
                atual = []
            atual.append(r)
            if r.confiavel:
                prev = r.media
        segs.append(atual)
        corridas.append({"serie": g["numero_serie"].iloc[0], "segs": [
            {"rows": s, "nivel": np.median([x.media for x in s if x.confiavel]) if any(x.confiavel for x in s) else np.nan}
            for s in segs]})
    return corridas


def condicao_modal(seg):
    return pd.Series([x.condicao for x in seg["rows"]]).mode().iloc[0]


def propostas(corridas, k=K_CURTO):
    out = []
    for a, b in zip(corridas[:-1], corridas[1:]):
        if a["serie"] == b["serie"]:
            continue
        casos = (("fim_de_A", a["segs"][-1], b["segs"][0], a, b), ("inicio_de_B", b["segs"][0], a["segs"][-1], b, a))
        for lado, curto, vizinho, dono, destino in casos:
            if len(dono["segs"]) < 2 or len(curto["rows"]) > k or np.isnan(curto["nivel"]):
                continue
            borda = vizinho["rows"][0] if lado == "fim_de_A" else vizinho["rows"][-1]
            if not borda.confiavel:
                continue
            rel = abs(curto["nivel"] / borda.media - 1)
            if rel > TOL_MATCH:
                continue
            ref = dono["segs"][-2] if lado == "fim_de_A" else dono["segs"][1]
            out.append({"lado": lado, "rows": curto["rows"], "serie_registrada": dono["serie"], "serie_proposta": destino["serie"],
                        "condicao_proposta": condicao_modal(vizinho), "nivel_bloco_proprio": ref["nivel"],
                        "nivel_bloco_proposto": borda.media, "dif_proposto": rel,
                        "dif_proprio": abs(curto["nivel"] / ref["nivel"] - 1) if ref["nivel"] == ref["nivel"] else np.nan,
                        "forca": "forte" if rel <= TOL_FORTE else "moderada"})
    return out


def nulo(corridas, k, rng):
    n = []
    for _ in range(N_PERM):
        p = [corridas[i] for i in rng.permutation(len(corridas))]
        n.append(len(propostas(p, k)))
    return np.array(n)


def premissa(d):
    par = d.assign(prev=d.groupby("numero_serie")["media"].shift(), prevc=d.groupby("numero_serie")["condicao"].shift(),
                   prevok=d.groupby("numero_serie")["confiavel"].shift())
    par = par[(par.condicao == par.prevc) & par.confiavel & (par.prevok == True)].copy()  # noqa: E712
    par["salto"] = (par.media / par.prev - 1).abs() > TOL_SALTO
    return par.groupby("condicao")["salto"].agg(pares="size", saltos="sum", frac="mean").round(3)


def tabela_saida(obs):
    linhas = []
    for o in obs:
        for r in o["rows"]:
            linhas.append({"experimento_idx": r.experimento_idx, "serie_registrada": o["serie_registrada"],
                           "condicao_registrada": r.condicao, "serie_proposta": o["serie_proposta"],
                           "condicao_proposta": o["condicao_proposta"], "condicao_muda": r.condicao != o["condicao_proposta"],
                           "lado": o["lado"], "nivel_ciclo": r.media, "nivel_bloco_proprio": o["nivel_bloco_proprio"],
                           "nivel_bloco_proposto": o["nivel_bloco_proposto"], "dif_proposto": o["dif_proposto"],
                           "dif_proprio": o["dif_proprio"], "forca": o["forca"]})
    return pd.DataFrame(linhas).sort_values("experimento_idx")


def fronteiras_f02(corridas, desloc):
    for a, b in zip(corridas[:-1], corridas[1:]):
        ra, rb = a["segs"][-1]["rows"][-1], b["segs"][0]["rows"][0]
        if a["serie"] != b["serie"] and ra.condicao == rb.condicao == "F02":
            marca = "DESLOCADO" if ra.experimento_idx in desloc or rb.experimento_idx in desloc else "sem proposta"
            print(ra.experimento_idx, a["serie"], round(ra.media, 4), "| prox", rb.experimento_idx, b["serie"], round(rb.media, 4), marca)


if __name__ == "__main__":
    d = carregar()
    corridas = montar_corridas(d)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)

    print("PREMISSA (saltos > 2% entre ciclos consecutivos da mesma unidade e condicao)")
    print(premissa(d).to_string())

    print(f"\nPROPOSTAS (corridas={len(corridas)}); nulo = ordem das corridas embaralhada, {N_PERM} permutacoes")
    for k in (1, 2, 3):
        obs = propostas(corridas, k)
        nl = nulo(corridas, k, rng)
        forte = sum(o["forca"] == "forte" for o in obs)
        print(f"k<={k}: observadas {len(obs)} (fortes {forte}); nulo media {nl.mean():.2f} p95 {np.percentile(nl, 95):.0f} "
              f"max {nl.max()}; p={(np.sum(nl >= len(obs)) + 1) / (N_PERM + 1):.4f}")

    out = tabela_saida(propostas(corridas, K_CURTO))
    out.to_csv(DADOS / "pbit_auditoria_continuidade.csv", index=False, lineterminator="\n")
    print()
    print(out.round(4).to_string(index=False))
    print("\npor condicao registrada:", out.condicao_registrada.value_counts().to_dict(), "| condicao muda:", int(out.condicao_muda.sum()))

    print("\nFRONTEIRAS F02|F02")
    fronteiras_f02(corridas, set(out.experimento_idx))

    f2 = d[d.condicao == "F02"].groupby("numero_serie").size()
    aj = out[out.condicao_registrada == "F02"]
    f2c = f2.sub(aj.groupby("serie_registrada").size(), fill_value=0).add(aj.groupby("serie_proposta").size(), fill_value=0)
    print("\nciclos F02 por unidade: registrado e apos reatribuicao proposta")
    print(pd.DataFrame({"registrado": f2, "proposto": f2c}).fillna(0).astype(int).T.to_string())
