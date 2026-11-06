"""Calcula o CMR (ciclos Hazardous/Catastrophic classificados como No Safety Effect) a partir das predicoes salvas.
Saida: resultados/cmr_ic95_por_sn.csv, com IC de 95% por bootstrap de SN."""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / "resultados"
CRITICAS, SEM_EFEITO = ["F03", "F06"], ["N01", "N02"]

base = pd.read_csv(R / "predicoes_fora_da_amostra.csv").rename(columns={"estrategia": "config"})
e4 = pd.read_csv(R / "e4_predicoes.csv")
e0 = pd.read_csv(R / "predicoes_baselines.csv").rename(columns={"estrategia": "config"})

unidades = sorted(base["numero_serie"].unique())
pos = {u: i for i, u in enumerate(unidades)}
pesos = np.random.default_rng(2026).multinomial(len(unidades), np.full(len(unidades), 1 / len(unidades)), size=2000)

linhas = []
for origem in (base, e4, e0):
    for (config, modelo), x in origem.groupby(["config", "modelo"]):
        s = x[x["condicao"].isin(CRITICAS)]
        i = s["numero_serie"].map(pos).values
        perdas, n = np.zeros(len(unidades)), np.zeros(len(unidades))
        np.add.at(perdas, i, s["pred"].isin(SEM_EFEITO).values.astype(float))
        np.add.at(n, i, 1.0)
        v = (pesos @ perdas) / (pesos @ n)
        linhas.append({"config": config, "modelo": modelo, "cmr": perdas.sum() / n.sum(),
                       "ic_inf": np.nanquantile(v, 0.025), "ic_sup": np.nanquantile(v, 0.975)})
pd.DataFrame(linhas).to_csv(R / "cmr_ic95_por_sn.csv", index=False, lineterminator="\n")
