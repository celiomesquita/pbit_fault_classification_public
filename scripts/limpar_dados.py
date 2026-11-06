"""Gera dados/pbit_features_limpo.csv e dados/pbit_exclusoes.csv a partir do CSV original."""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"

d = pd.read_csv(DADOS / "pbit_features_completo.csv", float_precision="round_trip")
features = list(d.columns[3:])

exclusoes = []
for _, g in d[d.duplicated(subset=features, keep=False)].groupby(features, sort=False):
    conflito = g["condicao"].nunique() > 1
    excluidas = g if conflito else g.iloc[1:]
    motivo = "rotulo_conflitante" if conflito else "duplicata_mesmo_rotulo"
    for _, r in excluidas.iterrows():
        outros = g.loc[g["experimento_idx"] != r["experimento_idx"], "experimento_idx"]
        exclusoes.append({
            "experimento_idx": r["experimento_idx"],
            "numero_serie": r["numero_serie"],
            "condicao": r["condicao"],
            "motivo": motivo,
            "experimento_idx_relacionado": " ".join(map(str, outros)),
        })

log = pd.DataFrame(exclusoes).sort_values("experimento_idx")
limpo = d[~d["experimento_idx"].isin(log["experimento_idx"])]

limpo.to_csv(DADOS / "pbit_features_limpo.csv", index=False, lineterminator="\n")
log.to_csv(DADOS / "pbit_exclusoes.csv", index=False, lineterminator="\n")
