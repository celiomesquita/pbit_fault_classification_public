"""Aplica as reatribuicoes de numero de serie classificadas como fortes em pbit_auditoria_continuidade.csv.
Entrada: dados/pbit_features_limpo.csv. Saidas: dados/pbit_features_corrigido.csv e dados/pbit_reatribuicoes.csv."""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"

d = pd.read_csv(DADOS / "pbit_features_limpo.csv", float_precision="round_trip")
aud = pd.read_csv(DADOS / "pbit_auditoria_continuidade.csv")
fortes = aud[aud["forca"] == "forte"]

mapa = fortes.set_index("experimento_idx")["serie_proposta"]
assert mapa.index.is_unique
corrigido = d.copy()
corrigido["numero_serie"] = corrigido["experimento_idx"].map(mapa).fillna(corrigido["numero_serie"])

log = fortes[["experimento_idx", "condicao_registrada", "serie_registrada", "serie_proposta", "nivel_ciclo",
              "nivel_bloco_proprio", "nivel_bloco_proposto", "dif_proposto", "dif_proprio"]]
assert (d.set_index("experimento_idx").loc[log["experimento_idx"], "numero_serie"].values == log["serie_registrada"].values).all()

corrigido.to_csv(DADOS / "pbit_features_corrigido.csv", index=False, lineterminator="\n")
log.to_csv(DADOS / "pbit_reatribuicoes.csv", index=False, lineterminator="\n")
