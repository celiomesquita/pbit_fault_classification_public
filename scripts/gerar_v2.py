"""Gera a versao 2 do CSV corrigido. Decisoes do autor: queda de corrente com retorno e queda a zero caracterizam F06
(Command transmission failure); corrente muito baixa e <= 0.05. Entrada: dados/pbit_features_corrigido.csv.
Saidas: dados/pbit_features_corrigido_v2.csv, dados/pbit_reclassificacoes.csv, dados/pbit_quarentena.csv."""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"

RECLASSIFICAR = {
    994: ("N01", "F06", "queda a ~0.01 e retorno a 2.29; vizinhos estaveis em 3.048"),
    997: ("N01", "F06", "queda a ~0.01 e retorno a 3.05; vizinhos estaveis em 3.048"),
    1135: ("N01", "F06", "platô em 1.76 e queda a 0.003 sem retorno"),
}
QUARENTENA = {
    917: ("F06", "corrente estavel e negativa (-3.66); nao atende a F06 (<=0.05 ou queda a zero); provavel polaridade invertida, nao verificavel"),
    950: ("F06", "corrente estavel em 0.64 (>0.05) sem queda a zero; nao atende a F06 nem a N02 (nivel da unidade 3.3)"),
    951: ("F06", "corrente estavel em 0.64 (>0.05) sem queda a zero; nao atende a F06 nem a N02 (nivel da unidade 3.3)"),
}

d = pd.read_csv(DADOS / "pbit_features_corrigido.csv", float_precision="round_trip")
idx = d.set_index("experimento_idx")
for i, (antes, _, _) in RECLASSIFICAR.items():
    assert idx.loc[i, "condicao"] == antes
for i, (antes, _) in QUARENTENA.items():
    assert idx.loc[i, "condicao"] == antes

v2 = d.copy()
for i, (_, depois, _) in RECLASSIFICAR.items():
    v2.loc[v2["experimento_idx"] == i, "condicao"] = depois
quar = v2[v2["experimento_idx"].isin(QUARENTENA)].copy()
v2 = v2[~v2["experimento_idx"].isin(QUARENTENA)]

quar["motivo"] = quar["experimento_idx"].map({i: m for i, (_, m) in QUARENTENA.items()})
rec = pd.DataFrame([{"experimento_idx": i, "numero_serie": idx.loc[i, "numero_serie"], "condicao_anterior": a, "condicao_nova": b, "motivo": m}
                    for i, (a, b, m) in RECLASSIFICAR.items()])

v2.to_csv(DADOS / "pbit_features_corrigido_v2.csv", index=False, lineterminator="\n")
rec.to_csv(DADOS / "pbit_reclassificacoes.csv", index=False, lineterminator="\n")
quar.to_csv(DADOS / "pbit_quarentena.csv", index=False, lineterminator="\n")
