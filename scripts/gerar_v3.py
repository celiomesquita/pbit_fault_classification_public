"""Gera a versao 3: reclassifica como F06 os ciclos que atendem a regra do autor (corrente <= 0.05 o ciclo todo, ou platô
seguido de queda a zero). Entrada: dados/pbit_features_corrigido_v2.csv.
Saidas: dados/pbit_features_corrigido_v3.csv e dados/pbit_reclassificacoes_v3.csv."""
from pathlib import Path

import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"
LIMITE = 0.05
ALVOS = {20: "F01", 170: "F02", 1372: "F03", 1378: "F03", 1379: "F03"}

d = pd.read_csv(DADOS / "pbit_features_corrigido_v2.csv", float_precision="round_trip")
idx = d.set_index("experimento_idx")
assert (idx.loc[list(ALVOS), "condicao"] == pd.Series(ALVOS)).all()
assert (idx.loc[list(ALVOS), "minimo"] <= LIMITE).all()

def padrao(r):
    return "corrente <= 0.05 o ciclo todo" if r.maximo <= LIMITE else "platô e queda a zero (minimo <= 0.05)"

rec = pd.DataFrame([{"experimento_idx": i, "numero_serie": idx.loc[i, "numero_serie"], "condicao_anterior": a, "condicao_nova": "F06",
                     "padrao": padrao(idx.loc[i]), "valor_inicial": idx.loc[i, "valor_inicial"], "minimo": idx.loc[i, "minimo"],
                     "maximo": idx.loc[i, "maximo"], "valor_final": idx.loc[i, "valor_final"]} for i, a in ALVOS.items()])

v3 = d.copy()
v3.loc[v3["experimento_idx"].isin(ALVOS), "condicao"] = "F06"
v3.to_csv(DADOS / "pbit_features_corrigido_v3.csv", index=False, lineterminator="\n")
rec.to_csv(DADOS / "pbit_reclassificacoes_v3.csv", index=False, lineterminator="\n")
