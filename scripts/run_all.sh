#!/usr/bin/env bash
# Rebuild the audited data from the two input files and run every experiment, in dependency order.
# Usage: bash scripts/run_all.sh. Set PYTHON if the interpreter is not called python.
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

# Data audit
"$PYTHON" limpar_dados.py
"$PYTHON" auditar_continuidade.py
"$PYTHON" aplicar_reatribuicoes.py
"$PYTHON" gerar_v2.py
"$PYTHON" gerar_v3.py
"$PYTHON" build_spectral_features.py

# Leakage, controls, hierarchy and feature removal
"$PYTHON" experimentos_base.py
"$PYTHON" experimentos_controles_hierarquia.py
"$PYTHON" calcular_cmr.py
"$PYTHON" experimento_s2_ablacao.py
"$PYTHON" analise_s2_regra_nivel.py

# Normal-versus-failure models
"$PYTHON" train_binary_model.py
"$PYTHON" train_binary_model.py --no-f02
"$PYTHON" train_binary_model.py --no-f02 --spectral
"$PYTHON" analyze_within_unit.py
"$PYTHON" compare_spectral_gain.py

# F02, generalisation and label errors
"$PYTHON" analyze_f02_vs_n01.py
"$PYTHON" analyze_f02_100hz.py
"$PYTHON" evaluate_generalization.py
"$PYTHON" evaluate_label_noise.py
