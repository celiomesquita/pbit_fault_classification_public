#!/usr/bin/env bash
# Run every experiment from dataset version 3, in dependency order.
# Usage: bash scripts/run_all.sh. Set PYTHON if the interpreter is not called python.
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

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

# Decision analysis and robustness of the selected procedure (require the predictions of the model without F02)
"$PYTHON" evaluate_decision_analysis.py
"$PYTHON" evaluate_robustness_procedure.py

# Post-result XGBoost extension (requires the predictions of the model without F02)
"$PYTHON" compare_xgboost.py
