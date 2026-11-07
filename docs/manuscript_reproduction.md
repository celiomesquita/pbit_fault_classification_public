# Where each result of the article comes from

Every table, figure and number in the article is computed by a script in `scripts/` from the data in `dados/`. This guide maps each one to the script that produces it and the file in `resultados/` where it can be read. Run `bash scripts/run_all.sh` first (a few hours), or a single script from `scripts/` after its prerequisites.

Conventions: intervals are 95% unit-bootstrap intervals (`ci_low`, `ci_high`, or `ic95_inf`, `ic95_sup` in older files); `_10` and `_20` denote the 10% and 20% false-alarm targets; `procedure` is the configuration chosen inside each outer fold. Older files use Portuguese names: `estrategia` (splitting strategy), `modelo` (model), `metrica` (metric), `estimativa` (estimate), `recall_falha` (failure recall), `falso_alarme_N01` (false alarm on N01).

Large prediction files (`*predic*.csv`) and `c1_permutacoes.csv` are not versioned; the scripts regenerate them.

## Data and exploratory analysis

| Article item | Script | File and field |
|---|---|---|
| Class table: cycles and units per condition | (direct count) | `dados/pbit_features_v3.csv`, columns `condicao`, `numero_serie` |
| Cycles per unit (4 to 79, median 20), units with several conditions, N02 dominant-unit share | `exploratory_analysis.py` | `exploratory_summary.csv` |
| N01 median level per unit, within-unit coefficient of variation | `exploratory_analysis.py` | `exploratory_n01_by_unit.csv`; summary in `exploratory_summary.csv` |
| Low-frequency power share by unit and condition | `exploratory_analysis.py` | `exploratory_unit_condition.csv`; class medians in `exploratory_summary.csv` |
| Exploratory figure | `exploratory_analysis.py` | `figures/fig_exploratory_analysis.pdf` |
| Algebraic identities among features | `exploratory_analysis.py` | `exploratory_summary.csv`, `*_identity_max_error` |
| Fault level lower than N01 in the same unit (F01–F05) | `exploratory_analysis.py` | `exploratory_level_shift.csv`; counts in `exploratory_summary.csv` |
| Frequency descriptors | `build_spectral_features.py` | `dados/pbit_spectral_features_v3.csv` |

## RQ1: splitting strategy and leakage

| Article item | Script | File and field |
|---|---|---|
| Leakage table (E0–E3b, random forest and logistic regression) | `experimentos_base.py`, `calcular_cmr.py` | `metricas_ic95_por_sn.csv`; critical miss in `cmr_ic95_por_sn.csv` |
| E1 minus E3b differences | `experimentos_base.py` | `diferencas_vazamento_ic95.csv` |
| C1 block-label permutation (macro-recall 0.800 vs 0.795; E3b values and p) | `experimentos_controles_hierarquia.py` | `c1_resumo.csv` |
| C2 serial-number identification (0.81 vs 0.054) | `experimentos_controles_hierarquia.py` | `c2_identificacao_sn.csv` |
| Removal of absolute level (feature table, identification 0.806 → 0.216, AUC) | `experimento_s2_ablacao.py` | `s2_identificacao_sn.csv`, `s2_auc_pontos_de_operacao.csv`, `s2_metricas_ic95_por_sn.csv`, `s2_diferencas_ic95.csv` |

## RQ2 and RQ3: classes and hierarchy

| Article item | Script | File and field |
|---|---|---|
| Per-class recall under E3b | `experimentos_base.py` | `metricas_ic95_por_sn.csv`, rows `E3b,RF,recall_<class>` |
| Classes exceeding the shuffled-label control (Bonferroni over eight classes) | `experimentos_controles_hierarquia.py` | `c1_resumo.csv`, rows `E3b,recall_<class>`, column `p` |
| Share of F02 classified as N01 | `experimentos_base.py` | `confusao_media.csv`, row `E3b,RF,F02` |
| F04 recall by unit | `experimentos_base.py` | `recall_por_sn.csv` |
| F04 against F05: exact test, level relative to N01, file position | `analyze_f04_f05.py` | `f04_f05_separability.csv`, `f04_f05_level_shift.csv`, `f04_f05_position.csv` |
| Hierarchy table and paired differences | `experimentos_controles_hierarquia.py`, `calcular_cmr.py` | `e4_metricas_ic95_por_sn.csv`, `e4_auc_falha_vs_n01.csv`, `e4_diferencas_ic95.csv`, `e4_diferencas_pontos_de_operacao.csv`, `cmr_ic95_por_sn.csv` |
| Recall of F02 inside the abnormal group | `experimentos_controles_hierarquia.py` | `e4_metricas_por_nivel.csv` |

## RQ4: normal-versus-failure decision

| Article item | Script | File and field |
|---|---|---|
| Binary table (three populations, three thresholds) | `train_binary_model.py` (no flag; `--no-f02`; `--no-f02 --spectral`) | `binary_results.csv`, `binary_no_f02_results.csv`, `binary_no_f02_spectral_results.csv`, rows `configuration == procedure` |
| Independence and real-signal controls, random-split AUC | `train_binary_model.py` | `binary*_control.csv` (`real_grouped_auc`, `p_value`, `leaky_auc`), `binary*_unit_identification.csv` |
| Alarm rate by condition (10% and 20%) | `train_binary_model.py`, `evaluate_decision_analysis.py` | `binary*_alarm_rate_by_condition_at_10.csv`, `binary_no_f02_spectral_alarm_rate_by_condition_at_20.csv` |
| F02 against N01, including bands assuming 100 Hz | `analyze_f02_vs_n01.py`, `analyze_f02_100hz.py` | `f02_vs_n01.csv`, `f02_100hz.csv` |
| Paired gain from frequency descriptors | `compare_spectral_gain.py` | `spectral_gain.csv` |
| Cost ratio r* and cost table | `evaluate_decision_analysis.py` | `cost_rstar.csv`, `cost_curve.csv`, `cost_inner_optimal.csv` |
| Item-level decisions | `evaluate_decision_analysis.py` | `item_level.csv`; item definitions in `item_blocks.csv` |
| Model and feature set chosen in each fold; final model weights | `train_binary_model.py --no-f02 --spectral` | `binary_no_f02_spectral_procedure_choices.csv`, `binary_no_f02_spectral_final_model.txt` |
| Within-unit AUC (10 units with sustained failures), logistic regression on E | `compare_spectral_gain.py` | `binary_no_f02_spectral_within_unit.csv` |
| N01 against N02 | `train_binary_model.py --no-f02 --spectral` | `binary_no_f02_spectral_second_stage_n01_n02.csv` |
| XGBoost comparison and text | `compare_xgboost.py` | `xgboost_results.csv`, `xgboost_paired_differences.csv`, `xgboost_expanded_procedure_choices.csv`, `xgboost_importance.csv` |

## RQ5: robustness

| Article item | Script | File and field |
|---|---|---|
| Unseen failure modes (procedure; fixed logistic regression as sensitivity) | `evaluate_robustness_procedure.py`; `evaluate_generalization.py` | `procedure_unseen_mode.csv`; `unseen_mode.csv` |
| Detector trained on N01 only | `evaluate_generalization.py` | `n01_only.csv`, `n01_only_alarm_rate_by_condition_at_20.csv` |
| Training-label errors (procedure; fixed logistic regression; N01 only) | `evaluate_robustness_procedure.py`; `evaluate_label_noise.py` | `procedure_label_noise.csv`; `label_noise.csv` |
| F02 escaping inspection | `evaluate_robustness_procedure.py`; `evaluate_generalization.py` | `procedure_f02_escape.csv`; `f02_escape.csv` |

## Exploratory unit-specific reference

| Article item | Script | File and field |
|---|---|---|
| Detector relative to each unit's N01 baseline (AUC 0.961; 0.687 without labels) | `analise_s2_regra_nivel.py` | `s2_regra_nivel.csv` |

## Reproducibility check

`bash scripts/check_reproducibility.sh` runs the model scripts twice and compares their outputs byte by byte; the outcome is written to `resultados/reproducibility.txt`. Results were produced with the versions in `requirements.txt` on Windows 11; other platforms or library versions may change the last digits of tree-based models.
