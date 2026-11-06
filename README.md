# Power-on built-in test of airborne actuators: leakage, label audit and frequency descriptors

Reproducibility package for the article *From Recognising the Unit to Classifying its Condition: Leakage, Label Audit and Frequency Descriptors in Power-On Built-In Test of Airborne Actuators* (provisional title, under preparation), by Antonio C. P. Mesquita and Guilherme C. Rocha, Aerospace Logistics Engineering Laboratory (AeroLogLab), Instituto Tecnológico de Aeronáutica (ITA), São José dos Campos, Brazil.

The package contains the anonymised data, the analysis code, the acceptance criteria and the unit tests. Manuscript sources are not included.

## Contents

| Folder | Content |
|---|---|
| `dados/` | Input data, audited versions and audit logs |
| `scripts/` | Audit, experiment and verification scripts |
| `tests/` | Unit tests |
| `resultados/` | Published result tables, for comparison with a new run |
| `docs/acceptance_criteria.md` | Acceptance criteria fixed before each experiment, and what was amended afterwards |

Older scripts and file names are in Portuguese (`dados` means data, `resultados` means results); newer code is in English.

## Data

The data cover power-on built-in test (PBIT) cycles of 123 samples from 65 serial-numbered airborne actuators, in eight condition classes. System type and operator identity are anonymised, and serial numbers are recoded (`SN 001`, ...). The physical quantity, the engineering unit and the sampling rate were not disclosed. Numerical results do not depend on the sampling rate.

| File | Description |
|---|---|
| `PBIT_atuador_DESCARAC_r1_long.csv` | Input: raw signals in long format, 1,490 cycles |
| `pbit_features_completo.csv` | Input: 18 statistics per cycle, 1,486 cycles, as received |
| `pbit_features_limpo.csv`, `pbit_exclusoes.csv` | 11 cycles with duplicated feature vectors removed, with the log |
| `pbit_auditoria_continuidade.csv` | Serial-number candidates from the continuity audit |
| `pbit_features_corrigido.csv`, `pbit_reatribuicoes.csv` | 23 strong serial-number corrections applied, with the log |
| `pbit_features_corrigido_v2.csv`, `pbit_reclassificacoes.csv`, `pbit_quarentena.csv` | First condition corrections and three quarantined cycles |
| `pbit_features_corrigido_v3.csv`, `pbit_reclassificacoes_v3.csv` | **Version used in all experiments**: 1,472 cycles, with the remaining condition corrections |
| `pbit_spectral_features_v3.csv` | Frequency descriptors of version 3 |

Labels were filled in by hand and have no independent confirmation. Corrections rest on rules defined by the authors and logged cycle by cycle. The severity categories in the article are the authors' working mapping informed by ARP4761A, not an aircraft-level hazard classification.

## Environment

The analyses were run with Python 3.14.0 on Windows 11.

```
pip install -r requirements.txt
```

## Reproducing the analyses

Run everything, from the repository root:

```
bash scripts/run_all.sh
```

This takes a few hours. It first rebuilds the audited data from the two input files, then runs the experiments in this order:

| Script (run from `scripts/`) | Purpose |
|---|---|
| `limpar_dados.py`, `auditar_continuidade.py`, `aplicar_reatribuicoes.py`, `gerar_v2.py`, `gerar_v3.py`, `build_spectral_features.py` | Data audit, up to version 3 and its frequency descriptors |
| `experimentos_base.py` | Baselines and splitting strategies (E0 to E3b) |
| `experimentos_controles_hierarquia.py`, `calcular_cmr.py` | Leakage controls, hierarchy and critical miss rate |
| `experimento_s2_ablacao.py`, `analise_s2_regra_nivel.py` | Removal of absolute level; exploratory detector relative to each unit's baseline |
| `train_binary_model.py` | Normal-versus-failure model, with all failures |
| `train_binary_model.py --no-f02`, `train_binary_model.py --no-f02 --spectral` | The same without F02, without and with frequency descriptors |
| `analyze_within_unit.py`, `compare_spectral_gain.py` | Comparison within units; paired gain from frequency descriptors |
| `analyze_f02_vs_n01.py`, `analyze_f02_100hz.py` | F02 against N01; the second assumes 100 Hz |
| `evaluate_generalization.py`, `evaluate_label_noise.py` | Failure modes absent from training, training on N01 only, and label errors |

To check determinism, `bash scripts/check_reproducibility.sh` runs the model scripts twice and compares their CSV and TXT outputs byte by byte. In our run, the 38 files compared were identical. `experimentos_base.py`, `experimentos_controles_hierarquia.py`, `experimento_s2_ablacao.py` and `analyze_f02_100hz.py` were not part of that comparison.

For a quick test, `train_binary_model.py --rapido` writes to `resultados_teste/`.

## Tests

```
python -m unittest discover tests
```

## Scope of the evidence

All data were examined during the audit and there is no external cohort, so the results are internal evidence. F02 was removed from the main model after the results were seen, on the assumption that visual inspection intercepts it; both results are reported. See `docs/acceptance_criteria.md`.

## Citation and contact

Antonio C. P. Mesquita, celio@ita.br. The full citation will be added on publication.
