# Power-on built-in test of airborne actuators: leakage and frequency descriptors

Reproducibility package for the article *From Recognising the Unit to Classifying its Condition: Leakage and Frequency Descriptors in Power-On Built-In Test of Airborne Actuators* (provisional title, under preparation), by Antonio C. P. Mesquita and Guilherme C. Rocha, Aerospace Logistics Engineering Laboratory (AeroLogLab), Instituto Tecnológico de Aeronáutica (ITA), São José dos Campos, Brazil.

The package contains the anonymised dataset, the analysis code, the acceptance criteria, the unit tests, the published result tables and exploratory figure, and a self-contained dashboard. Every table, figure and number in the article can be traced to a script and a result file (`docs/manuscript_reproduction.md`). Manuscript sources are not included.

## Contents

| Folder | Content |
|---|---|
| `dados/` | Dataset version 3 |
| `scripts/` | Experiment and verification scripts |
| `tests/` | Unit tests |
| `dashboard/index.html` | Offline interactive exploration and grouped-validation evidence |
| `resultados/` | Published result tables, for comparison with a new run |
| `docs/acceptance_criteria.md` | Acceptance criteria fixed before each experiment, and what was amended afterwards |
| `docs/manuscript_reproduction.md` | Which script and result file produce each table, figure and number of the article |
| `figures/` | Exploratory figure, for comparison with a new run |

Older scripts and file names are in Portuguese (`dados` means data, `resultados` means results); newer code is in English.

## Data

Dataset version 3 has 1,472 power-on built-in test (PBIT) cycles of 123 samples from 65 serial-numbered airborne actuators, in eight condition classes. It is the starting point of this work. System type and operator identity are anonymised, and serial numbers are recoded (`SN 001`, ...). The physical quantity, the engineering unit and the sampling rate were not disclosed. Numerical results do not depend on the sampling rate.

| File | Description |
|---|---|
| `pbit_features_v3.csv` | Cycle identifier, serial number, condition and 18 statistics per cycle |
| `pbit_signals_v3.csv` | The 123 samples of each cycle, in long format; a few samples are missing |
| `pbit_spectral_features_v3.csv` | Frequency descriptors derived from the signals |

Labels have no independent confirmation. F06 is defined by a pattern of the signal, so results for F06 are partly circular. The severity categories in the article are the authors' working mapping informed by ARP4761A, not an aircraft-level hazard classification.

## Environment

The analyses were run with Python 3.14.0 on Windows 11.

```
pip install -r requirements.txt
```

## Reproducing the analyses

Every table, figure and number in the article can be regenerated from this package; `docs/manuscript_reproduction.md` maps each of them to its script and result file.

Run everything, from the repository root:

```
bash scripts/run_all.sh
```

This takes a few hours. The scripts run in this order, and several depend on earlier outputs: `evaluate_decision_analysis.py`, `evaluate_robustness_procedure.py` and `compare_xgboost.py` read the held-out predictions of `train_binary_model.py --no-f02 --spectral`, which are regenerated but not versioned.

| Script (run from `scripts/`) | Purpose |
|---|---|
| `build_spectral_features.py` | Frequency descriptors |
| `exploratory_analysis.py` | Unit coverage, signal level, low-frequency content, feature identities and the exploratory figure |
| `experimentos_base.py` | Baselines and splitting strategies (E0 to E3b) |
| `experimentos_controles_hierarquia.py`, `calcular_cmr.py` | Leakage controls, hierarchy and critical miss rate |
| `experimento_s2_ablacao.py`, `analise_s2_regra_nivel.py` | Removal of absolute level; exploratory detector relative to each unit's baseline |
| `analyze_f04_f05.py` | Exact test of F04 against F05 over all assignments of their eight units |
| `train_binary_model.py` | Normal-versus-failure model, with all failures |
| `train_binary_model.py --no-f02`, `train_binary_model.py --no-f02 --spectral` | The same without F02, without and with frequency descriptors |
| `analyze_within_unit.py`, `compare_spectral_gain.py` | Comparison within units; paired gain from frequency descriptors |
| `analyze_f02_vs_n01.py`, `analyze_f02_100hz.py` | F02 against N01; the second assumes 100 Hz |
| `evaluate_generalization.py`, `evaluate_label_noise.py` | Failure modes absent from training, training on N01 only, and label errors |
| `evaluate_decision_analysis.py` | Cost of the 10% and 20% false-alarm targets, cost-selected thresholds, item-level decisions and alarm rate by condition at 20% |
| `evaluate_robustness_procedure.py` | Unseen failure modes, label errors and F02 escape for the procedure selected within each outer fold (set E) |
| `compare_xgboost.py` | Post-result extension: XGBoost added to the comparison without F02, alone and within the within-fold selection; refits logistic regression to verify the saved predictions before reusing them |

To check determinism, `bash scripts/check_reproducibility.sh` runs the model scripts twice and compares their CSV and TXT outputs byte by byte. In our run, the 38 files compared were identical. `experimentos_base.py`, `experimentos_controles_hierarquia.py`, `experimento_s2_ablacao.py`, `analyze_f02_100hz.py`, `exploratory_analysis.py`, `analyze_f04_f05.py`, `evaluate_decision_analysis.py`, `evaluate_robustness_procedure.py` and `compare_xgboost.py` were not part of that comparison; `tests/test_xgboost_comparison.py` checks that XGBoost fits are deterministic. Results were produced with the versions in `requirements.txt` on Windows 11; other platforms or library versions may change the last digits of tree-based models.

For a quick test, `train_binary_model.py --rapido` writes to `resultados_teste/`.

## Dashboard

Download [`dashboard/index.html`](dashboard/index.html) using GitHub's **Download raw file** button, then open the saved file in a browser. This single file includes the signals, features, displayed held-out predictions, styles and code. No installation, server, account or internet connection is required. GitHub's preview displays the HTML source; open the downloaded file to use the dashboard.

The interface is in Brazilian Portuguese. It provides:

- Filters by condition and unit, signal and feature inspection, and CSV export of the selection.
- A condition filter in **Detalhe do ciclo** that narrows the cycle selector within the general filters, without changing the exploration table.
- Held-out scores, fold-specific thresholds and decisions for the internally selected procedure or logistic regression with feature set E, at false-alarm targets of 5%, 10% and 20%.
- A **Método e fontes** tab recording source paths and SHA-256 hashes. Large prediction outputs referred to there can be regenerated by the analysis scripts; they are not all versioned as separate CSV files.

F02 remains visible in exploration but is excluded from the displayed binary evaluation. The dashboard does not classify new readings; alarm counts across repetitions are not calibrated failure probabilities. It is a research inspection tool, not an operational release system. The public dashboard is distributed as the generated, self-contained HTML file.

## Tests

```
python -m unittest discover tests
```

The tests cover the binary-model pipeline (grouped folds, absence of unit overlap, thresholds from inner validation only), the descriptive figures cited in the exploratory analysis, and the XGBoost comparison (training-only class weights, deterministic fits, paired alignment).

## Scope of the evidence

The same data guided exploration and evaluation, and there is no external cohort, so the results are internal evidence. F02 was removed from the main model after the results were seen, on the assumption that visual inspection intercepts it; both results are reported. Without F02, no threshold meets the false-alarm and critical-miss criteria jointly; the 20% false-alarm target was adopted after the results, on a cost argument (`cost_rstar.csv`) that depends on a cost ratio not measured for any fleet. Item-level decisions approximate removal events by runs of consecutive cycles of one unit and condition, because the chronology of the identifiers is unconfirmed. N02 rests on two units and F04 and F05 on four each, with F04 and F05 on disjoint units and separate blocks of the file, so their separability is not demonstrated. Several analyses were fixed after seeing results and are listed, with their order, in `docs/acceptance_criteria.md`.

## Citation and contact

Antonio C. P. Mesquita, celio@ita.br. The full citation will be added on publication.
