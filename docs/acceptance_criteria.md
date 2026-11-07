# Acceptance criteria

Criteria were fixed before running each experiment, except where marked. Numerical limits were proposed by the analysts and are not validated operational requirements. All evaluations use units never seen in training. Intervals are 95% from a bootstrap of units.

Definitions: *failure recall*, share of F01 to F06 cycles not classified as N01 or N02; *false alarm*, share of N01 and N02 cycles classified as failure; *critical miss*, share of F03 and F06 cycles classified as N01 or N02.

## Normal-versus-failure model

| ID | Requirement | If it fails |
|---|---|---|
| C1 | Serial number is not an input; no unit appears in training and test of the same fold | Run is invalid |
| C2 | AUC above a control with labels shuffled between blocks of unit and condition (p <= 0.05), and lower interval limit above 0.50 | The model did not learn the condition |
| C3 | Failure recall of at least 0.80 with an actual false-alarm rate of at most 10% | State that these features do not support decision support; add no complexity |
| C4 | Critical misses of at most 0.05 at the same point | Do not recommend use as a safety barrier |
| C5 | Among candidates within 0.01 of the best inner AUC, keep the simplest: shallow tree, logistic regression, random forest, gradient boosting | Record why the simplest was rejected |
| C6 | Fixed seeds, passing unit tests, identical output on a repeated run | Phase not complete |

Model and feature set are chosen inside each outer fold, using inner validation only.

## Later experiments

| ID | Experiment | Requirement | If it fails |
|---|---|---|---|
| C7 | F02 against N01, relative and frequency descriptors | AUC of at least 0.75, lower limit above 0.50, p <= 0.05 | F02 stays outside the reach of PBIT with these descriptors |
| C8 | Frequency descriptors in the model without F02 | At least 0.03 of recall at the same point, or meeting C4 | Frequency stays out of the model |
| C9 | Failure mode absent from training (20% target) | For F01, F03, F04 and F05, recall of at least 0.80 and lower AUC limit above 0.50; F06 reported only | The model serves known modes only |
| C10 | Mahalanobis distance on N01 only (20% target), chosen before results | AUC of at least 0.85, recall of at least 0.80, critical misses of at most 0.05 | Supervised model stays as the main one |
| C11 | 5% of label errors, in training only (20% target) | At most 0.05 of recall lost and critical misses of at most 0.10 | State the sensitivity and the tolerated level |
| C12 | F02 against N01, bands in Hz assuming 100 Hz, 6 combinations | AUC of at least 0.75, lower limit above 0.50, p below 0.05/6 for logistic regression | No further variations on these data |

## Outcome

| ID | Outcome |
|---|---|
| C1, C2 | Met, with and without F02 |
| C3 | Not met with all failures (recall 0.514). Met without F02 at the default threshold, where C4 is not met |
| C4 | Not met with all failures (0.236). Without F02 and with frequency: 0.047 [0.012; 0.076] at the 10% target, with an actual false-alarm rate of 0.105; 0.020 [0.001; 0.045] at the 20% target, whose false-alarm rate of 0.199 exceeds the C3 ceiling. No threshold meets C3 and C4 jointly |
| C5 | Met |
| C6 | Met for the scripts compared, after a correction to the forest's parallel summation |
| C7, C12 | Not met |
| C8 | Met by the point estimate at the 10% and 20% targets (recall gain 0.051 and 0.035), not by the lower interval limit |
| C9 | Met by the point estimate for the procedure selected within each fold (recall 0.873 to 1.000); for F05 the interval reaches 0.745. Also met by the fixed logistic model |
| C10 | Met at the point estimate; the critical-miss interval reaches 0.083 |
| C11 | Met by the selected procedure (recall loss 0.008, critical misses 0.028) and by the fixed logistic model; not met by the N01-only detector |

## Amendments after seeing results

- **Removal of F02.** Decided after seeing that F02 caused the rejection under C3 and C4, on the assumption that visual inspection intercepts foreign objects. That assumption was not tested. Results with and without F02 are both reported.
- **Operating point.** The 20% target was adopted after results, because of the cost of a failure reaching stock. It does not meet the C3 ceiling.
- **Reading of C4.** The criteria did not state whether limits apply to point estimates or interval limits. After results, the upper limit of the 95% interval was adopted as the conservative reading; both are reported.
- **Cost and item-level analyses.** Fixed after results and before execution: the cost ratio above which the 20% target costs less than the 10% target (3.1 [1.3; 11.5]), thresholds minimising cost on inner validation, and item-level decisions (runs of consecutive cycles of one unit and condition; median and majority rules only). They justify, but do not validate, the adopted operating point.
- **Robustness of the selected procedure.** C9 and C11 were first evaluated with logistic regression fixed at C = 0.01. They were repeated with the procedure selected within each outer fold, restricted to set E for computational feasibility; the fixed-model results are kept as a sensitivity analysis.
- **XGBoost extension.** Added after results to the comparison without F02, with no acceptance criterion of its own: six configurations (maximum depth 2, 3 or 4; quadratic leaf-weight penalty 1 or 10; 200 trees; learning rate 0.05; no early stopping), the same outer and inner folds, and XGBoost ranked after the four original families under the unchanged C5 margin. Paired differences from logistic regression on E include zero for every metric, and the within-fold selection is unchanged in all 25 folds.
- **Earlier leakage and hierarchy experiments.** These used a pre-specified protocol with a relevance margin of 0.03, but not the table above.
