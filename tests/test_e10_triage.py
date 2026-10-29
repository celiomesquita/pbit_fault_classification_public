"""Decision curves checked against exhaustive cycle-level evaluation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbit.triage_policy import CLASSES, curve, evaluate, predict, select_curve


def test_cumulative_curves_match_exhaustive_decisions_with_ties():
    rng = np.random.default_rng(20261028)
    y = np.tile(CLASSES, 4)
    probability = rng.integers(0, 4, (len(y), 8)).astype(float) + 1
    probability /= probability.sum(axis=1, keepdims=True)
    for promotion in (False, True):
        values = curve(y, probability, promotion)
        for i, threshold in enumerate(values["thresholds"]):
            detected = predict(probability, threshold, promotion)
            metrics = evaluate(y, detected)
            assert np.isclose(values["FA0"][i], metrics["FA0"])
            for j, cls in enumerate(CLASSES[2:7]):
                assert np.isclose(values["recalls"][i, j], metrics[f"detect_{cls}"])


def test_exact_identity_and_f06_are_not_primary_detection_requirements():
    y = CLASSES.copy()
    prediction = np.array(["N02", "N01", "F03", "F03", "F01", "F03", "F03", "N01"])
    metrics = evaluate(y, prediction)
    assert metrics["min_recall"] == 1
    assert metrics["detect_F06"] == 0
    assert metrics["exact_F01"] == 0
    assert metrics["detect_F01"] == 1
    assert metrics["CMR"] == .5


def test_normal_fault_score_can_remove_original_alarms():
    y = CLASSES.copy()
    probability = np.zeros((8, 8))
    probability[:2] = [.3, .2, .4, .1, 0, 0, 0, 0]  # Original argmax is a fault.
    probability[2:, 0] = .1
    probability[2:, 2] = .9
    complete = select_curve(curve(y, probability), .90, .40)
    unilateral = select_curve(curve(y, probability, True), .90, .40)
    assert complete["feasible"] and complete["FA0"] == 0
    assert unilateral["status"] == "alarm_budget_infeasible"


def test_absent_primary_class_is_undefined_not_success():
    y = CLASSES[CLASSES != "F05"]
    probability = np.full((len(y), 8), 1 / 8)
    selection = select_curve(curve(y, probability))
    assert selection["status"] == "endpoint_undefined"
    assert not selection["feasible"]


def test_never_alarm_is_not_a_successful_recall_fallback():
    y = CLASSES.copy()
    probability = np.full((len(y), 8), 1 / 8)
    selection = select_curve(curve(y, probability), .90, .40)
    assert selection["status"] == "recall_infeasible"
    assert selection["min_recall"] == 0
    assert selection["FA0"] == 0


def test_selection_matches_exhaustive_per_cycle_objective_for_all_scenarios():
    rng = np.random.default_rng(741)
    y = np.tile(CLASSES, 20)
    p = rng.dirichlet(np.ones(8), size=len(y))
    # Give faulty cycles higher fault mass, retaining heterogeneous class responses.
    p[2::8, 2:] *= 3
    p /= p.sum(axis=1, keepdims=True)
    for promotion in (False, True):
        candidates = np.r_[np.inf, np.unique(p[:, 2:].sum(axis=1))[::-1], 0.]
        measured = [(t, evaluate(y, predict(p, t, promotion))) for t in candidates]
        for target in (.85, .90, .95):
            for budget in (.30, .40, .50):
                selection = select_curve(curve(y, p, promotion), target, budget)
                within = [(t, m) for t, m in measured if m["FA0"] <= budget + 1e-12]
                feasible = [(t, m) for t, m in within if m["min_recall"] >= target-1e-12]
                if feasible:
                    expected = min(feasible, key=lambda x: (x[1]["FA0"], -x[1]["min_recall"], -x[1]["mean_recall"], -x[0]))
                elif within:
                    expected = min(within, key=lambda x: (-x[1]["min_recall"], x[1]["FA0"], -x[1]["mean_recall"], -x[0]))
                else:
                    expected = measured[0]
                assert selection["threshold"] == expected[0]
                assert selection["feasible"] == bool(feasible)
