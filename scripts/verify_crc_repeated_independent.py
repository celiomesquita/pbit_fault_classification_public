"""Independent count-based audit of E6 decisions (also usable during a run)."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/crc_repeated_partitions"


def main(partial):
    manifests = sorted((OUT / "run").glob("*/manifest.json"))
    if not partial:
        assert len(manifests) == 600
    partitions = pd.read_csv(OUT / "partitions.csv")
    population = pd.read_csv(ROOT / "data/ress_revision/freeze_v1/features_primary.csv")
    critical = sorted(population.loc[population.Severity_Index.eq(3), "Serial_Number"].unique())
    other = sorted(set(population.Serial_Number) - set(critical))
    boundary_rounding = []
    observed_pairs = set()
    for path in manifests:
        directory = path.parent
        manifest = json.loads(path.read_text(encoding="utf-8"))
        n, rep = manifest["n_cal"], manifest["repetition"]
        assert (rep, n) not in observed_pairs
        observed_pairs.add((rep, n))
        roles = partitions[(partitions.n_cal == n) & (partitions.repetition == rep)]
        assert roles.Serial_Number.is_unique and len(roles) == 65
        rng = np.random.default_rng(20261011 + rep)
        cp, op = rng.permutation(critical).tolist(), rng.permutation(other).tolist()
        expected_roles = {"train": set(cp[2+n:] + op[10:]),
                          "calibration": set(cp[2:2+n]), "evaluation": set(cp[:2] + op[:10])}
        for role, units in expected_roles.items():
            assert set(roles.loc[roles.role.eq(role), "Serial_Number"]) == units
        train = population[population.Serial_Number.isin(roles.loc[roles.role.eq("train"), "Serial_Number"])]
        assert len(train) == manifest["train_cycles"]
        assert set(train.Fault_Mode) == set(manifest["model_classes"])
        assert train.loc[train.Severity_Index.eq(3), "Serial_Number"].nunique() == 14 - n
        counts = train.Fault_Mode.value_counts()
        assert manifest["synthetic_cycles"] == sum(max(0, 160-int(counts[c]))
                                                   for c in ["F04", "F05", "F06"] if c in counts)
        scores = pd.read_csv(directory / "scores.csv", float_precision="round_trip")
        for role in ["calibration", "evaluation"]:
            actual = scores[scores.role.eq(role)]
            expected = population[population.Serial_Number.isin(roles.loc[roles.role.eq(role), "Serial_Number"])]
            assert set(actual.Experiment_Index) == set(expected.Experiment_Index)
        cal = scores[scores.role.eq("calibration") & scores.Severity_Index.eq(3)]
        evaluation = scores[scores.role.eq("evaluation")]
        candidates = [np.inf] + sorted(set(cal.severity3_score), reverse=True) + [0.]
        exact_criteria = []
        loss_scores = [(unit.loc[unit.base_severity.eq(0), "severity3_score"].to_numpy(), len(unit))
                       for _, unit in cal.groupby("Serial_Number")]
        for tau in candidates:
            total_loss = sum((Fraction(int(np.count_nonzero(values < tau)), count)
                              for values, count in loss_scores), Fraction(0))
            exact_criteria.append((total_loss + 1) / (n + 1))
        curve = pd.read_csv(directory / "calibration_curve.csv", float_precision="round_trip")
        np.testing.assert_allclose(curve.corrected_criterion, [float(c) for c in exact_criteria], atol=1e-15)
        results = pd.read_csv(directory / "results.csv", float_precision="round_trip")
        assert len(results) == 5 and results.alpha.isna().sum() == 1
        assert set(results.alpha.dropna()) == {.10, .125, .15, .20}
        assert results.repetition.eq(rep).all() and results.n_cal.eq(n).all()
        for _, row in results.iterrows():
            tau = row.threshold
            if pd.notna(row.alpha):
                eligible = [i for i, c in enumerate(exact_criteria) if c <= Fraction(str(row.alpha))]
                assert row.empty_set == (len(eligible) == 0)
                exact_tau = candidates[eligible[0]] if eligible else 0.
                if tau != exact_tau:
                    # The saved floating-point rule must still be conservative.
                    assert tau <= exact_tau
                    boundary_rounding.append(dict(n_cal=n, repetition=rep, alpha=row.alpha,
                                                  threshold=tau, exact_threshold=exact_tau))
            missed = evaluation.Severity_Index.eq(3) & evaluation.base_severity.eq(0) & evaluation.severity3_score.lt(tau)
            alarms = evaluation.Severity_Index.eq(0) & (evaluation.base_severity.gt(0) | evaluation.severity3_score.ge(tau))
            assert missed.sum() == row.misses and alarms.sum() == row.false_alarms
            unit_loss = []
            for _, unit in evaluation[evaluation.Severity_Index.eq(3)].groupby("Serial_Number"):
                unit_loss.append(Fraction(int((unit.base_severity.eq(0) & unit.severity3_score.lt(tau)).sum()), len(unit)))
            assert len(unit_loss) == 2
            np.testing.assert_allclose(float(sum(unit_loss) / 2), row.mean_unit_loss, atol=1e-15)
            assert row.alarm_all == (tau == 0) and row.base_selected == (tau == np.inf)
    if not partial:
        assert observed_pairs == {(rep, n) for rep in range(200) for n in [7, 9, 11]}
    record = dict(partitions=len(manifests), exact_fraction_checks=True,
                  conservative_boundary_rounding=boundary_rounding, all_checks_passed=True)
    name = "independent_verification_partial.json" if partial else "independent_verification.json"
    (OUT / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Independent audit: {len(manifests)} partitions; {len(boundary_rounding)} conservative rounding cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    main(parser.parse_args().partial)
