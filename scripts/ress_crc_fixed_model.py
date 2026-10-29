"""Evaluate unit-level CRC using a fixed model and disjoint calibration units."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.augmentation import augment_training_set
from pbit.features import extract_features
from pbit.modeling import META_COLS, SEVERITY_MAP, compute_sample_weights

FROZEN = ROOT / "data/ress_revision/freeze_v1"
PROTOCOL = ROOT / "protocols/ress_crc_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    out = ROOT / cfg["outputs"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_manifest.json").exists():
        raise ValueError("Completed run already exists; use a new protocol/version for another run.")
    frozen_manifest = json.loads((FROZEN / "freeze_manifest.json").read_text(encoding="utf-8"))
    for name, expected in frozen_manifest["output_sha256"].items():
        assert sha(ROOT / name) == expected, name
    train_population = pd.read_csv(FROZEN / "train_primary.csv")
    evaluation = pd.read_csv(FROZEN / "historical_test_primary.csv")
    long_df = pd.read_csv(FROZEN / "long_primary.csv")
    critical_units = sorted(train_population.loc[train_population.Severity_Index.eq(3), "Serial_Number"].unique())
    rng = np.random.default_rng(cfg["calibration_selection_seed"])
    cal_units = sorted(rng.choice(critical_units, cfg["calibration_unit_count"], replace=False).tolist())
    cal = train_population[train_population.Serial_Number.isin(cal_units)].copy()
    train = train_population[~train_population.Serial_Number.isin(cal_units)].copy()
    sets = [set(frame.Serial_Number) for frame in [train, cal, evaluation]]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert len(cal_units) == cal.Serial_Number.nunique()
    partition = pd.concat([frame[META_COLS].assign(role=role) for role, frame in [
        ("train", train), ("calibration", cal), ("historical_evaluation", evaluation)]], ignore_index=True)
    partition.to_csv(out / "partition.csv", index=False)
    print(json.dumps({"calibration_units": cal_units,
                      "train_cycles": len(train), "train_units": len(sets[0]),
                      "calibration_cycles": len(cal), "calibration_units_count": len(sets[1]),
                      "evaluation_cycles": len(evaluation), "evaluation_units": len(sets[2])}), flush=True)
    cols = [c for c in train if c not in META_COLS]
    prep = cfg["preprocessing"]
    start = time.monotonic()
    aug_long = augment_training_set(long_df, set(train.Experiment_Index),
                                    target_per_class=prep["augmentation_target"],
                                    noise_scale=prep["augmentation_noise"], random_state=prep["augmentation_seed"])
    aug = extract_features(aug_long)
    expanded = pd.concat([train, aug], ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X = imputer.fit_transform(expanded[cols])
    y = expanded.Fault_Mode.to_numpy()
    model = lgb.LGBMClassifier(**cfg["model"])
    model.fit(X, y, sample_weight=compute_sample_weights(y, "freq_sev"))
    model.booster_.save_model(str(out / "fixed_model.txt"))
    pd.DataFrame({"feature": cols, "training_imputation_statistic": imputer.statistics_}).to_csv(out / "imputation.csv", index=False)
    scored = []
    for role, frame in [("calibration", cal), ("historical_evaluation", evaluation)]:
        p = model.predict_proba(imputer.transform(frame[cols]))
        classes = list(model.classes_)
        d = frame[META_COLS].copy()
        d["role"] = role
        d["base_prediction"] = np.asarray(classes)[p.argmax(axis=1)]
        d["base_severity"] = d.base_prediction.map(SEVERITY_MAP)
        d["severity3_score"] = sum(p[:, classes.index(label)] if label in classes else np.zeros(len(d)) for label in ["F03", "F06"])
        scored.append(d)
    scores = pd.concat(scored, ignore_index=True)
    scores.to_csv(out / "scores.csv", index=False)
    c = scores[(scores.role == "calibration") & scores.Severity_Index.eq(3)].copy()
    e = scores[scores.role == "historical_evaluation"].copy()
    candidates = np.concatenate([[np.inf], np.sort(c.severity3_score.unique())[::-1], [0.0]])
    cfinal = np.where(c.severity3_score.to_numpy()[:, None] >= candidates, 3, c.base_severity.to_numpy()[:, None])
    unit_losses = np.stack([(cfinal[c.Serial_Number.eq(unit).to_numpy()] == 0).mean(axis=0) for unit in cal_units])
    assert np.all(np.diff(unit_losses, axis=1) <= 1e-15)
    assert np.all(unit_losses[:, -1] == 0)
    n = len(cal_units)
    mean_losses = unit_losses.mean(axis=0)
    corrected = (n * mean_losses + 1) / (n + 1)
    loss_table = pd.DataFrame(unit_losses.T, columns=cal_units)
    loss_table.insert(0, "threshold", candidates)
    loss_table["mean_unit_loss"], loss_table["corrected_criterion"] = mean_losses, corrected
    loss_table.to_csv(out / "calibration_loss_curve.csv", index=False)

    def evaluate(threshold: float) -> dict:
        final = np.where(e.severity3_score >= threshold, 3, e.base_severity)
        crit, normal = e.Severity_Index.eq(3).to_numpy(), e.Severity_Index.eq(0).to_numpy()
        missed = crit & (final == 0)
        critical_sns = sorted(e.loc[crit, "Serial_Number"].unique())
        losses = [(final[crit & e.Serial_Number.eq(sn).to_numpy()] == 0).mean() for sn in critical_sns]
        return {"evaluation_CMR": float(missed.sum() / crit.sum()),
                "evaluation_FA0": float(((final >= 1) & normal).sum() / normal.sum()),
                "evaluation_mean_critical_unit_loss": float(np.mean(losses)),
                "evaluation_critical_units": len(losses),
                "evaluation_units_with_miss": int(np.count_nonzero(np.asarray(losses) > 0)),
                "evaluation_critical_misses": int(missed.sum())}

    rows = [{"alpha": np.nan, "threshold": np.inf, "empty_admissible_set": False,
             "criterion_satisfied": np.nan, "calibration_mean_unit_loss": mean_losses[0],
             "corrected_criterion": corrected[0], "policy": "uncalibrated_base", **evaluate(np.inf)}]
    for alpha in cfg["alphas"]:
        admissible = np.flatnonzero(corrected <= alpha)
        empty = len(admissible) == 0
        selected = int(admissible[0]) if not empty else len(candidates) - 1
        threshold = float(candidates[selected])
        row = {"alpha": alpha, "threshold": threshold, "empty_admissible_set": empty,
               "criterion_satisfied": bool(corrected[selected] <= alpha),
               "calibration_mean_unit_loss": mean_losses[selected],
               "corrected_criterion": corrected[selected],
               "policy": "alarm_all_empty_set" if empty else "CRC_selected", **evaluate(threshold)}
        if empty:
            assert row["evaluation_CMR"] == 0 and row["evaluation_FA0"] == 1
        else:
            assert row["corrected_criterion"] <= alpha
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "results.csv", index=False)
    sources = [PROTOCOL, Path(__file__).resolve(), FROZEN / "freeze_manifest.json",
               ROOT / "src/pbit/augmentation.py", ROOT / "src/pbit/features.py", ROOT / "src/pbit/modeling.py"]
    manifest = {"protocol_id": cfg["protocol_id"], "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                "output_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                "runtime_seconds": time.monotonic() - start, "model_classes": list(model.classes_),
                "synthetic_training_cycles": len(aug), "calibration_units": cal_units,
                "environment": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm"]},
                "checks": {"three_disjoint_unit_sets": True, "monotone_unit_losses": True,
                           "zero_loss_at_alarm_all": True, "empty_set_distinguished_from_criterion_feasibility": True},
                "scope": cfg["empirical_scope"]}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
