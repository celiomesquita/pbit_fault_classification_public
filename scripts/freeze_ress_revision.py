"""Freeze the audited population and grouped folds for RESS action 18."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "ress_revision" / "freeze_v1"
PROTOCOL = ROOT / "protocols" / "ress_revision_v1.json"
META = ["Experiment_Index", "Serial_Number", "Serial_Run_Index", "Fault_Mode", "Severity_Index"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: str) -> None:
    raw = content.encode("utf-8")
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Frozen artifact differs: {path}. Create a new revision.")
    if not path.exists():
        path.write_bytes(raw)


def save_frame(name: str, frame: pd.DataFrame) -> None:
    write_once(OUT / name, frame.to_csv(index=False, lineterminator="\n"))


def check_metadata(left: pd.DataFrame, right: pd.DataFrame) -> None:
    cols = ["Experiment_Index", "Serial_Number", "Fault_Mode", "Severity_Index"]
    pd.testing.assert_frame_equal(
        left[cols].sort_values("Experiment_Index").reset_index(drop=True),
        right[cols].sort_values("Experiment_Index").reset_index(drop=True),
        check_dtype=False,
    )


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(DATA / "audit_dataset_r1.csv")
    features = pd.read_csv(DATA / "PBIT_features_r1.csv")
    train_old = pd.read_csv(DATA / "train_real.csv")
    test_old = pd.read_csv(DATA / "test_real.csv")
    clean = pd.read_csv(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    for frame in (audit, features, train_old, test_old):
        assert frame.Experiment_Index.is_unique
    assert set(train_old.Serial_Number).isdisjoint(test_old.Serial_Number)
    assert set(train_old.Experiment_Index).isdisjoint(test_old.Experiment_Index)
    old = pd.concat([train_old, test_old], ignore_index=True)
    check_metadata(old, features)
    numeric = [c for c in features if c not in META]
    assert len(numeric) == 246
    pd.testing.assert_frame_equal(
        old.sort_values("Experiment_Index")[numeric].reset_index(drop=True),
        features.sort_values("Experiment_Index")[numeric].reset_index(drop=True),
        check_exact=False, rtol=1e-14, atol=1e-14,
    )
    old_values = old.sort_values("Experiment_Index")[numeric].to_numpy()
    source_values = features.sort_values("Experiment_Index")[numeric].to_numpy()
    max_feature_delta = float(np.nanmax(np.abs(old_values - source_values)))
    excluded = set(audit.loc[~audit.in_D_primary, "Experiment_Index"])
    assert excluded == set(protocol["population"]["excluded_experiments"])
    keep = set(audit.loc[audit.in_D_primary, "Experiment_Index"])
    primary = features[features.Experiment_Index.isin(keep)].copy().reset_index(drop=True)
    train = primary[primary.Serial_Number.isin(train_old.Serial_Number)].copy().reset_index(drop=True)
    test = primary[primary.Serial_Number.isin(test_old.Serial_Number)].copy().reset_index(drop=True)
    assert set(primary.Experiment_Index) == keep
    assert (len(primary), primary.Serial_Number.nunique()) == (1487, 65)
    assert (len(train), train.Serial_Number.nunique()) == (1195, 54)
    assert (len(test), test.Serial_Number.nunique()) == (292, 11)
    assert set(train.Experiment_Index) | set(test.Experiment_Index) == keep
    check_metadata(primary, audit[audit.in_D_primary])
    mapping = protocol["severity_map"]
    assert primary.Fault_Mode.map(mapping).eq(primary.Severity_Index).all()
    clean_primary = clean[clean.Experiment_Index.isin(keep)]
    assert set(clean_primary.Experiment_Index) == keep
    assert clean_primary.groupby("Experiment_Index").size().eq(123).all()
    assert not clean_primary.Measurement.isna().any()
    assert not clean_primary.duplicated(["Experiment_Index", "Sample_Index"]).any()
    check_metadata(primary, clean_primary.drop_duplicates("Experiment_Index"))

    manifest = audit.copy()
    train_units = set(train.Serial_Number)
    manifest["historical_partition"] = np.where(manifest.Serial_Number.isin(train_units), "train", "test")
    manifest["primary_included"] = manifest.in_D_primary
    manifest["sensitivity_included"] = manifest.in_D_sens
    manifest["exclusion_reason"] = np.select(
        [manifest.Experiment_Index.isin([931, 949]), manifest.Experiment_Index.eq(1107)],
        ["R1_missing_more_than_half", "R2_dead_channel_normal_class"], default="",
    )
    save_frame("cycle_manifest.csv", manifest)
    save_frame("features_primary.csv", primary)
    save_frame("train_primary.csv", train)
    save_frame("historical_test_primary.csv", test)
    save_frame("long_primary.csv", clean_primary)

    outer_rows, inner_rows, coverage = [], [], []

    def describe(frame: pd.DataFrame, seed: int, outer: int, inner: int, role: str) -> None:
        row = {"seed": seed, "outer_fold": outer, "inner_fold": inner, "role": role,
               "cycles": len(frame), "units": frame.Serial_Number.nunique(),
               "severity3_cycles": int(frame.Severity_Index.eq(3).sum()),
               "severity3_units": frame.loc[frame.Severity_Index.eq(3), "Serial_Number"].nunique()}
        for label in protocol["class_order"]:
            part = frame[frame.Fault_Mode.eq(label)]
            row[f"cycles_{label}"] = len(part)
            row[f"units_{label}"] = part.Serial_Number.nunique()
        coverage.append(row)

    for seed in protocol["partition_seeds"]:
        splitter = StratifiedGroupKFold(protocol["outer_folds"], shuffle=True, random_state=seed)
        for outer, (tr_idx, te_idx) in enumerate(splitter.split(train, train.Fault_Mode, train.Serial_Number)):
            tr, te = train.iloc[tr_idx], train.iloc[te_idx]
            assert set(tr.Serial_Number).isdisjoint(te.Serial_Number)
            assigned = te[META].copy()
            assigned["seed"], assigned["outer_fold"] = seed, outer
            outer_rows.append(assigned)
            describe(tr, seed, outer, -1, "outer_train")
            describe(te, seed, outer, -1, "outer_evaluation")
            inner_splitter = StratifiedGroupKFold(protocol["inner_folds"], shuffle=True, random_state=seed)
            for inner, (itr, ite) in enumerate(inner_splitter.split(tr, tr.Fault_Mode, tr.Serial_Number)):
                itrain, ival = tr.iloc[itr], tr.iloc[ite]
                assert set(itrain.Serial_Number).isdisjoint(ival.Serial_Number)
                assert set(ival.Serial_Number).isdisjoint(te.Serial_Number)
                assigned = ival[META].copy()
                assigned["seed"], assigned["outer_fold"], assigned["inner_fold"] = seed, outer, inner
                inner_rows.append(assigned)
                describe(itrain, seed, outer, inner, "inner_train")
                describe(ival, seed, outer, inner, "inner_validation")
    outer_df = pd.concat(outer_rows, ignore_index=True)
    inner_df = pd.concat(inner_rows, ignore_index=True)
    assert not outer_df.duplicated(["seed", "Experiment_Index"]).any()
    assert not inner_df.duplicated(["seed", "outer_fold", "Experiment_Index"]).any()
    assert outer_df.groupby("seed").size().eq(len(train)).all()
    for (_, outer), part in inner_df.groupby(["seed", "outer_fold"]):
        seed = int(part.seed.iloc[0])
        external = outer_df[(outer_df.seed == seed) & (outer_df.outer_fold == outer)]
        assert set(part.Experiment_Index) == set(train.Experiment_Index) - set(external.Experiment_Index)
    save_frame("outer_assignment.csv", outer_df)
    save_frame("inner_assignment.csv", inner_df)
    coverage_df = pd.DataFrame(coverage)
    save_frame("fold_coverage.csv", coverage_df)

    count_rows = []
    for name, frame in [("full_primary", primary), ("train_primary", train), ("historical_test", test)]:
        for label in protocol["class_order"]:
            part = frame[frame.Fault_Mode.eq(label)]
            count_rows.append({"population": name, "class": label, "cycles": len(part),
                               "units": part.Serial_Number.nunique(), "severity": mapping[label]})
    save_frame("population_counts.csv", pd.DataFrame(count_rows))
    sources = [PROTOCOL, DATA / "audit_dataset_r1.csv", DATA / "PBIT_features_r1.csv",
               DATA / "train_real.csv", DATA / "test_real.csv", DATA / "PBIT_atuador_DESCARAC_r1_long.csv",
               DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv", ROOT / "src/pbit/features.py",
               ROOT / "src/pbit/modeling.py", ROOT / "src/pbit/augmentation.py", Path(__file__).resolve()]
    input_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    outputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(OUT.glob("*.csv"))}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    frozen = {"protocol_id": protocol["protocol_id"], "frozen_at_utc": protocol["frozen_at_utc"],
              "git_head": head, "working_tree": "modified; relevant_file_hashes_are_authoritative",
              "environment": {"python": platform.python_version(),
                              **{p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]}},
              "input_sha256": input_hashes, "output_sha256": outputs,
              "checks": {"metadata_consistent": True, "historical_feature_values_within_csv_roundoff": True,
                         "maximum_absolute_historical_feature_delta": max_feature_delta,
                         "canonical_feature_source": "data/PBIT_features_r1.csv",
                         "historical_test_units_preserved": True, "all_fold_unit_boundaries_disjoint": True,
                         "primary_cycles": len(primary), "primary_units": primary.Serial_Number.nunique(),
                         "train_cycles": len(train), "train_units": train.Serial_Number.nunique(),
                         "historical_test_cycles": len(test), "historical_test_units": test.Serial_Number.nunique(),
                         "outer_assignment_rows": len(outer_df), "inner_assignment_rows": len(inner_df)},
              "coverage_warnings": {"outer_evaluations_without_F06": int(coverage_df.loc[coverage_df.role.eq("outer_evaluation"), "cycles_F06"].eq(0).sum()),
                                    "inner_training_sets_without_F06": int(coverage_df.loc[coverage_df.role.eq("inner_train"), "cycles_F06"].eq(0).sum()),
                                    "inner_validations_without_severity3": int(coverage_df.loc[coverage_df.role.eq("inner_validation"), "severity3_cycles"].eq(0).sum())}}
    write_once(OUT / "freeze_manifest.json", json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"checks": frozen["checks"], "coverage_warnings": frozen["coverage_warnings"]}, indent=2))


if __name__ == "__main__":
    main()
