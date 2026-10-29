"""Exploratory E12: saved prediction errors, grouped geometry and label traceability."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e12_diagnostic"
E10 = ROOT / "data/e10_v1"
FROZEN = ROOT / "data/ress_revision/freeze_v1"
FOCAL = ("F02", "F05")
ANCHORS = ("B1_complete", "B3_joint")
NORMAL = ("N01", "N02")
META = ["Experiment_Index", "Serial_Number", "Serial_Run_Index", "Fault_Mode", "Severity_Index"]
from e12_geometry import nearest, scale_from_training, shape_rows, unit_weighted_quantiles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def source_list():
    return [Path(__file__), ROOT / "scripts/e12_geometry.py", ROOT / "scripts/ablation_features.py",
            ROOT / "tests/test_e12_geometry.py", ROOT / "tests/test_e12_errors.py", ROOT / "docs/protocolos/E12.md"]


def partition_inputs(item):
    folder = E10 / f"seed_{item['seed']}_outer_{item['outer']}"
    assert read(folder / "complete.json") == item
    def checked(name):
        path = folder / name
        assert sha(path) == item["output_sha256"][name], path
        return path
    selections = read(checked("inner_selections.json"))
    selected = {r["arm"]: r for r in selections if r["arm"] in ANCHORS and r["target"] == .90 and r["budget"] == .40}
    trial = selected["B3_joint"]["trial"]
    e1_item = next(r for r in read(ROOT / "data/e1_full_v1/completion_manifest.json")["completed"]
                   if (r["seed"], r["outer"]) == (item["seed"], item["outer"]))
    origin = ROOT / e1_item["output"]
    old = read(origin / "run_manifest.json")
    assert sha(origin / "run_manifest.json") == e1_item["run_manifest_sha256"]
    def prior(name):
        path = origin / name
        assert sha(path) == old["output_sha256"][name], path
        return path
    result = dict(item=item, folder=folder, selected=selected,
                  predictions=checked("external_predictions.csv"),
                  outside_meta=prior("external_predictions.csv"), inside_meta=prior("inner_predictions.csv"),
                  B1_outside=prior("untuned_external_probability.npy"),
                  B1_inside=prior("untuned_inner_probability.npy"),
                  B3_outside=checked(f"trial_{trial:03d}_external_probability.npy"),
                  B3_inside=checked(f"trial_{trial:03d}_inner_probability.npy"))
    return result


def freeze(partitions):
    assert not OUT.exists(), "Preserve the existing diagnostic and use an amended output version."
    inputs = [FROZEN / name for name in ("train_primary.csv", "long_primary.csv", "outer_assignment.csv", "freeze_manifest.json")]
    old = read(FROZEN / "freeze_manifest.json")
    for p in inputs[:-1]:
        assert sha(p) == old["output_sha256"][p.relative_to(ROOT).as_posix()]
    inputs += [ROOT / name for name in ("data/PBIT_atuador_DESCARAC_r1.csv", "data/PBIT_atuador_DESCARAC_r1.xlsx",
               "data/PBIT_atuador_DESCARAC_r1_long_clean.csv", "data/PBIT_features_r1.csv",
               "data/failure_modes.csv", "docs/proveniencia_e_decisao.md", "data/e10_verification.json",
               "data/e10_v1/completion.json", "data/e10_v1/summary.csv")]
    verification = read(ROOT / "data/e10_verification.json")
    assert sha(E10 / "completion.json") == verification["completion_sha256"]
    assert sha(E10 / "summary.csv") == verification["summary_sha256"]
    for part in partitions:
        inputs += [v for v in part.values() if isinstance(v, Path) and v.is_file()]
        inputs.append(part["folder"] / "inner_selections.json")
    OUT.mkdir()
    for p in source_list():
        target = OUT / "frozen_sources" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write(OUT / "freeze.json", dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
          scope="exploratory E12; no classifier fitting, new threshold or target selection",
          source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_list()},
          input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(inputs))},
          environment={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn")}))


def label_traceability(population, long):
    checks = []
    with (ROOT / "data/PBIT_atuador_DESCARAC_r1.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [next(csv.reader(stream)) for _ in range(3)]
    raw = {int(i): (c, u) for c, i, u in zip(*rows)}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(ROOT / "data/PBIT_atuador_DESCARAC_r1.xlsx") as z:
        strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = workbook.findall("m:sheets/m:sheet", ns)
        data_sheet = next(s for s in sheets if s.attrib["name"] == "DATA")
        rel_id = data_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        links = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(e.attrib["Target"] for e in links if e.attrib["Id"] == rel_id)
        xml_name = target.lstrip("/") if target.startswith("/") else "xl/"+target
        values = {}
        for row in ET.fromstring(z.read(xml_name)).findall("m:sheetData/m:row", ns):
            if int(row.attrib["r"]) > 3:
                break
            for cell in row.findall("m:c", ns):
                value = cell.find("m:v", ns)
                if value is not None:
                    values[cell.attrib["r"]] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
        original = {}
        for address, experiment in values.items():
            if address.endswith("2") and address[:-1].isalpha():
                column = address[:-1]
                original[int(float(experiment))] = (values.get(column+"1"), values.get(column+"3"))
    features = pd.read_csv(ROOT / "data/PBIT_features_r1.csv").set_index("Experiment_Index")
    source_long = pd.read_csv(ROOT / "data/PBIT_atuador_DESCARAC_r1_long_clean.csv")
    source_meta = source_long.groupby("Experiment_Index").agg(Fault_Mode=("Fault_Mode", "first"),
            Serial_Number=("Serial_Number", "first"), classes=("Fault_Mode", "nunique"), units=("Serial_Number", "nunique"))
    for row in population.itertuples():
        exp, expected = row.Experiment_Index, (row.Fault_Mode, row.Serial_Number)
        checks.append(dict(Experiment_Index=exp, Fault_Mode=row.Fault_Mode, Serial_Number=row.Serial_Number,
                           raw_csv_matches=raw.get(exp) == expected, xlsx_matches=original.get(exp) == expected,
                           features_matches=tuple(features.loc[exp, ["Fault_Mode", "Serial_Number"]]) == expected,
                           clean_long_matches=tuple(source_meta.loc[exp, ["Fault_Mode", "Serial_Number"]]) == expected,
                           clean_long_single_label=source_meta.loc[exp, "classes"] == 1,
                           clean_long_single_unit=source_meta.loc[exp, "units"] == 1))
    pd.DataFrame(checks).to_csv(OUT / "label_traceability.csv", index=False)
    return dict(workbook_sheets=[s.attrib["name"] for s in sheets],
                external_label_confirmation="not available in examined project records",
                acquisition_date="not documented; file dates not used",
                real_or_induced="not documented", record_consistency_only=True)


def error_tables(predictions, population):
    focal = predictions[predictions.Fault_Mode.isin(FOCAL)].copy()
    focal["miss"] = focal.prediction.isin(NORMAL)
    focal["exact"] = focal.prediction == focal.Fault_Mode
    cycles = focal.groupby(["arm", "Experiment_Index", "Serial_Number", "Fault_Mode"], sort=True).agg(
        seed_count=("seed", "nunique"), miss_count=("miss", "sum"), exact_count=("exact", "sum")).reset_index()
    assert cycles.seed_count.eq(10).all()
    cycles["miss_fraction"] = cycles.miss_count / 10
    cycles["detect_recall"] = 1-cycles.miss_fraction
    cycles["exact_recall"] = cycles.exact_count / 10
    cycles["always_missed"] = cycles.miss_count == 10
    cycles["never_missed"] = cycles.miss_count == 0
    cycles.to_csv(OUT / "cycle_errors.csv", index=False)
    seed_unit = focal.groupby(["arm", "Fault_Mode", "Serial_Number", "seed"]).agg(
        n_cycles=("Experiment_Index", "size"), misses=("miss", "sum"), exact=("exact", "sum")).reset_index()
    seed_unit["detect_recall"] = 1-seed_unit.misses/seed_unit.n_cycles
    seed_unit["exact_recall"] = seed_unit.exact/seed_unit.n_cycles
    seed_unit.to_csv(OUT / "unit_by_seed.csv", index=False)
    unit_rows, class_rows = [], []
    for (arm, cls), part in cycles.groupby(["arm", "Fault_Mode"]):
        units = []
        total_misses = part.miss_fraction.sum()
        for unit, group in part.groupby("Serial_Number"):
            per_seed = seed_unit[(seed_unit.arm == arm) & (seed_unit.Fault_Mode == cls) & (seed_unit.Serial_Number == unit)]
            units.append(dict(arm=arm, Fault_Mode=cls, Serial_Number=unit, n_cycles=len(group),
                 mean_misses_per_seed=group.miss_fraction.sum(), detect_recall=group.detect_recall.mean(),
                 min_seed_recall=per_seed.detect_recall.min(), max_seed_recall=per_seed.detect_recall.max(),
                 exact_recall=group.exact_recall.mean(), always_missed_cycles=int(group.always_missed.sum()),
                 never_missed_cycles=int(group.never_missed.sum()),
                 exposure_share=len(group)/len(part),
                 miss_share=group.miss_fraction.sum()/total_misses if total_misses else np.nan,
                 same_SN_normal_cycles=int(((population.Serial_Number == unit) & population.Fault_Mode.isin(NORMAL)).sum())))
        units = sorted(units, key=lambda x: (-x["mean_misses_per_seed"], x["Serial_Number"]))
        unit_rows.extend(units)
        row = dict(arm=arm, Fault_Mode=cls, n_cycles=len(part), n_units=len(units),
              cycle_weighted_recall=part.detect_recall.mean(), equal_unit_recall=np.mean([r["detect_recall"] for r in units]),
              units_below_90=sum(r["detect_recall"] < .90-1e-12 for r in units),
              always_missed_cycles=int(part.always_missed.sum()), never_missed_cycles=int(part.never_missed.sum()))
        for k in (1, 2):
            row[f"top{k}_units"] = "; ".join(r["Serial_Number"] for r in units[:k])
            row[f"top{k}_miss_share"] = sum(r["miss_share"] for r in units[:k])
            row[f"top{k}_exposure_share"] = sum(r["exposure_share"] for r in units[:k])
        class_rows.append(row)
    pd.DataFrame(unit_rows).to_csv(OUT / "unit_errors.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "error_concentration.csv", index=False)
    destinations = focal.groupby(["arm", "Fault_Mode", "Serial_Number", "prediction"]).size().rename("count_over_10_seeds").reset_index()
    destinations["mean_count_per_seed"] = destinations.count_over_10_seeds/10
    destinations.to_csv(OUT / "prediction_destinations.csv", index=False)
    return focal, cycles


def geometry_tables(partitions, population, raw, shape, features18):
    population = population.set_index("Experiment_Index", drop=False)
    all_ids = population.index.to_numpy()
    geometry, coverage, scores = [], [], []
    dimensions = []
    representations = {"raw_absolute": raw, "raw_shape": shape,
                       "F246_IQR": population.drop(columns=META), "F18_IQR": features18}
    for part in partitions:
        item = part["item"]
        seed, fold = item["seed"], item["outer"]
        outside = pd.read_csv(part["outside_meta"])
        inside = pd.read_csv(part["inside_meta"])
        test_ids = outside.Experiment_Index.to_numpy()
        train_ids = inside.Experiment_Index.to_numpy()
        assert set(train_ids) | set(test_ids) == set(all_ids)
        assert set(train_ids).isdisjoint(test_ids)
        train = population.loc[train_ids]
        test = population.loc[test_ids]
        assert set(train.Serial_Number).isdisjoint(test.Serial_Number)
        query = test[test.Fault_Mode.isin(FOCAL)]
        normal = train.Fault_Mode.isin(NORMAL).to_numpy()
        for cls in FOCAL:
            for unit, group in query[query.Fault_Mode == cls].groupby("Serial_Number"):
                coverage.append(dict(seed=seed, outer=fold, Fault_Mode=cls, Serial_Number=unit,
                     test_cycles=len(group), train_fault_cycles=int((train.Fault_Mode == cls).sum()),
                     train_fault_units=int(train.loc[train.Fault_Mode == cls, "Serial_Number"].nunique()),
                     train_normal_cycles=int(normal.sum()), train_normal_units=int(train.loc[normal, "Serial_Number"].nunique()),
                     train_N02_cycles=int((train.Fault_Mode == "N02").sum())))
        for name, table in representations.items():
            fitting = table.loc[train_ids].to_numpy(float)
            evaluation = table.loc[query.index].to_numpy(float)
            if name.endswith("IQR"):
                fitting, evaluation, keep = scale_from_training(fitting, evaluation)
                n_dimensions = int(keep.sum())
            else:
                n_dimensions = fitting.shape[1]
            dimensions.append(dict(seed=seed, outer=fold, representation=name, dimensions=n_dimensions))
            distances, _ = nearest(fitting[normal], fitting[normal], train.loc[normal, "Serial_Number"].to_numpy(),
                                   train.loc[normal, "Serial_Number"].to_numpy())
            q05, q50, q95 = unit_weighted_quantiles(distances, train.loc[normal, "Serial_Number"].to_numpy())
            normal_positions = np.flatnonzero(normal)
            dn, ni = nearest(evaluation, fitting[normal])
            for cls in FOCAL:
                positions = np.flatnonzero(query.Fault_Mode.to_numpy() == cls)
                candidates = np.flatnonzero(train.Fault_Mode.to_numpy() == cls)
                df, fi = nearest(evaluation[positions], fitting[candidates])
                for j, k in enumerate(positions):
                    normal_id = int(train_ids[normal_positions[ni[k]]]) if ni[k] >= 0 else None
                    fault_id = int(train_ids[candidates[fi[j]]]) if fi[j] >= 0 else None
                    ratio = dn[k]/df[j] if np.isfinite(df[j]) and df[j] > 0 else np.nan
                    geometry.append(dict(seed=seed, outer=fold, Experiment_Index=int(query.index[k]),
                         Serial_Number=query.Serial_Number.iloc[k], Fault_Mode=cls, representation=name,
                         normal_neighbor_id=normal_id, normal_neighbor_class=population.loc[normal_id, "Fault_Mode"] if normal_id else None,
                         normal_neighbor_SN=population.loc[normal_id, "Serial_Number"] if normal_id else None,
                         fault_neighbor_id=fault_id, fault_neighbor_SN=population.loc[fault_id, "Serial_Number"] if fault_id else None,
                         distance_normal=dn[k], distance_same_fault=df[j], normal_to_fault_ratio=ratio,
                         normal_closer_or_equal=bool(dn[k] <= df[j]+1e-12) if np.isfinite(df[j]) else np.nan,
                         normal_training_q05=q05, normal_training_q50=q50, normal_training_q95=q95,
                         within_normal_q95=bool(dn[k] <= q95+1e-12) if np.isfinite(q95) else np.nan))
        query_positions = outside.Experiment_Index.isin(query.index).to_numpy()
        for arm, prefix in zip(ANCHORS, ("B1", "B3")):
            selection = part["selected"][arm]
            threshold = np.inf if selection["threshold"] == "infinity" else float(selection["threshold"])
            p_out, p_in = np.load(part[prefix+"_outside"]), np.load(part[prefix+"_inside"])
            assert p_out.shape == (len(outside), 8) and p_in.shape == (len(inside), 8)
            q_out, q_in = p_out[:, 2:].sum(axis=1), p_in[:, 2:].sum(axis=1)
            for k in np.flatnonzero(query_positions):
                cls = outside.Fault_Mode.iloc[k]
                inner_class = inside.Fault_Mode.to_numpy() == cls
                inner_recall = np.mean(q_in[inner_class] >= threshold) if inner_class.any() else np.nan
                assert np.isclose(inner_recall, selection["recalls"][cls], atol=1e-12)
                scores.append(dict(seed=seed, outer=fold, arm=arm, Experiment_Index=int(outside.Experiment_Index.iloc[k]),
                     Fault_Mode=cls, Serial_Number=outside.Serial_Number.iloc[k], score_q=q_out[k], threshold=threshold,
                     score_margin=q_out[k]-threshold, inner_class_recall=inner_recall,
                     inner_class_q_median=np.median(q_in[inner_class]) if inner_class.any() else np.nan,
                     inner_FA0=selection["FA0"], inner_status=selection["status"]))
        print(f"E12 geometry complete: seed {seed}, fold {fold}", flush=True)
    pd.DataFrame(geometry).to_csv(OUT / "geometry.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "training_coverage.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "score_diagnostics.csv", index=False)
    pd.DataFrame(dimensions).to_csv(OUT / "representation_dimensions.csv", index=False)
    return pd.DataFrame(geometry)


def within_unit_oracle(population, raw, shape):
    rows = []
    for cls in FOCAL:
        for unit, query in population[population.Fault_Mode == cls].groupby("Serial_Number"):
            reference = population[(population.Serial_Number == unit) & population.Fault_Mode.isin(NORMAL)]
            for name, table in (("raw_absolute", raw), ("raw_shape", shape)):
                distance, index = nearest(table.loc[query.Experiment_Index], table.loc[reference.Experiment_Index])
                for k, exp in enumerate(query.Experiment_Index):
                    rows.append(dict(Experiment_Index=int(exp), Fault_Mode=cls, Serial_Number=unit,
                         representation=name, normal_reference_cycles=len(reference), distance_normal_same_SN=distance[k],
                         normal_neighbor_id=int(reference.Experiment_Index.iloc[index[k]]) if index[k] >= 0 else None,
                         scope="within-unit retrospective oracle; unavailable to held-unit training"))
    pd.DataFrame(rows).to_csv(OUT / "within_unit_oracle.csv", index=False)


def duplicate_signals(population, raw):
    meta = population.set_index("Experiment_Index")
    rows = []
    for precision in (None, 6, 3):
        values = raw.to_numpy() if precision is None else raw.to_numpy().round(precision)
        buckets = defaultdict(list)
        for exp, value in zip(raw.index, values):
            buckets[hashlib.sha256(value.tobytes()).hexdigest()].append(int(exp))
        for digest, ids in buckets.items():
            classes = set(meta.loc[ids, "Fault_Mode"])
            if len(classes) < 2 or not classes.intersection(FOCAL):
                continue
            for a in ids:
                if meta.loc[a, "Fault_Mode"] not in FOCAL:
                    continue
                for b in ids:
                    if meta.loc[a, "Fault_Mode"] == meta.loc[b, "Fault_Mode"]:
                        continue
                    rows.append(dict(precision="exact" if precision is None else str(precision),
                         focal_id=a, focal_class=meta.loc[a, "Fault_Mode"], focal_SN=meta.loc[a, "Serial_Number"],
                         other_id=b, other_class=meta.loc[b, "Fault_Mode"], other_SN=meta.loc[b, "Serial_Number"],
                         same_SN=bool(meta.loc[a, "Serial_Number"] == meta.loc[b, "Serial_Number"]), signature=digest))
    pd.DataFrame(rows, columns=["precision", "focal_id", "focal_class", "focal_SN", "other_id", "other_class", "other_SN", "same_SN", "signature"]).to_csv(OUT / "cross_label_signal_matches.csv", index=False)


def run():
    started = time.monotonic()
    items = read(E10 / "completion.json")["completed"]
    assert len(items) == 50
    partitions = [partition_inputs(item) for item in items]
    freeze(partitions)
    population = pd.read_csv(FROZEN / "train_primary.csv")
    assert len(population) == 1195 and population.Experiment_Index.is_unique
    long = pd.read_csv(FROZEN / "long_primary.csv")
    long = long[long.Experiment_Index.isin(population.Experiment_Index)]
    assert long.groupby("Experiment_Index").size().eq(123).all() and long.Measurement.notna().all()
    raw = long.pivot(index="Experiment_Index", columns="Sample_Index", values="Measurement").sort_index()
    shape_values, flat = shape_rows(raw.to_numpy())
    shape = pd.DataFrame(shape_values, index=raw.index, columns=raw.columns)
    pd.DataFrame(dict(Experiment_Index=raw.index, flat_signal=flat)).to_csv(OUT / "flat_signals.csv", index=False)
    with warnings.catch_warnings():
        from ablation_features import f18_table
        features18 = f18_table(long)
    frames = []
    for part in partitions:
        frame = pd.read_csv(part["predictions"])
        frame = frame[np.isclose(frame.target, .90) & np.isclose(frame.budget, .40)]
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    assert len(predictions) == 1195*10*8
    assert all(len(g) == 1195 and g.Experiment_Index.is_unique for _, g in predictions.groupby(["seed", "arm"]))
    focal, cycles = error_tables(predictions, population)
    geometry = geometry_tables(partitions, population, raw, shape, features18)
    within_unit_oracle(population, raw, shape)
    duplicate_signals(population, raw)
    provenance = label_traceability(population, long)
    write(OUT / "provenance_scope.json", provenance)
    anchors = focal[focal.arm.isin(ANCHORS)][["seed", "arm", "Experiment_Index", "miss"]]
    joined = geometry.merge(anchors, on=["seed", "Experiment_Index"], validate="many_to_many")
    summaries = []
    for (arm, cls, rep), group in joined.groupby(["arm", "Fault_Mode", "representation"]):
        units = group.groupby("Serial_Number").agg(normal_closer=("normal_closer_or_equal", "mean"),
                                                   within_normal=("within_normal_q95", "mean"))
        summaries.append(dict(arm=arm, Fault_Mode=cls, representation=rep, n_units=len(units),
             n_case_seed_records=len(group), normal_closer_cycle_fraction=group.normal_closer_or_equal.mean(),
             normal_closer_equal_unit_fraction=units.normal_closer.mean(),
             within_normal_q95_cycle_fraction=group.within_normal_q95.mean(),
             within_normal_q95_equal_unit_fraction=units.within_normal.mean(),
             normal_closer_missed_fraction=group.loc[group.miss, "normal_closer_or_equal"].mean(),
             normal_closer_detected_fraction=group.loc[~group.miss, "normal_closer_or_equal"].mean(),
             mean_signal_normal_distance=group.distance_normal.mean(),
             missing_same_class_references=int(group.distance_same_fault.isna().sum())))
    pd.DataFrame(summaries).to_csv(OUT / "geometry_summary.csv", index=False)
    # The primary cycle/seed decisions must match the q/threshold records exactly.
    score = pd.read_csv(OUT / "score_diagnostics.csv").merge(anchors, on=["seed", "arm", "Experiment_Index"], validate="one_to_one")
    assert np.array_equal(score.score_q.to_numpy() < score.threshold.to_numpy(), score.miss.to_numpy())
    candidates = cycles[cycles.arm == "B1_complete"].merge(
        geometry[(geometry.seed == 42) & (geometry.representation == "raw_absolute")][
            ["Experiment_Index", "normal_to_fault_ratio", "normal_neighbor_id", "fault_neighbor_id", "outer"]],
        on="Experiment_Index", validate="one_to_one")
    examples = []
    for cls in FOCAL:
        group = candidates[candidates.Fault_Mode == cls]
        difficult = group.sort_values(["miss_fraction", "normal_to_fault_ratio", "Experiment_Index"], ascending=[False, True, True]).iloc[0]
        detected = group[group.Experiment_Index != difficult.Experiment_Index].sort_values(["miss_fraction", "Experiment_Index"]).iloc[0]
        for kind, row in (("largest_miss_fraction", difficult), ("smallest_miss_fraction", detected)):
            examples.append(dict(kind=kind, Fault_Mode=cls, Experiment_Index=int(row.Experiment_Index),
                 Serial_Number=row.Serial_Number, miss_fraction=row.miss_fraction, seed=42, outer=int(row.outer),
                 normal_neighbor_id=int(row.normal_neighbor_id) if pd.notna(row.normal_neighbor_id) else None,
                 fault_neighbor_id=int(row.fault_neighbor_id) if pd.notna(row.fault_neighbor_id) else None))
    pd.DataFrame(examples).to_csv(OUT / "illustrative_examples.csv", index=False)
    checks = pd.read_csv(OUT / "label_traceability.csv")
    output = {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob("*") if p.is_file() and "frozen_sources" not in p.parts}
    write(OUT / "completion.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-started, population_cycles=1195, population_units=54,
          target_cycle_seed_records=len(focal), geometry_records=len(geometry),
          geometry_fold_count=len(partitions), classifier_fits=0, thresholds_changed=False,
          label_discrepancies={c: int((~checks[c]).sum()) for c in checks if c.endswith("matches") or c.endswith("label") or c.endswith("unit")},
          output_sha256=output))
    print(f"E12 complete: {len(geometry)} geometry records; no classifier fits.", flush=True)


if __name__ == "__main__":
    run()
