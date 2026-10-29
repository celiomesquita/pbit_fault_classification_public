"""E6: fixed-model CRC with repeated, disjoint unit partitions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.augmentation import augment_training_set
from pbit.features import extract_features
from pbit.modeling import META_COLS, SEVERITY_MAP, compute_sample_weights

OUT = ROOT / "data/crc_repeated_partitions"
FROZEN = ROOT / "data/ress_revision/freeze_v1"
PROTOCOL = ROOT / "docs/protocolos/E6.md"
CONFIG = dict(repetitions=200, n_cal=[7, 9, 11], alphas=[0.10, 0.125, 0.15, 0.20],
              partition_seed=20261011, evaluation_critical=2, evaluation_other=10,
              augmentation=dict(target_per_class=160, noise_scale=0.01, random_state=42),
              model=dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                         min_child_samples=5, subsample=0.8, subsample_freq=0,
                         colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1))
SOURCE_PATHS = [PROTOCOL, Path(__file__), FROZEN / "freeze_manifest.json",
                FROZEN / "features_primary.csv", FROZEN / "long_primary.csv",
                ROOT / "src/pbit/augmentation.py", ROOT / "src/pbit/features.py",
                ROOT / "src/pbit/modeling.py"]
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def check_sources():
    freeze = json.loads((OUT / "protocol_freeze.json").read_text(encoding="utf-8"))
    assert freeze["config"] == CONFIG
    for name, expected in freeze["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert sha(OUT / "partitions.csv") == freeze["partitions_sha256"]
    return freeze


def prepare():
    if (OUT / "protocol_freeze.json").exists():
        check_sources()
        print("Existing protocol and partitions verified.", flush=True)
        return
    source_manifest = json.loads((FROZEN / "freeze_manifest.json").read_text(encoding="utf-8"))
    for path in [FROZEN / "features_primary.csv", FROZEN / "long_primary.csv"]:
        assert sha(path) == source_manifest["output_sha256"][relative(path)]
    features = pd.read_csv(FROZEN / "features_primary.csv")
    assert len(features) == 1487 and features.Serial_Number.nunique() == 65
    assert not set(features.Experiment_Index) & {931, 949, 1107}
    assert features.Experiment_Index.is_unique
    critical = sorted(features.loc[features.Severity_Index.eq(3), "Serial_Number"].unique())
    others = sorted(set(features.Serial_Number) - set(critical))
    assert len(critical) == 16 and len(others) == 49
    rows = []
    for rep in range(CONFIG["repetitions"]):
        rng = np.random.default_rng(CONFIG["partition_seed"] + rep)
        cp, op = rng.permutation(critical).tolist(), rng.permutation(others).tolist()
        evaluation = cp[:2] + op[:10]
        for n in CONFIG["n_cal"]:
            calibration = cp[2:2 + n]
            train = cp[2 + n:] + op[10:]
            for role, units in [("train", train), ("calibration", calibration), ("evaluation", evaluation)]:
                rows.extend(dict(repetition=rep, n_cal=n, Serial_Number=u, role=role,
                                 critical=u in critical) for u in sorted(units))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "partitions.csv", index=False)
    write_json(OUT / "protocol_freeze.json", dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
        source_sha256={relative(p): sha(p) for p in SOURCE_PATHS},
        partitions_sha256=sha(OUT / "partitions.csv"),
        environment={p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm"]},
        python=sys.version, scope="retrospective repeated partitions; not external validation"))
    print("Frozen 600 partitions before fitting.", flush=True)


def init_worker():
    global FEATURES, LONG, PARTITIONS, COLS
    FEATURES = pd.read_csv(FROZEN / "features_primary.csv")
    LONG = pd.read_csv(FROZEN / "long_primary.csv")
    PARTITIONS = pd.read_csv(OUT / "partitions.csv")
    COLS = [c for c in FEATURES if c not in META_COLS]
    assert len(COLS) == 246
    assert set(LONG.Experiment_Index) == set(FEATURES.Experiment_Index)


def calibration_curve(cal):
    c = cal[cal.Severity_Index.eq(3)]
    candidates = np.r_[np.inf, np.sort(c.severity3_score.unique())[::-1], 0.0]
    final = np.where(c.severity3_score.to_numpy()[:, None] >= candidates,
                     3, c.base_severity.to_numpy()[:, None])
    losses = np.stack([(final[c.Serial_Number.eq(u).to_numpy()] == 0).mean(axis=0)
                       for u in sorted(c.Serial_Number.unique())])
    assert np.all(np.diff(losses, axis=1) <= 1e-15) and np.all(losses[:, -1] == 0)
    n = len(losses)
    table = pd.DataFrame(losses.T, columns=sorted(c.Serial_Number.unique()))
    table.insert(0, "threshold", candidates)
    table["mean_unit_loss"] = losses.mean(axis=0)
    table["corrected_criterion"] = (losses.sum(axis=0) + 1) / (n + 1)
    return table


def evaluate(scores, threshold):
    e = scores.copy()
    e["final_severity"] = np.where(e.severity3_score >= threshold, 3, e.base_severity)
    e["miss"] = e.Severity_Index.eq(3) & e.final_severity.eq(0)
    e["false_alarm"] = e.Severity_Index.eq(0) & e.final_severity.gt(0)
    units = []
    for unit, group in e.groupby("Serial_Number", sort=True):
        c, normal = group.Severity_Index.eq(3), group.Severity_Index.eq(0)
        units.append(dict(Serial_Number=unit, critical_cycles=int(c.sum()),
                          normal_cycles=int(normal.sum()), misses=int(group["miss"].sum()),
                          false_alarms=int(group.false_alarm.sum())))
    units = pd.DataFrame(units)
    c, normal = units.critical_cycles.gt(0), units.normal_cycles.gt(0)
    cmr = units.misses.sum() / units.critical_cycles.sum()
    fa0 = units.false_alarms.sum() / units.normal_cycles.sum() if normal.any() else np.nan
    loss = (units.loc[c, "misses"] / units.loc[c, "critical_cycles"]).mean()
    return dict(mean_unit_loss=float(loss), CMR=float(cmr), FA0=float(fa0),
                FA0_unit_mean=float((units.loc[normal, "false_alarms"] /
                                     units.loc[normal, "normal_cycles"]).mean()),
                critical_cycles=int(units.critical_cycles.sum()), misses=int(units.misses.sum()),
                normal_cycles=int(units.normal_cycles.sum()),
                false_alarms=int(units.false_alarms.sum())), units


def fit_partition(task):
    rep, n, destination = task
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    partition = PARTITIONS[(PARTITIONS.repetition == rep) & (PARTITIONS.n_cal == n)]
    sets = {role: set(partition.loc[partition.role.eq(role), "Serial_Number"])
            for role in ["train", "calibration", "evaluation"]}
    assert all(sets[a].isdisjoint(sets[b]) for a, b in
               [("train", "calibration"), ("train", "evaluation"), ("calibration", "evaluation")])
    assert len(sets["calibration"]) == n and len(sets["evaluation"]) == 12
    train = FEATURES[FEATURES.Serial_Number.isin(sets["train"])]
    train_ids = set(train.Experiment_Index)
    start = time.monotonic()
    aug_long = augment_training_set(LONG, train_ids, **CONFIG["augmentation"])
    assert set(aug_long.Serial_Number) <= sets["train"]
    aug = extract_features(aug_long) if len(aug_long) else train.iloc[:0]
    expanded = pd.concat([train, aug], ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X = imputer.fit_transform(expanded[COLS])
    model = lgb.LGBMClassifier(**CONFIG["model"])
    y = expanded.Fault_Mode.to_numpy()
    model.fit(X, y, sample_weight=compute_sample_weights(y, "freq_sev"))
    model_bytes = model.booster_.model_to_string().encode("utf-8")
    if rep == 0:
        (destination / "model.txt.gz").write_bytes(gzip.compress(model_bytes, mtime=0))
    pd.DataFrame(dict(feature=COLS, statistic=imputer.statistics_)).to_csv(
        destination / "imputation.csv", index=False)
    scored = []
    for role in ["calibration", "evaluation"]:
        frame = FEATURES[FEATURES.Serial_Number.isin(sets[role])]
        p = model.predict_proba(imputer.transform(frame[COLS]))
        classes = list(model.classes_)
        scores = frame[META_COLS].copy().assign(role=role)
        scores["base_prediction"] = np.asarray(classes)[p.argmax(axis=1)]
        scores["base_severity"] = scores.base_prediction.map(SEVERITY_MAP)
        scores["severity3_score"] = sum(p[:, classes.index(c)] if c in classes else np.zeros(len(p))
                                          for c in ["F03", "F06"])
        for index, label in enumerate(classes):
            scores["P_" + label] = p[:, index]
        scored.append(scores)
    scores = pd.concat(scored, ignore_index=True)
    scores.to_csv(destination / "scores.csv", index=False)
    curve = calibration_curve(scores[scores.role.eq("calibration")])
    curve.to_csv(destination / "calibration_curve.csv", index=False)
    evaluation = scores[scores.role.eq("evaluation")]
    results, unit_results = [], []
    for alpha in [None] + CONFIG["alphas"]:
        if alpha is None:
            idx, empty, policy = 0, False, "uncalibrated_base"
        else:
            eligible = np.flatnonzero(curve.corrected_criterion.to_numpy() <= alpha)
            empty = len(eligible) == 0
            idx = len(curve) - 1 if empty else int(eligible[0])
            policy = "alarm_all_empty_set" if empty else "CRC_selected"
        threshold = float(curve.threshold.iloc[idx])
        metrics, units = evaluate(evaluation, threshold)
        if threshold == 0:
            assert metrics["CMR"] == 0
            assert np.isnan(metrics["FA0"]) or metrics["FA0"] == 1
        if alpha is not None and not empty:
            assert curve.corrected_criterion.iloc[idx] <= alpha
        results.append(dict(repetition=rep, n_cal=n, alpha=alpha, policy=policy,
                            threshold=threshold, empty_set=empty, alarm_all=threshold == 0,
                            base_selected=threshold == np.inf,
                            calibration_loss=float(curve.mean_unit_loss.iloc[idx]),
                            corrected_criterion=float(curve.corrected_criterion.iloc[idx]),
                            **metrics))
        unit_results.append(units.assign(repetition=rep, n_cal=n, alpha=alpha, policy=policy))
    pd.DataFrame(results).to_csv(destination / "results.csv", index=False)
    pd.concat(unit_results, ignore_index=True).to_csv(destination / "unit_results.csv", index=False)
    write_json(destination / "manifest.json", dict(repetition=rep, n_cal=n,
        runtime_seconds=time.monotonic() - start, model_classes=list(model.classes_),
        train_cycles=len(train), train_units=len(sets["train"]),
        train_critical_units=train.loc[train.Severity_Index.eq(3), "Serial_Number"].nunique(),
        synthetic_cycles=len(aug), model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        training_features_sha256=hashlib.sha256(expanded.to_csv(index=False).encode()).hexdigest(),
        output_sha256={p.name: sha(p) for p in sorted(destination.iterdir()) if p.name != "manifest.json"}))
    return rep, n


def run(workers, limit):
    check_sources()
    tasks = [(r, n, str(OUT / "run" / f"n{n}_r{r:03d}"))
             for r in range(CONFIG["repetitions"]) for n in CONFIG["n_cal"]]
    pending = []
    for task in tasks:
        manifest_path = Path(task[2]) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, expected in manifest["output_sha256"].items():
                assert sha(manifest_path.parent / name) == expected
        else:
            pending.append(task)
    if limit:
        pending = pending[:limit]
    print(f"Starting {len(pending)} fits, {workers} workers.", flush=True)
    start, last_print = time.monotonic(), time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        futures = {pool.submit(fit_partition, task): task for task in pending}
        done = 0
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                write_json(OUT / "failure.json", dict(task=futures[future], error=repr(error)))
                raise
            done += 1
            if time.monotonic() - last_print >= 30 or done == len(pending):
                print(f"Completed {done}/{len(pending)} new fits in {time.monotonic()-start:.0f}s.", flush=True)
                last_print = time.monotonic()
    check_sources()


def summarise():
    check_sources()
    paths = sorted((OUT / "run").glob("*/results.csv"))
    assert len(paths) == 600
    results = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(results) == 3000
    results.to_csv(OUT / "results.csv", index=False)
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "run").glob("*/manifest.json"))]
    pd.DataFrame([{k: m[k] for k in ["repetition", "n_cal", "runtime_seconds", "train_cycles",
                                   "train_units", "train_critical_units", "synthetic_cycles"]} |
                 {"has_F06": "F06" in m["model_classes"], "has_F03": "F03" in m["model_classes"]}
                 for m in manifests]).to_csv(OUT / "training_summary.csv", index=False)
    rows = []
    for (n, alpha), group in results.groupby(["n_cal", "alpha"], dropna=False, sort=True):
        row = dict(n_cal=int(n), alpha=alpha, repetitions=len(group),
                   empty_set_fraction=group.empty_set.mean(), alarm_all_fraction=group.alarm_all.mean(),
                   base_selected_fraction=group.base_selected.mean(),
                   loss_above_alpha_fraction=(group.mean_unit_loss > alpha).mean() if pd.notna(alpha) else np.nan,
                   undefined_FA0=int(group.FA0.isna().sum()))
        for metric in ["mean_unit_loss", "CMR", "FA0", "FA0_unit_mean"]:
            v = group[metric]
            row.update({f"{metric}_{s}": value for s, value in
                        [("mean", v.mean()), ("median", v.median()), ("q05", v.quantile(.05)),
                         ("q95", v.quantile(.95)), ("min", v.min()), ("max", v.max())]})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    lines = ["# E6 — CRC em 600 divisões por unidade", "",
             "200 rodadas por tamanho; as mesmas duas SN críticas ficam na avaliação dos três tamanhos.",
             "Dez SN não críticas também ficam na avaliação. Treino crítico: 7, 5 e 3 SN.", "",
             "| Calibração | Alvo | Perda média por SN | CMR média | FA0 médio | Conjunto vazio | Alarme para todos |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        alpha = "base" if pd.isna(row["alpha"]) else f'{row["alpha"]:.3f}'
        lines.append(f'| {row["n_cal"]} | {alpha} | {row["mean_unit_loss_mean"]:.6f} | '
                     f'{row["CMR_mean"]:.6f} | {row["FA0_mean"]:.6f} | '
                     f'{row["empty_set_fraction"]:.3f} | {row["alarm_all_fraction"]:.3f} |')
    lines += ["", "FA0 e CMR são médias das razões de cada divisão, não razões de totais acumulados.",
              "Percentis, extremos, perdas por SN, classes ausentes e scores estão nos CSVs.",
              "As rodadas reutilizam as mesmas unidades: não são novas observações independentes.",
              "A comparação entre tamanhos também muda o treino. Não demonstra garantia populacional",
              "nem permutabilidade entre F03/F06; não se aplicam testes ou IC reprovados no E3.", ""]
    (ROOT / "data/crc_repeated_summary.md").write_text("\n".join(lines), encoding="utf-8")
    table = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$n_{\rm cal}$ & $\alpha$ & Unit loss & CMR & FA0 & Alarm-all \\", r"\midrule"]
    for row in rows:
        alpha = "Base" if pd.isna(row["alpha"]) else f'{row["alpha"]:.3f}'
        table.append(f'{row["n_cal"]} & {alpha} & {row["mean_unit_loss_mean"]:.3f} & '
                     f'{row["CMR_mean"]:.3f} & {row["FA0_mean"]:.3f} & {row["alarm_all_fraction"]:.3f} ' + r"\\")
    table += [r"\bottomrule", r"\end{tabular}", ""]
    (ROOT / "paper/supplement/tables/tab_crc_repeated.tex").write_text("\n".join(table), encoding="utf-8")
    write_json(OUT / "run_manifest.json", dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(), fits=600,
        protocol_freeze_sha256=sha(OUT / "protocol_freeze.json"),
        output_sha256={relative(p): sha(p) for p in sorted((OUT / "run").rglob("*")) if p.is_file()} |
                      {relative(p): sha(p) for p in [OUT / "results.csv", OUT / "summary.csv",
                       OUT / "training_summary.csv", ROOT / "data/crc_repeated_summary.md",
                       ROOT / "paper/supplement/tables/tab_crc_repeated.tex"]}))
    print(summary[["n_cal", "alpha", "mean_unit_loss_mean", "FA0_mean", "alarm_all_fraction"]].to_string(index=False))


def verify(reproduce):
    check_sources()
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["output_sha256"].items():
        assert sha(ROOT / name) == expected, name
    init_worker()
    checked = 0
    for directory in sorted((OUT / "run").iterdir()):
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        rep, n = metadata["repetition"], metadata["n_cal"]
        part = PARTITIONS[(PARTITIONS.repetition == rep) & (PARTITIONS.n_cal == n)]
        assert len(part) == 65 and part.Serial_Number.is_unique
        assert part.loc[part.role.eq("train") & part.critical, "Serial_Number"].nunique() == 14 - n
        scores = pd.read_csv(directory / "scores.csv", float_precision="round_trip")
        assert scores.Experiment_Index.is_unique
        for role in ["calibration", "evaluation"]:
            expected = FEATURES[FEATURES.Serial_Number.isin(part.loc[part.role.eq(role), "Serial_Number"])]
            actual = scores[scores.role.eq(role)]
            pd.testing.assert_frame_equal(actual[META_COLS].reset_index(drop=True),
                                          expected[META_COLS].reset_index(drop=True))
        prob_cols = [c for c in scores if c.startswith("P_")]
        np.testing.assert_allclose(scores[prob_cols].sum(axis=1), 1, atol=1e-12)
        predicted = np.array([c[2:] for c in prob_cols])[scores[prob_cols].to_numpy().argmax(axis=1)]
        assert np.array_equal(predicted, scores.base_prediction)
        np.testing.assert_allclose(scores.severity3_score,
            sum(scores[c] if c in scores else np.zeros(len(scores)) for c in ["P_F03", "P_F06"]), atol=1e-15)
        curve = calibration_curve(scores[scores.role.eq("calibration")])
        saved_curve = pd.read_csv(directory / "calibration_curve.csv", float_precision="round_trip")
        np.testing.assert_allclose(curve.to_numpy(), saved_curve.to_numpy(), atol=1e-15)
        results = pd.read_csv(directory / "results.csv", float_precision="round_trip")
        for _, row in results.iterrows():
            if pd.notna(row.alpha):
                eligible = np.flatnonzero(curve.corrected_criterion.to_numpy() <= row.alpha)
                idx = int(eligible[0]) if len(eligible) else len(curve) - 1
                assert row.empty_set == (len(eligible) == 0)
                assert row.threshold == curve.threshold.iloc[idx]
            else:
                assert row.threshold == np.inf
            metrics, units = evaluate(scores[scores.role.eq("evaluation")], row.threshold)
            for k, value in metrics.items():
                np.testing.assert_allclose(value, row[k], atol=1e-15)
            saved_units = pd.read_csv(directory / "unit_results.csv")
            select = saved_units.alpha.isna() if pd.isna(row.alpha) else saved_units.alpha.eq(row.alpha)
            pd.testing.assert_frame_equal(units.reset_index(drop=True),
                                          saved_units.loc[select, units.columns].reset_index(drop=True))
        checked += 1
    reproduction = []
    if reproduce:
        for n in CONFIG["n_cal"]:
            destination = OUT / "verification_reproduction" / f"n{n}_r000"
            fit_partition((0, n, str(destination)))
            original = OUT / "run" / f"n{n}_r000"
            a = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
            b = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            assert a["model_sha256"] == b["model_sha256"]
            assert a["training_features_sha256"] == b["training_features_sha256"]
            for name in ["scores.csv", "results.csv", "imputation.csv", "calibration_curve.csv", "unit_results.csv"]:
                assert sha(original / name) == sha(destination / name), name
            reproduction.append(n)
    write_json(OUT / "verification.json", dict(hashes_checked=len(manifest["output_sha256"]),
        partitions_reconstructed=checked, models_reproduced=reproduction,
        all_checks_passed=True, run_manifest_sha256=sha(OUT / "run_manifest.json")))
    print(f"Verified {checked} partitions, {len(manifest['output_sha256'])} hashes; reproduced {reproduction}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "summarise", "verify"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "run":
        run(args.workers, args.limit)
    elif args.mode == "summarise":
        summarise()
    else:
        verify(args.reproduce)
