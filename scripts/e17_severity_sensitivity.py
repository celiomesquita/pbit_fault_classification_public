"""E17: sensitivity of the conclusions to the set of critical classes (docs/protocolos/E17.md).

Endpoint sensitivity with fixed models: saved predictions and probabilities are recounted under
alternative critical sets; no model is refitted and no threshold or cost ratio is reselected.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pbit.modeling import FAULT_ORDER  # noqa: E402

PROTOCOL = ROOT / "docs/protocolos/E17.md"
OUT = ROOT / "data/e17_severity_v1"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
RESAMPLES, RESAMPLE_SEED = 10_000, 20261031
MAPS = {"M0": ["F03", "F06"], "M1": ["F03", "F05", "F06"], "M2": ["F02", "F03", "F06"],
        "M3": ["F02", "F03", "F04", "F05", "F06"], "M4": ["F06"]}
NORMAL = ["N01", "N02"]
CIDX = {c: i for i, c in enumerate(FAULT_ORDER)}
INPUTS: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    INPUTS[path.relative_to(ROOT).as_posix()] = sha(path)
    return pd.read_csv(path)


def masks(critical):
    crit = np.array([c in critical for c in FAULT_ORDER])
    norm = np.array([c in NORMAL for c in FAULT_ORDER])
    return crit, norm


def unit_tensor(frame: pd.DataFrame, column: str, units: list[str]) -> np.ndarray:
    """(G, 8, 8) confusion counts per serial number (rows: truth, columns: prediction)."""
    uidx = {u: i for i, u in enumerate(units)}
    t = np.zeros((len(units), 8, 8))
    np.add.at(t, (frame.Serial_Number.map(uidx).to_numpy(), frame.Fault_Mode.map(CIDX).to_numpy(),
                  frame[column].map(CIDX).to_numpy()), 1)
    return t


def metrics(t: np.ndarray, w: np.ndarray, critical) -> dict[str, np.ndarray]:
    """CMR, unit-average loss and FA0 for weights w (D, G) under critical set `critical`."""
    crit, norm = masks(critical)
    crit_n = t[:, crit, :].sum(axis=(1, 2))
    miss = t[:, crit][:, :, norm].sum(axis=(1, 2))
    norm_n = t[:, norm, :].sum(axis=(1, 2))
    alarm = t[:, norm][:, :, ~norm].sum(axis=(1, 2))
    has_c = crit_n > 0
    loss = np.where(has_c, miss / np.maximum(crit_n, 1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"CMR": w @ miss / (w @ crit_n), "Lbar": w @ loss / (w @ has_c),
                "FA0": w @ alarm / (w @ norm_n)}


class Source:
    """Predictions of several arms over fixed seeds; seeds share the resampled units."""

    def __init__(self, frames: dict, units: list[str], weights: np.ndarray):
        self.frames, self.units, self.weights = frames, units, weights
        self.cache: dict = {}

    def arm(self, column, critical):
        key = (column, tuple(critical))
        if key not in self.cache:
            point, boot = [], []
            for frame in self.frames.values():
                t = unit_tensor(frame, column, self.units)
                point.append(metrics(t, np.ones((1, len(self.units))), critical))
                boot.append(metrics(t, self.weights, critical))
            self.cache[key] = ({k: float(np.mean([p[k][0] for p in point])) for k in point[0]},
                               {k: np.nanmean(np.stack([b[k] for b in boot]), axis=0) for k in boot[0]})
        return self.cache[key]


def interval(values):
    v = values[np.isfinite(values)]
    lo, hi = np.quantile(v, [0.025, 0.975])
    return float(lo), float(hi), 1 - len(v) / len(values)


def contrast(source, a, b, metric, critical):
    pa, ba = source.arm(a, critical)
    pb, bb = source.arm(b, critical)
    lo, hi, discard = interval(ba[metric] - bb[metric])
    return dict(A=a, B=b, metric=metric, estimate=pa[metric] - pb[metric], low=lo, high=hi,
                discard=discard)


def by_seed(path_pattern, units_from=None):
    return {seed: read(ROOT / path_pattern.format(seed=seed)) for seed in SEEDS}


def stage_a(weights_for):
    rows = []
    # RQ1: leakage (random minus grouped), same cycles in both arms
    frames = by_seed("data/leakage/preds_seed{seed}.csv")
    leak = {}
    for seed, f in frames.items():
        g = f[f.arm.eq("grouped_faultmode")].set_index("Experiment_Index")
        r = f[f.arm.eq("random_stratified")].set_index("Experiment_Index").loc[g.index]
        g = g.reset_index()
        for model in ("rf", "lgbm"):
            g[f"{model}_random"] = r[model].to_numpy()
        leak[seed] = g
    units = sorted(leak[42].Serial_Number.unique())
    src = Source(leak, units, weights_for(units))
    for name, critical in MAPS.items():
        for model in ("rf", "lgbm"):
            rows.append(dict(stage="A1_leakage", mapping=name, **contrast(src, f"{model}_random", model, "CMR", critical)))
    # RQ2: design choices (rf/lgbm F246 come from the ablation run, as in the manuscript)
    abl, fam, imb = (by_seed(f"data/{s}/preds_seed{{seed}}.csv") for s in ("ablation", "families", "imbalance"))
    for seed in SEEDS:
        for column in ("rf_F246", "lgbm_F246"):
            fam[seed][column] = abl[seed].set_index("Experiment_Index").loc[fam[seed].Experiment_Index, column].to_numpy()
    units = sorted(abl[42].Serial_Number.unique())
    w = weights_for(units)
    sources = {"families": Source(fam, units, w), "ablation": Source(abl, units, w), "imbalance": Source(imb, units, w)}
    for name, critical in MAPS.items():
        for column in ("rf_F246", "gb_F246", "lgbm_F246", "xgb_F246"):
            p, _ = sources["families"].arm(column, critical)
            rows.append(dict(stage="A2_family_level", mapping=name, A=column, metric="CMR", estimate=p["CMR"]))
        for other in ("lgbm_F246", "xgb_F246"):
            rows.append(dict(stage="A2_family", mapping=name, **contrast(sources["families"], "rf_F246", other, "CMR", critical)))
        for a, b in (("lgbm_F246", "lgbm_F115"), ("lgbm_F246", "lgbm_F18"), ("lgbm_F246", "B4_threshold_safety")):
            rows.append(dict(stage="A2_features", mapping=name, **contrast(sources["ablation"], a, b, "CMR", critical)))
        p, _ = sources["ablation"].arm("B4_threshold_safety", critical)
        rows.append(dict(stage="A2_B4_level", mapping=name, A="B4_threshold_safety", metric="CMR", estimate=p["CMR"]))
        rows.append(dict(stage="A2_imbalance", mapping=name,
                         **contrast(sources["imbalance"], "lgbm_freq_sev_gaussian", "lgbm_freq_gaussian", "CMR", critical)))
    # RQ2: historical operating point (single seed, nested outer folds)
    nested = pd.concat([read(ROOT / f"data/nested_cv/preds_cmr_rel_outer{k}.csv") for k in range(5)], ignore_index=True)
    units = sorted(nested.Serial_Number.unique())
    src = Source({42: nested}, units, weights_for(units))
    for name, critical in MAPS.items():
        for column in ("pred_tuned", "pred_untuned"):
            p, _ = src.arm(column, critical)
            rows.append(dict(stage="A3_operating_point", mapping=name, A=column, metric="CMR", estimate=p["CMR"]))
            rows.append(dict(stage="A3_operating_point", mapping=name, A=column, metric="FA0", estimate=p["FA0"]))
        rows.append(dict(stage="A3_contrast", mapping=name, **contrast(src, "pred_tuned", "pred_untuned", "CMR", critical)))
    # RQ2: policies selected on inner data (E1), pooled outer predictions per seed
    manifest = json.loads((ROOT / "data/e1_full_v1/completion_manifest.json").read_text())
    INPUTS["data/e1_full_v1/completion_manifest.json"] = sha(ROOT / "data/e1_full_v1/completion_manifest.json")
    parts: dict = {}
    for item in manifest["completed"]:
        parts.setdefault(item["seed"], []).append(read(ROOT / item["output"] / "external_predictions.csv"))
    e1 = {s: pd.concat(v, ignore_index=True) for s, v in parts.items()}
    units = sorted(e1[SEEDS[0]].Serial_Number.unique())
    src = Source(e1, units, weights_for(units))
    for name, critical in MAPS.items():
        for budget in ("b30", "b40", "b50"):
            for arm in ("CMR_fixed", "CMR_optimized", "macro_F1_fixed", "macro_F1_optimized", "minimum_cost", "metacost"):
                for metric in ("CMR", "Lbar", "FA0"):
                    rows.append(dict(stage="A4_inner_policies", mapping=name, budget=budget,
                                     **contrast(src, f"{arm}_{budget}", f"untuned_{budget}", metric, critical)))
    # RQ3: held-out test
    test = read(ROOT / "data/final_test_predictions.csv")
    for name, critical in MAPS.items():
        for column in ("pred_main_F1-tuned", "pred_main_CMR-tuned"):
            c = test[test.Fault_Mode.isin(critical)]
            miss_units = c.assign(miss=c[column].isin(NORMAL)).groupby("Serial_Number").miss.any()
            n, k = len(miss_units), int(miss_units.sum())
            upper = float(beta.ppf(0.975, k + 1, n - k)) if n > k else 1.0
            rows.append(dict(stage="A5_test", mapping=name, A=column, metric="units_with_miss",
                             estimate=k, n_units=n, critical_cycles=len(c),
                             misses=int(c[column].isin(NORMAL).sum()), upper95=upper))
    # RQ3: fully nested leave-one-unit-out of F06 (E5)
    runs = sorted((ROOT / "data/e5_nested_loso/runs").glob("*/predictions.csv"))
    loso = pd.concat([read(p).assign(run=p.parent.name) for p in runs], ignore_index=True)
    loso["target"] = loso.run.str.split("_").str[0].str.replace("SN", "SN ", regex=False)
    f06 = loso[loso.Fault_Mode.eq("F06") & loso.Serial_Number.eq(loso.target)]
    for name, critical in MAPS.items():
        if "F06" not in critical:
            continue
        for column in ("F1_tuned", "CMR_tuned", "untuned"):
            share = f06.assign(hit=f06[column].isin(critical)).groupby("Serial_Number").hit.mean()
            for unit, value in share.items():
                rows.append(dict(stage="A6_nested_loso", mapping=name, A=column, unit=unit,
                                 metric="F06_predicted_critical", estimate=float(value)))
    return pd.DataFrame(rows)


def stage_b():
    manifest = read(ROOT / "data/ress_revision/freeze_v1/cycle_manifest.csv")
    audited = manifest[manifest.in_D_primary]
    test_units = set(read(ROOT / "data/ress_revision/freeze_v1/historical_test_primary.csv").Serial_Number)
    rows = []
    for name, critical in MAPS.items():
        c = audited[audited.Fault_Mode.isin(critical)]
        units = set(c.Serial_Number)
        rows.append(dict(mapping=name, critical=" ".join(critical), critical_cycles=len(c),
                         critical_units=len(units), training_units=len(units - test_units),
                         test_units=len(units & test_units), crc_minimum_alpha_005=19,
                         binomial_units_upper_005=72, meets_crc_minimum=len(units) >= 19,
                         training_meets_crc_minimum=len(units - test_units) >= 19))
    return pd.DataFrame(rows)


def calibration_curve(cal, critical):
    c = cal[cal.Fault_Mode.isin(critical)]
    if c.empty:
        return None
    candidates = np.r_[np.inf, np.sort(c.score.unique())[::-1], 0.0]
    final_alarm = c.score.to_numpy()[:, None] >= candidates
    normal_base = c.base_prediction.isin(NORMAL).to_numpy()[:, None]
    miss = ~final_alarm & normal_base
    units = sorted(c.Serial_Number.unique())
    losses = np.stack([miss[c.Serial_Number.eq(u).to_numpy()].mean(axis=0) for u in units])
    n = len(units)
    return candidates, (losses.sum(axis=0) + 1) / (n + 1)


def evaluate(e, threshold, critical):
    alarm = e.score >= threshold
    final_normal = ~alarm & e.base_prediction.isin(NORMAL)
    final_fault = alarm | ~e.base_prediction.isin(NORMAL)
    crit, normal = e.Fault_Mode.isin(critical), e.Fault_Mode.isin(NORMAL)
    units = e.assign(miss=crit & final_normal, crit=crit).groupby("Serial_Number").agg(c=("crit", "sum"), m=("miss", "sum"))
    units = units[units.c > 0]
    loss = float((units.m / units.c).mean()) if len(units) else np.nan
    fa0 = float((normal & final_fault).sum() / normal.sum()) if normal.any() else np.nan
    return loss, fa0, len(units)


def stage_c():
    """CRC with the fixed E6 models: M0 (reproduction) and M4 only, as fixed in the protocol."""
    freeze = ROOT / "data/crc_repeated_partitions/protocol_freeze.json"
    INPUTS[freeze.relative_to(ROOT).as_posix()] = sha(freeze)
    rows = []
    for folder in sorted((ROOT / "data/crc_repeated_partitions/run").iterdir()):
        scores = pd.read_csv(folder / "scores.csv")
        n_cal, rep = int(folder.name.split("_")[0][1:]), int(folder.name.split("_r")[1])
        for name in ("M0", "M4"):
            critical = MAPS[name]
            # A class absent from a partition's training set has no probability column: P = 0.
            present = [f"P_{c}" for c in critical if f"P_{c}" in scores]
            s = scores.assign(score=scores[present].sum(axis=1) if present else 0.0)
            if name == "M0":
                assert np.allclose(s.score, s.severity3_score, atol=1e-12)
                s["score"] = s.severity3_score
            curve = calibration_curve(s[s.role.eq("calibration")], critical)
            for alpha in (0.10, 0.125, 0.15, 0.20):
                if curve is None:
                    threshold, n_crit_cal, empty = 0.0, 0, True
                else:
                    candidates, criterion = curve
                    eligible = np.flatnonzero(criterion <= alpha)
                    empty = len(eligible) == 0
                    threshold = 0.0 if empty else float(candidates[eligible[0]])
                    n_crit_cal = s[s.role.eq("calibration") & s.Fault_Mode.isin(critical)].Serial_Number.nunique()
                loss, fa0, n_eval = evaluate(s[s.role.eq("evaluation")], threshold, critical)
                rows.append(dict(mapping=name, n_cal=n_cal, repetition=rep, alpha=alpha,
                                 critical_calibration_units=n_crit_cal, critical_evaluation_units=n_eval,
                                 empty_set=empty, alarm_all=threshold == 0.0, mean_unit_loss=loss, FA0=fa0))
    return pd.DataFrame(rows)


REFERENCE = [  # published M0 values (main text), checked before any alternative mapping is read
    ("A2_family_level", dict(A="rf_F246"), 0.071), ("A2_family_level", dict(A="gb_F246"), 0.071),
    ("A2_family_level", dict(A="lgbm_F246"), 0.081),
    ("A2_family", dict(B="lgbm_F246"), -0.010), ("A2_family", dict(B="xgb_F246"), -0.005),
    ("A1_leakage", dict(A="rf_random"), -0.022),
    ("A2_features", dict(B="lgbm_F115"), -0.002), ("A2_features", dict(B="lgbm_F18"), -0.049),
    ("A2_features", dict(B="B4_threshold_safety"), -0.091), ("A2_B4_level", dict(), 0.172),
    ("A2_imbalance", dict(), -0.002),
]


def check_reproduction(a, c):
    m0 = a[a.mapping.eq("M0")]
    failures = []
    for stage, keys, value in REFERENCE:
        sel = m0[m0.stage.eq(stage)]
        for k, v in keys.items():
            sel = sel[sel[k].eq(v)]
        got = float(sel.estimate.iloc[0])
        if abs(got - value) > 0.0005 + 1e-12:
            failures.append((stage, keys, value, got))
    op = m0[m0.stage.eq("A3_operating_point") & m0.metric.eq("CMR")].set_index("A").estimate
    if abs(op["pred_tuned"] - 0.030) > 0.0006 or abs(op["pred_untuned"] - 0.070) > 0.0006:
        failures.append(("A3", op.to_dict()))
    e1 = m0[m0.stage.eq("A4_inner_policies") & m0.budget.eq("b40") & m0.A.str.startswith("CMR_")]
    got = sorted(round(-x, 3) for x in e1[e1.metric.eq("CMR")].estimate)
    if got != [0.052, 0.065]:
        failures.append(("A4 CMR", got))
    got = sorted(round(-x, 3) for x in e1[e1.metric.eq("Lbar")].estimate)
    if got != [0.088, 0.107]:
        failures.append(("A4 Lbar", got))
    test = m0[m0.stage.eq("A5_test")].set_index("A")
    if round(test.loc["pred_main_CMR-tuned", "upper95"], 2) != 0.84 or round(test.loc["pred_main_F1-tuned", "upper95"], 2) != 0.99:
        failures.append(("A5", test.upper95.to_dict()))
    loso = m0[m0.stage.eq("A6_nested_loso") & m0.unit.eq("SN 045")].set_index("A").estimate
    if round(loso["F1_tuned"], 2) != 0.35 or round(loso["CMR_tuned"], 2) != 0.65:
        failures.append(("A6", loso.to_dict()))
    crc = c[c.mapping.eq("M0")].groupby(["n_cal", "alpha"]).mean_unit_loss.mean()
    for (n, alpha), value in {(7, 0.125): 0.005, (9, 0.10): 0.003, (11, 0.10): 0.027}.items():
        if round(crc.loc[(n, alpha)], 3) != value:
            failures.append(("C", n, alpha, crc.loc[(n, alpha)]))
    return failures


def main():
    assert sha(PROTOCOL) == "ccfbff8f4144183631c06b1751bfb63497b259c56dc381709e36eed5a97a4c9d", "protocol changed"
    OUT.mkdir(parents=True, exist_ok=True)
    weights_cache: dict = {}

    def weights_for(units):
        key = tuple(units)
        if key not in weights_cache:
            rng = np.random.default_rng(RESAMPLE_SEED)
            weights_cache[key] = rng.multinomial(len(units), np.ones(len(units)) / len(units), size=RESAMPLES).astype(float)
        return weights_cache[key]

    a = stage_a(weights_for)
    b = stage_b()
    c = stage_c()
    failures = check_reproduction(a, c)
    (OUT / "freeze.json").write_text(json.dumps({"protocol": "docs/protocolos/E17.md", "protocol_sha256": sha(PROTOCOL),
                                                 "mappings": MAPS, "resamples": RESAMPLES, "resample_seed": RESAMPLE_SEED,
                                                 "seeds": SEEDS}, indent=2), encoding="utf-8")
    if failures:
        (OUT / "reproduction_failure.json").write_text(json.dumps(failures, indent=2, default=str), encoding="utf-8")
        raise SystemExit(f"M0 reproduction failed; protocol requires interruption: {failures}")
    a.to_csv(OUT / "stage_a.csv", index=False)
    b.to_csv(OUT / "stage_b.csv", index=False)
    c.to_csv(OUT / "stage_c.csv", index=False)
    manifest = {"protocol_sha256": sha(PROTOCOL), "input_sha256": INPUTS,
                "code_sha256": {"scripts/e17_severity_sensitivity.py": sha(Path(__file__))},
                "output_sha256": {p.name: sha(p) for p in sorted(OUT.glob("stage_*.csv"))},
                "m0_reproduction": "passed",
                "implementation_notes": ["First run stopped at the M0 check: A3 read preds_cmr_outer* (original balanced-accuracy constraint, CMR 0.055) instead of preds_cmr_rel_outer* (amended constraint reported in the manuscript, CMR 0.030). Source path corrected before any alternative mapping was inspected.", "E6 partitions whose training lacked a class have no probability column for it; its probability is taken as zero."]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("M0 reproduction passed;", len(a), "stage-A rows,", len(c), "stage-C rows")


if __name__ == "__main__":
    main()
