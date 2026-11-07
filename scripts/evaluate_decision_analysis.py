"""Protocols P-B2, P-B3 and P-B4: cost of the operating points, item-level decisions and alarm rate by condition at 20%.

Reads the held-out predictions of the headline procedure (without F02, set E allowed). Costs: a missed failure costs r and a false
alarm costs 1; the secondary variant doubles the cost of missing F03 or F06. Items are maximal runs of consecutive cycles
(experimento_idx order) of the same unit and condition, an approximation of a removal event.
Usage, from scripts/: py evaluate_decision_analysis.py. Outputs in resultados/: cost_curve.csv, cost_rstar.csv,
cost_inner_optimal.csv, item_blocks.csv, item_level.csv, binary_no_f02_spectral_alarm_rate_by_condition_at_20.csv.
"""
import ast

import numpy as np
import pandas as pd

import evaluate_robustness_procedure as robustness
import experimentos_base as base
import train_binary_model as model

INPUT = base.RAIZ / "resultados"
N_BOOT = 1000
R_GRID = (0.5, 1, 2, 3, 5, 10, 20, 50, 100)
R_INNER = (1, 2, 5, 10, 20, 50, 100)
POINTS = ("default", "05", "10", "20")


def procedure_predictions():
    saved = pd.read_csv(INPUT / "binary_no_f02_spectral_predictions.csv")
    chosen = pd.read_csv(INPUT / "binary_no_f02_spectral_procedure_choices.csv")
    keys = chosen[["repetition", "fold", "feature_set", "model"]]
    return saved.merge(keys, on=list(keys.columns)), chosen


def alarm(frame, point):
    return frame["score"].values >= 0.5 if point == "default" else frame["score"].values > frame[f"threshold_{point}"].values


def counts(frame, alarms, weights):
    y = frame["y"].values == 1
    critical = np.isin(frame["condicao"].values, model.CRITICAL)
    return {"miss": (weights * (y & ~alarms)).sum(), "miss_critical": (weights * (critical & ~alarms)).sum(),
            "false_alarm": (weights * (~y & alarms)).sum(), "n_failure": (weights * y).sum(), "n_normal": (weights * ~y).sum(),
            "n": weights.sum()}


def r_star(frame, weights):
    """Cost ratio above which the 20% target costs less than the 10% target (per-cycle costs)."""
    c10, c20 = counts(frame, alarm(frame, "10"), weights), counts(frame, alarm(frame, "20"), weights)
    extra_alarms = c20["false_alarm"] - c10["false_alarm"]
    saved = c10["miss"] - c20["miss"]
    saved_weighted = saved + (c10["miss_critical"] - c20["miss_critical"])
    return {"r_star": extra_alarms / saved if saved > 0 else np.inf,
            "r_star_critical_double": extra_alarms / saved_weighted if saved_weighted > 0 else np.inf}


def unit_draws(frame, rng):
    units = sorted(frame["numero_serie"].unique())
    position = frame["numero_serie"].map({unit: i for i, unit in enumerate(units)}).values
    for draw in rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=N_BOOT):
        yield draw[position].astype(float)


def cost_analysis(procedure, rng):
    ones = np.ones(len(procedure))
    rows = []
    for point in POINTS:
        c = counts(procedure, alarm(procedure, point), ones)
        for r in R_GRID:
            rows.append({"point": point, "r": r, "cost_per_cycle": (r * c["miss"] + c["false_alarm"]) / c["n"],
                         "cost_per_cycle_critical_double": (r * (c["miss"] + c["miss_critical"]) + c["false_alarm"]) / c["n"]})
    curve = pd.DataFrame(rows)
    point = r_star(procedure, ones)
    samples = pd.DataFrame([r_star(procedure, w) for w in unit_draws(procedure, rng)])
    rstar = pd.DataFrame([{"metric": k, "estimate": v, "ci_low": samples[k].quantile(0.025), "ci_high": samples[k].quantile(0.975),
                           "share_draws_below_10": (samples[k] < 10).mean()} for k, v in point.items()])
    return curve, rstar


def best_threshold(scores, y, r):
    """Threshold minimising r * misses + false alarms on inner out-of-fold scores (alarm when score > threshold)."""
    candidates = np.concatenate([[-np.inf], np.unique(scores)])
    order = np.argsort(scores)
    s, t = scores[order], y[order]
    # with threshold = candidate k, cycles with score <= candidate are not alarmed
    below_fail = np.searchsorted(s, candidates, side="right")
    cum_fail = np.concatenate([[0], np.cumsum(t)])
    misses = cum_fail[below_fail]
    false_alarms = (t == 0).sum() - (below_fail - misses)
    return candidates[np.argmin(r * misses + false_alarms)]


def inner_optimal(procedure, chosen, rng):
    df_all = pd.read_csv(base.RAIZ / "dados" / "pbit_features_v3.csv").merge(
        pd.read_csv(base.RAIZ / "dados" / "pbit_spectral_features_v3.csv"), on="experimento_idx", validate="one_to_one")
    df = df_all[df_all["condicao"] != "F02"].reset_index(drop=True)
    conditions, groups = df["condicao"].values, df["numero_serie"].values
    y = model.make_target(conditions)
    X = model.build_feature_sets(df)["E"].values
    parts, gap = [], 0.0
    for row in chosen.itertuples():
        train = base.dobras(df, row.repetition)["E3b"] != row.fold
        seed = 1000 * row.repetition + row.fold
        oof = robustness.inner_scores(row.model, ast.literal_eval(row.params), X[train], y[train], conditions[train], groups[train], seed)
        held = procedure[(procedure.repetition == row.repetition) & (procedure.fold == row.fold)].copy()
        for alpha in model.ALPHAS:
            gap = max(gap, abs(np.quantile(oof[y[train] == 0], 1 - alpha) - held[f"threshold_{int(alpha * 100):02d}"].iloc[0]))
        for r in R_INNER:
            held[f"threshold_r{r}"] = best_threshold(oof, y[train], r)
        parts.append(held)
    assert gap < 1e-9, f"recomputed inner thresholds differ from saved ones: {gap}"
    held = pd.concat(parts, ignore_index=True)
    rows = []
    for r in R_INNER:
        a = held["score"].values > held[f"threshold_r{r}"].values
        point = summary(held, a, np.ones(len(held)), r)
        samples = pd.DataFrame([summary(held, a, w, r) for w in unit_draws(held, rng)])
        rows += [{"r": r, "metric": k, "estimate": v, "ci_low": samples[k].quantile(0.025), "ci_high": samples[k].quantile(0.975)} for k, v in point.items()]
        for target in ("10", "20"):
            rows.append({"r": r, "metric": f"cost_per_cycle_target_{target}", "estimate": summary(held, alarm(held, target), np.ones(len(held)), r)["cost_per_cycle"],
                         "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows), gap


def summary(frame, alarms, weights, r):
    c = counts(frame, alarms, weights)
    critical = np.isin(frame["condicao"].values, model.CRITICAL)
    return {"recall": 1 - c["miss"] / c["n_failure"], "false_alarm": c["false_alarm"] / c["n_normal"],
            "critical_miss": c["miss_critical"] / (weights * critical).sum(), "cost_per_cycle": (r * c["miss"] + c["false_alarm"]) / c["n"]}


def item_blocks(procedure):
    order = procedure[["experimento_idx", "numero_serie", "condicao"]].drop_duplicates().sort_values("experimento_idx")
    key = order["numero_serie"] + "|" + order["condicao"]
    order["item"] = (key != key.shift()).cumsum()
    return order


def item_level(procedure, blocks, rng):
    frame = procedure.merge(blocks[["experimento_idx", "item"]], on="experimento_idx", validate="many_to_one")
    rows = []
    for point in ("default", "10", "20"):
        frame["alarm"] = alarm(frame, point)
        threshold = 0.5 if point == "default" else None
        grouped = frame.groupby(["repetition", "item"])
        items = grouped.agg(numero_serie=("numero_serie", "first"), condicao=("condicao", "first"), y=("y", "first"),
                            median_score=("score", "median"), share_alarm=("alarm", "mean"),
                            threshold=(f"threshold_{point}" if point != "default" else "score", "first"))
        if threshold is not None:
            items["threshold"] = threshold
        decisions = {"median": items["median_score"] > items["threshold"] if point != "default" else items["median_score"] >= 0.5,
                     "majority": items["share_alarm"] >= 0.5}
        for rule, a in decisions.items():
            a = a.values
            point_est = item_summary(items, a, np.ones(len(items)))
            samples = pd.DataFrame([item_summary(items, a, w) for w in unit_draws(items, rng)])
            rows += [{"point": point, "rule": rule, "metric": k, "estimate": v, "ci_low": samples[k].quantile(0.025),
                      "ci_high": samples[k].quantile(0.975)} for k, v in point_est.items()]
    return pd.DataFrame(rows)


def item_summary(items, alarms, weights):
    y = items["y"].values == 1
    cond = items["condicao"].values
    critical = np.isin(cond, model.CRITICAL)

    def rate(mask, values):
        total = (weights * mask).sum()
        return (weights * mask * values).sum() / total if total > 0 else np.nan

    return {"recall": rate(y, alarms), "false_alarm": rate(~y, alarms), "false_alarm_n01": rate(cond == "N01", alarms),
            "false_alarm_n02": rate(cond == "N02", alarms), "critical_miss": rate(critical, ~alarms)}


def main():
    pd.set_option("display.width", 220, "display.max_rows", 200)
    output = base.SAIDA
    output.mkdir(exist_ok=True)
    procedure, chosen = procedure_predictions()
    assert len(procedure) == 5 * procedure["experimento_idx"].nunique()

    curve, rstar = cost_analysis(procedure, np.random.default_rng(2026))
    curve.to_csv(output / "cost_curve.csv", index=False, lineterminator="\n")
    rstar.to_csv(output / "cost_rstar.csv", index=False, lineterminator="\n")
    print(curve.pivot(index="r", columns="point", values="cost_per_cycle").round(4).to_string())
    print(rstar.round(3).to_string(index=False), flush=True)

    inner, gap = inner_optimal(procedure, chosen, np.random.default_rng(2026))
    inner.to_csv(output / "cost_inner_optimal.csv", index=False, lineterminator="\n")
    print(f"inner threshold check max gap {gap:.2e}")
    print(inner.pivot(index="r", columns="metric", values="estimate").round(3).to_string(), flush=True)

    blocks = item_blocks(procedure)
    blocks.to_csv(output / "item_blocks.csv", index=False, lineterminator="\n")
    print(f"items: {blocks['item'].nunique()}; by condition:", blocks.groupby("condicao")["item"].nunique().to_dict())
    items = item_level(procedure, blocks, np.random.default_rng(2026))
    items.to_csv(output / "item_level.csv", index=False, lineterminator="\n")
    view = items.assign(text=items.estimate.round(3).astype(str) + " [" + items.ci_low.round(3).astype(str) + "; " + items.ci_high.round(3).astype(str) + "]")
    print(view.pivot(index=["point", "rule"], columns="metric", values="text").to_string())

    rates = procedure.assign(alarm=procedure["score"] > procedure["threshold_20"]).groupby("condicao")["alarm"].mean().round(3)
    rates.to_frame("procedure").T.to_csv(output / "binary_no_f02_spectral_alarm_rate_by_condition_at_20.csv", lineterminator="\n")
    print(rates.to_dict())


if __name__ == "__main__":
    main()
