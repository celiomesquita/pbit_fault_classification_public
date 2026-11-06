"""Train and evaluate the normal-versus-failure model (plan actions P12 to P14) on data version 3.

The serial number is used only to keep units apart between training and test sets.
Usage, from scripts/: py train_binary_model.py [--rapido] [--no-f02] [--spectral]. Outputs go to resultados/binary_*
or, with --no-f02, resultados/binary_no_f02_*. --no-f02 assumes that visual inspection removes F02 items before PBIT.
"""
import ast
import sys
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import experimentos_base as base

EXCLUDE_F02 = "--no-f02" in sys.argv
USE_SPECTRAL = "--spectral" in sys.argv
PREFIX = ("binary_no_f02" if EXCLUDE_F02 else "binary") + ("_spectral" if USE_SPECTRAL else "")
NORMAL = ("N01", "N02")
CRITICAL = ("F03", "F06")
ALPHAS = (0.05, 0.10, 0.20)
LOG_COLUMNS = ["desvio_padrao", "pico_a_pico", "maior_salto", "desvio_primeira_metade", "desvio_segunda_metade"]
LEVEL_COLUMNS = ["media", "mediana", "minimo", "maximo", "rms", "valor_inicial", "valor_final", "energia"]
SIMPLICITY_ORDER = ["tree", "logistic", "forest", "boosting"]
SET_ORDER = ["D", "B", "A", "E"]
SIMPLICITY_MARGIN = 0.01
N_PERMUTATIONS = 3 if base.RAPIDO else 200
N_BOOT = 100 if base.RAPIDO else 1000

GRIDS = {
    "tree": [{"max_depth": depth} for depth in (3, 4, 5)],
    "logistic": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "forest": [{"min_samples_leaf": leaf} for leaf in (1, 3, 5)],
    "boosting": [{"max_depth": depth} for depth in (2, 3, 4)],
}
if base.RAPIDO:
    GRIDS = {name: grid[:1] for name, grid in GRIDS.items()}


class SequentialPredictForest(RandomForestClassifier):
    """Fits in parallel but sums tree votes in a fixed order: parallel summation changes the last digits between runs."""

    def predict_proba(self, X):
        workers = self.n_jobs
        self.n_jobs = 1
        try:
            return super().predict_proba(X)
        finally:
            self.n_jobs = workers


def make_target(conditions):
    return (~pd.Series(conditions).isin(NORMAL)).astype(int).values


def half_mean_difference(df):
    """Absolute difference between half-window means, recovered from the 61/62 split deviations."""
    variance = df["desvio_padrao"] ** 2 - (61 * df["desvio_primeira_metade"] ** 2 + 62 * df["desvio_segunda_metade"] ** 2) / 123
    return np.sqrt(variance.clip(lower=0) * 123**2 / (61 * 62))


def build_feature_sets(df):
    """A: all 18 features. B: without absolute level. D: every cycle compared only with its own mean; no unit information."""
    spectral = [column for column in df.columns if column.startswith(("spectral_", "lag1_", "lag2_"))]
    features = [column for column in df.columns[3:] if column not in spectral]
    logged = df[features].copy()
    for column in LOG_COLUMNS:
        logged[column] = np.log10(logged[column])
    level = df["media"].abs()
    relative = pd.DataFrame({
        "log_sd_ratio": np.log10(df["desvio_padrao"] / level),
        "log_peak_to_peak_ratio": np.log10(df["pico_a_pico"] / level),
        "log_largest_jump_ratio": np.log10(df["maior_salto"] / level),
        "log_first_half_sd_ratio": np.log10(df["desvio_primeira_metade"] / level),
        "log_second_half_sd_ratio": np.log10(df["desvio_segunda_metade"] / level),
        "start_end_ratio": df["delta_inicio_fim"] / level,
        "slope_ratio": df["inclinacao"] * 122 / level,
        "half_difference_ratio": half_mean_difference(df) / level,
        "minimum_ratio": df["minimo"] / level,
        "maximum_ratio": df["maximo"] / level,
        "skewness": df["assimetria"],
        "kurtosis": df["curtose"],
        "peak_count": df["n_picos"],
    })
    sets = {
        "A": logged,
        "B": logged[[column for column in features if column not in LEVEL_COLUMNS]],
        "D": relative,
    }
    if spectral:
        sets["E"] = pd.concat([relative, df[spectral]], axis=1)
    return sets


def make_model(name, params, seed):
    if name == "tree":
        return DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=10, random_state=seed, **params)
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000, **params))
    if name == "forest":
        return SequentialPredictForest(200, class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **params)
    return HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, learning_rate=0.05, max_iter=200, random_state=seed, **params)


def inner_fit(name, X, y, conditions, groups, seed):
    """Choose hyperparameters by AUC on inner folds grouped by unit; return the out-of-fold scores of the winner."""
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, conditions, groups))
    best = (None, -1.0, None)
    for params in GRIDS[name]:
        scores = np.zeros(len(y))
        for train, test in folds:
            scores[test] = make_model(name, params, seed).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        auc = roc_auc_score(y, scores)
        if auc > best[1]:
            best = (params, auc, scores)
    return best


def select_simplest(candidates):
    """Apply criterion C5: among candidates within the margin of the best inner AUC, keep the simplest."""
    top = candidates["inner_auc"].max()
    eligible = candidates[candidates["inner_auc"] >= top - SIMPLICITY_MARGIN].copy()
    eligible["model_rank"] = eligible["model"].map(SIMPLICITY_ORDER.index)
    eligible["set_rank"] = eligible["feature_set"].map(SET_ORDER.index)
    return eligible.sort_values(["model_rank", "set_rank", "inner_auc"], ascending=[True, True, False]).iloc[0]


def run_outer(df, sets, y, conditions, groups):
    predictions, inner = [], []
    for repetition in range(base.N_REP):
        fold_ids = base.dobras(df, repetition)["E3b"]
        for fold, (set_name, X_frame), model in product(range(base.N_FOLD), sets.items(), GRIDS):
            X = X_frame.values
            train, test = fold_ids != fold, fold_ids == fold
            assert not set(groups[train]) & set(groups[test])
            seed = 1000 * repetition + fold
            params, inner_auc, scores = inner_fit(model, X[train], y[train], conditions[train], groups[train], seed)
            thresholds = {alpha: np.quantile(scores[y[train] == 0], 1 - alpha) for alpha in ALPHAS}
            fitted = make_model(model, params, seed).fit(X[train], y[train])
            row = pd.DataFrame({"repetition": repetition, "fold": fold, "feature_set": set_name, "model": model,
                                "experimento_idx": df["experimento_idx"].values[test], "numero_serie": groups[test],
                                "condicao": conditions[test], "y": y[test], "score": fitted.predict_proba(X[test])[:, 1]})
            for alpha in ALPHAS:
                row[f"threshold_{int(alpha * 100):02d}"] = thresholds[alpha]
            predictions.append(row)
            inner.append({"repetition": repetition, "fold": fold, "feature_set": set_name, "model": model, "params": str(params), "inner_auc": inner_auc})
        print(f"repetition {repetition} done", flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(inner)


def procedure_predictions(predictions, inner):
    """Predictions of the full procedure: model and feature set chosen inside each outer fold."""
    chosen, parts = [], []
    for (repetition, fold), candidates in inner.groupby(["repetition", "fold"]):
        pick = select_simplest(candidates)
        chosen.append(pick)
        parts.append(predictions[(predictions.repetition == repetition) & (predictions.fold == fold)
                                 & (predictions.feature_set == pick.feature_set) & (predictions.model == pick.model)])
    return pd.concat(parts, ignore_index=True), pd.DataFrame(chosen)


def weighted_mean(values, weights):
    total = weights.sum()
    return (values * weights).sum() / total if total > 0 else np.nan


def summarise(frame, weights=None):
    w = np.ones(len(frame)) if weights is None else weights
    y = frame["y"].values
    failure, normal = y == 1, y == 0
    no_f06 = failure & (frame["condicao"].values != "F06")
    critical = np.isin(frame["condicao"].values, CRITICAL)
    out = {"auc": roc_auc_score(y, frame["score"].values, sample_weight=w)}
    operating = {"default": frame["score"].values >= 0.5}
    for alpha in ALPHAS:
        operating[f"{int(alpha * 100):02d}"] = frame["score"].values > frame[f"threshold_{int(alpha * 100):02d}"].values
    for name, alarm in operating.items():
        alarm = alarm.astype(float)
        out[f"recall_{name}"] = weighted_mean(alarm[failure], w[failure])
        out[f"recall_no_f06_{name}"] = weighted_mean(alarm[no_f06], w[no_f06])
        out[f"false_alarm_{name}"] = weighted_mean(alarm[normal], w[normal])
        out[f"critical_miss_{name}"] = weighted_mean(1 - alarm[critical], w[critical])
    return out


def with_intervals(frame, label, rng):
    units = sorted(frame["numero_serie"].unique())
    position = frame["numero_serie"].map({unit: i for i, unit in enumerate(units)}).values
    samples = []
    for draw in rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=N_BOOT):
        weights = draw[position].astype(float)
        if frame["y"].values[weights > 0].min() != frame["y"].values[weights > 0].max():
            samples.append(summarise(frame, weights))
    samples = pd.DataFrame(samples)
    point = summarise(frame)
    return [{"configuration": label, "metric": metric, "estimate": point[metric], "ci_low": samples[metric].quantile(0.025),
             "ci_high": samples[metric].quantile(0.975)} for metric in point]


def permuted_labels(conditions, groups, rng):
    blocks = pd.Series(groups + "|" + conditions)
    unique_blocks = blocks.drop_duplicates()
    labels = rng.permutation(unique_blocks.str.split("|").str[1].values)
    return blocks.map(dict(zip(unique_blocks.values, labels))).values


def grouped_auc(name, params, X, conditions, groups, seed):
    y = make_target(conditions)
    scores = np.zeros(len(y))
    for fold, (train, test) in enumerate(StratifiedGroupKFold(base.N_FOLD, shuffle=True, random_state=seed).split(X, conditions, groups)):
        scores[test] = make_model(name, params, 1000 * seed + fold).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return roc_auc_score(y, scores)


def leaky_auc(name, params, X, y, seed):
    scores = np.zeros(len(y))
    for fold, (train, test) in enumerate(KFold(base.N_FOLD, shuffle=True, random_state=seed).split(X)):
        scores[test] = make_model(name, params, 1000 * seed + fold).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return roc_auc_score(y, scores)


def unit_identification(X, groups):
    predicted = np.empty(len(groups), dtype=object)
    for train, test in KFold(base.N_FOLD, shuffle=True, random_state=0).split(X):
        predicted[test] = RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(X[train], groups[train]).predict(X[test])
    return (predicted == groups).mean()


def importance(name, params, X_frame, y, conditions, groups, df):
    fold_ids = base.dobras(df, 0)["E3b"]
    results = []
    for fold in range(base.N_FOLD):
        train, test = fold_ids != fold, fold_ids == fold
        fitted = make_model(name, params, fold).fit(X_frame.values[train], y[train])
        results.append(permutation_importance(fitted, X_frame.values[test], y[test], scoring="roc_auc", n_repeats=10, random_state=fold).importances_mean)
    return pd.DataFrame({"feature": X_frame.columns, "auc_drop": np.mean(results, axis=0)}).sort_values("auc_drop", ascending=False)


def save_final_model(name, params, X_frame, y, output):
    model = make_model(name, params, 0).fit(X_frame.values, y)
    joblib.dump(model, output / f"{PREFIX}_final_model.joblib")
    if name == "tree":
        text = export_text(model, feature_names=list(X_frame.columns))
    elif name == "logistic":
        coefficients = model[-1].coef_[0]
        text = "\n".join(f"{feature}: {value:.4f}" for feature, value in sorted(zip(X_frame.columns, coefficients), key=lambda item: -abs(item[1])))
    else:
        text = f"{name} {params}: see binary_importance.csv for the feature ranking"
    (output / f"{PREFIX}_final_model.txt").write_text(text + "\n", encoding="utf-8")


def load_data(exclude_f02):
    df = pd.read_csv(base.RAIZ / "dados" / "pbit_features_v3.csv")
    if USE_SPECTRAL:
        df = df.merge(pd.read_csv(base.RAIZ / "dados" / "pbit_spectral_features_v3.csv"), on="experimento_idx", validate="one_to_one")
    return df[df["condicao"] != "F02"].reset_index(drop=True) if exclude_f02 else df


def second_stage(df, X_frame):
    """Among cycles with no fault, separate N01 from N02 in units never seen. Only two units carry N02."""
    normal = df[df["condicao"].isin(NORMAL)].reset_index(drop=True)
    X = X_frame.loc[df["condicao"].isin(NORMAL)].values
    y = (normal["condicao"] == "N02").astype(int).values
    rows = []
    for repetition in range(base.N_REP):
        fold_ids = base.dobras(normal, repetition)["E3b"]
        scores = np.full(len(y), np.nan)
        for fold in range(base.N_FOLD):
            train, test = fold_ids != fold, fold_ids == fold
            if y[train].sum() > 0:
                scores[test] = make_model("logistic", {"C": 0.1}, 0).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        valid = ~np.isnan(scores)
        rows.append({"repetition": repetition, "auc": roc_auc_score(y[valid], scores[valid]), "recall_n02": (scores[y == 1] >= 0.5).mean(),
                     "recall_n01": (scores[y == 0] < 0.5).mean()})
    return pd.DataFrame(rows)


def main():
    output = base.SAIDA
    output.mkdir(exist_ok=True)
    df = load_data(EXCLUDE_F02)
    conditions, groups = df["condicao"].values, df["numero_serie"].values
    y = make_target(conditions)
    sets = build_feature_sets(df)
    rng = np.random.default_rng(2026)

    predictions, inner = run_outer(df, sets, y, conditions, groups)
    predictions.to_csv(output / f"{PREFIX}_predictions.csv", index=False, lineterminator="\n")
    inner.to_csv(output / f"{PREFIX}_inner_selection.csv", index=False, lineterminator="\n")

    procedure, chosen = procedure_predictions(predictions, inner)
    chosen.to_csv(output / f"{PREFIX}_procedure_choices.csv", index=False, lineterminator="\n")

    rows = with_intervals(procedure, "procedure", rng)
    for (feature_set, model), frame in predictions.groupby(["feature_set", "model"]):
        rows += with_intervals(frame, f"{feature_set}_{model}", rng)
    results = pd.DataFrame(rows)

    per_class = predictions.assign(configuration=predictions.feature_set + "_" + predictions.model)
    per_class = pd.concat([per_class, procedure.assign(configuration="procedure")])
    per_class = per_class.assign(alarm=per_class["score"] > per_class["threshold_10"]).groupby(["configuration", "condicao"])["alarm"].mean().unstack().round(3)
    per_class.to_csv(output / f"{PREFIX}_alarm_rate_by_condition_at_10.csv", lineterminator="\n")

    mean_inner = inner.groupby(["feature_set", "model"])["inner_auc"].mean().reset_index()
    final = select_simplest(mean_inner)
    params = ast.literal_eval(inner[(inner.feature_set == final.feature_set) & (inner.model == final.model)]["params"].value_counts().index[0])
    X_final = sets[final.feature_set]
    print("final choice:", final.feature_set, final.model, params, flush=True)

    null = []
    for permutation in range(N_PERMUTATIONS):
        shuffled = permuted_labels(conditions, groups, rng)
        if make_target(shuffled).min() != make_target(shuffled).max():
            null.append(grouped_auc(final.model, params, X_final.values, shuffled, groups, permutation))
    real = np.mean([grouped_auc(final.model, params, X_final.values, conditions, groups, s) for s in range(base.N_REP)])
    control = {"real_grouped_auc": real, "null_mean": np.mean(null), "null_p97_5": np.quantile(null, 0.975),
               "p_value": (np.sum(np.array(null) >= real) + 1) / (len(null) + 1), "n_permutations": len(null),
               "leaky_auc": np.mean([leaky_auc(final.model, params, X_final.values, y, s) for s in range(base.N_REP)])}
    identification = {name: unit_identification(frame.values, groups) for name, frame in sets.items()}
    pd.DataFrame([control | {"feature_set": final.feature_set, "model": final.model, "params": str(params)}]).to_csv(output / f"{PREFIX}_control.csv", index=False, lineterminator="\n")
    pd.Series(identification, name="unit_identification_accuracy").to_csv(output / f"{PREFIX}_unit_identification.csv", lineterminator="\n")

    importance(final.model, params, X_final, y, conditions, groups, df).to_csv(output / f"{PREFIX}_importance.csv", index=False, lineterminator="\n")
    save_final_model(final.model, params, X_final, y, output)
    results.to_csv(output / f"{PREFIX}_results.csv", index=False, lineterminator="\n")
    stage = second_stage(df, X_final)
    stage.to_csv(output / f"{PREFIX}_second_stage_n01_n02.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 30)
    show = results[results.metric.isin(["auc", "recall_default", "false_alarm_default", "recall_10", "false_alarm_10", "recall_no_f06_10", "critical_miss_10"])]
    print(show.assign(text=show.estimate.round(3).astype(str) + " [" + show.ci_low.round(3).astype(str) + "; " + show.ci_high.round(3).astype(str) + "]").pivot(index="configuration", columns="metric", values="text").to_string())
    print(per_class.to_string())
    print(chosen.groupby(["feature_set", "model"]).size().to_string())
    print(mean_inner.round(3).to_string(index=False))
    print(control)
    print(identification)
    print(stage.round(3).to_string(index=False))
    print(pd.read_csv(output / f"{PREFIX}_importance.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
