import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sv = pytest.importorskip("smote_variants")
import e8_smotewb as e8  # noqa: E402

TRAIN = pd.read_csv(ROOT / "data/train_real.csv")
COLS = [c for c in TRAIN.columns if c not in e8.META_COLS]


@pytest.mark.parametrize("seed", e8.SEEDS)
def test_grouped_folds_match_ablation(seed):
    saved = pd.read_csv(ROOT / f"data/imbalance/preds_seed{seed}.csv", usecols=["fold", "Experiment_Index"])
    expected = dict(zip(saved["Experiment_Index"], saved["fold"]))
    for k, (_, te) in enumerate(e8.outer_folds(TRAIN, seed, "grouped")):
        assert {expected[i] for i in TRAIN["Experiment_Index"].iloc[te]} == {k}
    assert len(expected) == len(TRAIN)


def test_random_folds_ignore_groups():
    folds = e8.outer_folds(TRAIN, 42, "random")
    shared = [len(set(TRAIN["Serial_Number"].iloc[tr]) & set(TRAIN["Serial_Number"].iloc[te])) for tr, te in folds]
    assert sorted(np.concatenate([te for _, te in folds])) == list(range(len(TRAIN)))
    assert min(shared) > 0


def test_environment_reproduces_saved_smote_predictions():
    saved = pd.read_csv(ROOT / "data/imbalance/preds_seed42.csv")
    saved = saved[saved["fold"] == 0]
    tr_idx, te_idx = e8.outer_folds(TRAIN, 42, "grouped")[0]
    tr, te = TRAIN.iloc[tr_idx], TRAIN.iloc[te_idx]
    y = tr["Fault_Mode"].values
    X = SimpleImputer(strategy="median").fit_transform(tr[COLS].values.astype(float))
    counts = pd.Series(y).value_counts()
    strategy = {c: 160 for c in ["F04", "F05", "F06"] if counts.get(c, 0) < 160}
    k_nn = max(1, min(5, int(min(counts[c] for c in strategy)) - 1))
    X_os, y_os = SMOTE(sampling_strategy=strategy, k_neighbors=k_nn, random_state=42).fit_resample(X, y)
    X_te = SimpleImputer(strategy="median").fit(X_os).transform(te[COLS].values.astype(float))
    model = lgb.LGBMClassifier(**{**e8.UNTUNED, "n_jobs": 2})
    model.fit(X_os, y_os, sample_weight=e8.compute_sample_weights(y_os, "freq_sev"))
    assert list(model.predict(X_te)) == list(saved["lgbm_freq_sev_smote"])


def test_resampling_log_counts_exact_copies():
    X = np.array([[0.0, 0], [1, 1], [5, 5], [6, 6]])
    y = np.array(["a", "a", "b", "b"])
    X_out = np.vstack([X, [[0, 0], [0.5, 0.5], [5, 5]]])
    y_out = np.concatenate([y, ["a", "a", "b"]])
    log = e8.resampling_log(X, y, X_out, y_out)
    assert log["a"] == dict(before=2, after=4, generated=2, exact_copies=1)
    assert log["b"] == dict(before=2, after=3, generated=1, exact_copies=1)


def test_li_parameters_map_to_lightgbm_names():
    params = e8.li_lgbm(e8.LI_TABLE4, 7, 1).get_params()
    assert params["n_estimators"] == 100 and params["random_state"] == 7
    assert (params["reg_alpha"], params["reg_lambda"], params["num_leaves"]) == (0.1049, 0.2042, 17)
    assert (params["colsample_bytree"], params["subsample"], params["learning_rate"]) == (0.5204, 0.9395, 0.0979)
    assert params["subsample_freq"] == 0


def test_batch_metrics_match_ablation_metrics():
    import e8_summary
    import imbalance_summary
    from sklearn.metrics import f1_score, precision_score

    rng = np.random.default_rng(3)
    cms = rng.poisson(2.0, size=(40, 8, 8)) * (rng.random((40, 8, 8)) < 0.6)
    cms[0, 7, :] = 0
    cms[1, :, 6] = 0
    batch = e8_summary.batch_metrics(cms)
    labels = np.arange(8)
    for i, cm in enumerate(cms):
        expected = imbalance_summary.metrics(cm)
        for key in ["CMR", "FA0", "DR", "R3", "macro_F1", "rec_F04", "rec_F05", "rec_F06"]:
            np.testing.assert_allclose(batch[key][i], expected[key], equal_nan=True)
        y_true = np.repeat(np.repeat(labels, 8), cm.ravel())
        y_pred = np.repeat(np.tile(labels, 8), cm.ravel())
        np.testing.assert_allclose(batch["micro_F1"][i], f1_score(y_true, y_pred, average="micro"))
        np.testing.assert_allclose(batch["weighted_P"][i],
                                   precision_score(y_true, y_pred, average="weighted", zero_division=0))


def test_successive_strategy_balances_to_largest_class_and_keeps_tiny_class():
    tr_idx, _ = e8.outer_folds(TRAIN, 42, "grouped")[0]
    tr = TRAIN.iloc[tr_idx]
    y = tr["Fault_Mode"].values
    X = SimpleImputer(strategy="median").fit_transform(tr[COLS].values.astype(float))
    _, y_os = e8.oversample("SMOTEWB", X, y, {}, 42)
    counts = pd.Series(y_os).value_counts()
    assert counts["F06"] == 2
    assert set(counts.drop("F06")) == {381}
