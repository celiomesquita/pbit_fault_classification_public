from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nested_cv_tuning as nt  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.modeling import (  # noqa: E402
    META_COLS,
    compute_safety_metrics,
    evaluate_predictions,
)


def main() -> None:
    train = pd.read_csv(nt.DATA / "train_real.csv")
    long_df = load_long_dataset(nt.DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    feat_cols = [c for c in train.columns if c not in META_COLS]
    rows = []
    for k in range(5):
        otr, _ = nt.folds_for(train, 5)[k]
        outer_train = train.iloc[otr].reset_index(drop=True)
        inner = nt.folds_for(outer_train, 5)
        cache = []
        for j, (itr, ite) in enumerate(inner):
            full = nt.augmented(long_df, outer_train, itr, 42 + j)
            cache.append((full[feat_cols].values.astype(float), full["Fault_Mode"].values,
                          outer_train.iloc[ite][feat_cols].values.astype(float),
                          outer_train.iloc[ite]["Fault_Mode"].values))

        def evaluate(params, f05, sev3):
            bals, cmrs, present = [], [], []
            for Xt, yt, Xv, yv in cache:
                m = lgb.LGBMClassifier(**params, n_jobs=8).fit(
                    Xt, yt, sample_weight=nt.weights(yt, f05, sev3))
                yp = m.predict(Xv)
                bals.append(evaluate_predictions(yv, yp)["balanced_accuracy"])
                cmrs.append(compute_safety_metrics(yv, yp)["cmr"])
                present.append(len(set(yv)))
            return float(np.mean(bals)), float(np.nanmean(cmrs)), bals, present

        b, c, bl, pr = evaluate(nt.UNTUNED, None, None)
        rows.append({"outer": k, "config": "untuned", "bal_acc": round(b, 3), "cmr": round(c, 3),
                     "bal_acc_por_fold_interno": [round(x, 3) for x in bl], "classes_presentes_por_fold": pr})
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=nt.SEED))

        def obj(trial):
            params, f05, sev3 = nt.suggest(trial, True)
            b, c, _, _ = evaluate(params, f05, sev3)
            trial.set_user_attr("bal", b)
            return 1.0 if b < 0.50 else c

        study.optimize(obj, n_trials=10)
        bals = [t.user_attrs["bal"] for t in study.trials]
        rows.append({"outer": k, "config": "10 tentativas aleatorias (TPE inicial)",
                     "bal_acc": f"min {min(bals):.3f} / mediana {np.median(bals):.3f} / max {max(bals):.3f}",
                     "cmr": "", "bal_acc_por_fold_interno": "", "classes_presentes_por_fold": ""})
        print(rows[-2], rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(nt.DATA / "nested_cv_constraint_diagnosis.csv", index=False)


if __name__ == "__main__":
    main()
