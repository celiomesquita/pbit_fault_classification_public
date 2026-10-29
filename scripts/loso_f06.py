from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_features import f18_table  # noqa: E402
from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = DATA / "loso_f06"
F06_SN = ["SN 045", "SN 052", "SN 054", "SN 056"]
SEEDS = [42, 0, 7, 13, 21]
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
               subsample=0.8, colsample_bytree=0.8, verbose=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["D_primary", "D_sens"], required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    audit = pd.read_csv(DATA / "audit_dataset_r1.csv")
    keep = set(audit.loc[audit["in_D_primary" if args.dataset == "D_primary" else "in_D_sens"], "Experiment_Index"])
    feats = pd.read_csv(DATA / "PBIT_features_r1.csv")
    feats = feats[feats["Experiment_Index"].isin(keep)].reset_index(drop=True)
    long_all = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in feats.columns if c not in META_COLS]
    real18 = f18_table(long_all[long_all["Experiment_Index"].isin(keep)]).reindex(feats["Experiment_Index"].values)
    real18.index = feats.index

    tgt = args.target
    test_mask = (feats["Serial_Number"] == tgt).values
    test = feats[test_mask]
    base_train_mask = ~test_mask
    sources = [s for s in F06_SN if s != tgt]
    variants = {"V0": sources, **{f"V1-{s}": [s] for s in sources}}
    rows = []

    def add(variant, model, seed, df, preds):
        for e, y, p in zip(df["Experiment_Index"], df["Fault_Mode"], preds):
            rows.append({"dataset": args.dataset, "target": tgt, "variant": variant, "model": model,
                         "seed": seed, "Experiment_Index": e, "Fault_Mode": y, "pred": p})

    for vname, f06_sources in variants.items():
        f06_ok = feats["Fault_Mode"].ne("F06") | feats["Serial_Number"].isin(f06_sources)
        tr_mask = base_train_mask & f06_ok.values
        tr = feats[tr_mask]
        tr_ids = set(tr["Experiment_Index"])
        for seed in SEEDS:
            aug_full = augment_training_set(long_all, tr_ids, target_per_class=160, noise_scale=0.01, random_state=seed)
            aug_nof06 = augment_training_set(long_all, tr_ids, target_per_class={"F04": 160, "F05": 160},
                                             noise_scale=0.01, random_state=seed)
            for model, aug_long, use18 in (("M1_lgbm_F246", aug_full, False), ("M2_lgbm_F246_noF06aug", aug_nof06, False),
                                           ("M3_rf_F246", aug_full, False), ("M4_lgbm_F18", aug_full, True)):
                a246 = extract_features(aug_long).reset_index(drop=True)
                y = np.concatenate([tr["Fault_Mode"].values, a246["Fault_Mode"].values])
                sw = compute_sample_weights(y, "freq_sev")
                if use18:
                    a18 = f18_table(aug_long).reindex(a246["Experiment_Index"].values).values
                    X = np.vstack([real18.loc[tr.index].values, a18]).astype(float)
                    Xt = real18.loc[test.index].values.astype(float)
                else:
                    X = np.vstack([tr[cols].values, a246[cols].values]).astype(float)
                    Xt = test[cols].values.astype(float)
                if model == "M3_rf_F246":
                    imp = SimpleImputer(strategy="median")
                    m = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                               random_state=seed, n_jobs=args.threads)
                    m.fit(imp.fit_transform(X), y, sample_weight=sw)
                    pred = m.predict(imp.transform(Xt))
                else:
                    m = lgb.LGBMClassifier(**UNTUNED, random_state=seed, n_jobs=args.threads)
                    m.fit(X, y, sample_weight=sw)
                    pred = m.predict(Xt)
                add(vname, model, seed, test, pred)
            print(f"{args.dataset} {tgt} {vname} seed {seed} done", flush=True)

        # binary F03 x F06 (reconciliation with Section 5.5), V0 only
        if vname == "V0":
            sub = tr[tr["Fault_Mode"].isin(["F03", "F06"])]
            ids_b = set(sub["Experiment_Index"])
            for seed in SEEDS:
                for tag, aug_on in (("BIN_aug", True), ("BIN_noaug", False)):
                    Xb, yb = sub[cols].values.astype(float), sub["Fault_Mode"].values
                    if aug_on:
                        al = augment_training_set(long_all, ids_b, target_per_class={"F06": 160},
                                                  noise_scale=0.01, random_state=seed)
                        ab = extract_features(al).reset_index(drop=True)
                        Xb = np.vstack([Xb, ab[cols].values]).astype(float)
                        yb = np.concatenate([yb, ab["Fault_Mode"].values])
                    swb = np.where(yb == "F06", (yb == "F03").sum() / max((yb == "F06").sum(), 1), 1.0)
                    mb = lgb.LGBMClassifier(**UNTUNED, random_state=seed, n_jobs=args.threads).fit(Xb, yb, sample_weight=swb)
                    t6 = test[test["Fault_Mode"] == "F06"]
                    if len(t6):
                        add("V0", tag, seed, t6, mb.predict(t6[cols].values.astype(float)))

    pd.DataFrame(rows).to_csv(OUT / f"preds_{args.dataset}_{tgt.replace(' ', '')}.csv", index=False)


if __name__ == "__main__":
    main()
