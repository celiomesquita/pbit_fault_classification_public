from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
SEEDS = [42, 0, 7, 13, 21]
F06_SN = ["SN 045", "SN 052", "SN 054", "SN 056"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    tgt = args.target
    audit = pd.read_csv(DATA / "audit_dataset_r1.csv")
    keep = set(audit.loc[audit["in_D_primary"], "Experiment_Index"])
    feats = pd.read_csv(DATA / "PBIT_features_r1.csv")
    feats = feats[feats["Experiment_Index"].isin(keep)].reset_index(drop=True)
    long_all = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols = [c for c in feats.columns if c not in META_COLS]
    test = feats[feats["Serial_Number"] == tgt]
    tr = feats[feats["Serial_Number"] != tgt]
    ids = set(tr["Experiment_Index"])

    p1 = json.loads((DATA / "best_lgbm_params.json").read_text())
    f05_1 = p1.pop("f05_boost")
    p2 = json.loads((DATA / "best_lgbm_cmr_params.json").read_text())
    f05_2 = p2.pop("f05_boost")
    s3_2 = p2.pop("sev3_boost")

    rows = []
    for seed in SEEDS:
        aug = extract_features(augment_training_set(long_all, ids, target_per_class=160, noise_scale=0.01,
                                                    random_state=seed)).reset_index(drop=True)
        y = np.concatenate([tr["Fault_Mode"].values, aug["Fault_Mode"].values])
        X = np.vstack([tr[cols].values, aug[cols].values]).astype(float)
        Xt = test[cols].values.astype(float)
        for name, params, f05, s3 in (("M5_lgbm_F1tuned", p1, f05_1, None), ("M6_lgbm_CMRtuned", p2, f05_2, s3_2)):
            sw = compute_sample_weights(y, "freq_sev")
            sw[y == "F05"] *= f05
            if s3:
                sw[np.isin(y, ["F03", "F06"])] *= s3
            m = lgb.LGBMClassifier(**{**params, "random_state": seed, "n_jobs": args.threads}).fit(X, y, sample_weight=sw)
            pred = m.predict(Xt)
            for e, yt, p in zip(test["Experiment_Index"], test["Fault_Mode"], pred):
                rows.append({"dataset": "D_primary", "target": tgt, "variant": "V0", "model": name,
                             "seed": seed, "Experiment_Index": e, "Fault_Mode": yt, "pred": p})
        print(f"{tgt} seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "loso_f06" / f"preds_D_primary_tuned_{tgt.replace(' ', '')}.csv", index=False)


if __name__ == "__main__":
    main()
