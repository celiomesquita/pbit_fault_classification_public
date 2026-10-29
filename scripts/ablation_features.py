from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pbit.augmentation import augment_training_set  # noqa: E402
from pbit.data_validation import load_long_dataset  # noqa: E402
from pbit.features import extract_features  # noqa: E402
from pbit.modeling import META_COLS, compute_sample_weights  # noqa: E402

DATA = PROJECT_ROOT / "data"
from population import load_train, out_dir  # noqa: E402
OUT = out_dir(DATA / "ablation")
CLASSES = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5,
               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
F18_NAMES = ["media", "mediana", "desvio_padrao", "minimo", "maximo", "pico_a_pico", "assimetria",
             "curtose", "rms", "valor_inicial", "valor_final", "delta_inicio_fim", "inclinacao",
             "desvio_primeira_metade", "desvio_segunda_metade", "maior_salto", "n_picos", "energia"]


def count_peaks(x: np.ndarray) -> int:
    s = np.sign(np.diff(x))
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if len(s) >= 2 else 0


def f18_row(x: np.ndarray) -> dict:
    n = len(x)
    half = n // 2
    slope = stats.linregress(np.arange(n), x).slope
    return {
        "media": np.mean(x), "mediana": np.median(x), "desvio_padrao": np.std(x),
        "minimo": np.min(x), "maximo": np.max(x), "pico_a_pico": np.max(x) - np.min(x),
        "assimetria": stats.skew(x), "curtose": stats.kurtosis(x), "rms": np.sqrt(np.mean(x ** 2)),
        "valor_inicial": x[0], "valor_final": x[-1], "delta_inicio_fim": x[-1] - x[0],
        "inclinacao": slope, "desvio_primeira_metade": np.std(x[:half]),
        "desvio_segunda_metade": np.std(x[half:]), "maior_salto": np.max(np.abs(np.diff(x))),
        "n_picos": count_peaks(x), "energia": np.sum(x ** 2),
    }


def f18_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, g in long_df.sort_values("Sample_Index").groupby("Experiment_Index", sort=True):
        row = f18_row(g["Measurement"].to_numpy(dtype=float))
        row["Experiment_Index"] = exp
        rows.append(row)
    return pd.DataFrame(rows).set_index("Experiment_Index")[F18_NAMES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    seed = args.seed
    OUT.mkdir(parents=True, exist_ok=True)

    train = load_train(DATA / "train_real.csv")
    long_df = load_long_dataset(DATA / "PBIT_atuador_DESCARAC_r1_long_clean.csv")
    cols246 = [c for c in train.columns if c not in META_COLS]
    cols115 = [c for c in cols246 if not (c.startswith("roll5_") or c.startswith("seg"))]
    ids = set(train["Experiment_Index"])
    real18 = f18_table(long_df[long_df["Experiment_Index"].isin(ids)]).reindex(train["Experiment_Index"].values)
    real18.index = train.index

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
        np.arange(len(train)), train["Fault_Mode"].values, train["Serial_Number"].values))
    pred_rows = []
    for k, (tr_idx, te_idx) in enumerate(folds):
        tr = train.iloc[tr_idx]
        te = train.iloc[te_idx]
        aug_long = augment_training_set(long_df, set(tr["Experiment_Index"]), target_per_class=160,
                                        noise_scale=0.01, random_state=seed + k)
        aug246 = extract_features(aug_long).reset_index(drop=True)
        aug18 = f18_table(aug_long).reindex(aug246["Experiment_Index"].values).reset_index(drop=True)
        y_tr = np.concatenate([tr["Fault_Mode"].values, aug246["Fault_Mode"].values])
        sw = compute_sample_weights(y_tr, "freq_sev")

        mats = {
            "F246": (np.vstack([tr[cols246].values, aug246[cols246].values]).astype(float),
                     te[cols246].values.astype(float)),
            "F115": (np.vstack([tr[cols115].values, aug246[cols115].values]).astype(float),
                     te[cols115].values.astype(float)),
            "F18": (np.vstack([real18.iloc[tr_idx].values, aug18.values]).astype(float),
                    real18.iloc[te_idx].values.astype(float)),
        }
        out = te[["Experiment_Index", "Serial_Number", "Fault_Mode"]].copy()
        out.insert(0, "fold", k)
        out.insert(0, "seed", seed)
        for fs, (Xtr, Xte) in mats.items():
            m = lgb.LGBMClassifier(**{**UNTUNED, "n_jobs": args.threads}).fit(Xtr, y_tr, sample_weight=sw)
            out[f"lgbm_{fs}"] = m.predict(Xte)
            imp = SimpleImputer(strategy="median")
            rf = RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2,
                                        random_state=42, n_jobs=args.threads)
            rf.fit(imp.fit_transform(Xtr), y_tr, sample_weight=sw)
            out[f"rf_{fs}"] = rf.predict(imp.transform(Xte))

        # trivial baselines (in-fold)
        n_tr = len(tr)
        rng = np.random.default_rng(seed * 100 + k)
        freq = tr["Fault_Mode"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
        n01_std = real18.iloc[tr_idx].loc[tr["Fault_Mode"].values == "N01", "desvio_padrao"]
        thr = 1.5 * n01_std.median()
        fault_counts = tr["Fault_Mode"][tr["Fault_Mode"] != "N01"].value_counts()
        common_fault = fault_counts.idxmax()
        std_te = real18.iloc[te_idx]["desvio_padrao"].values
        out["B1_always_N01"] = "N01"
        out["B2_prior_random"] = rng.choice(CLASSES, size=len(te), p=freq)
        out["B3_threshold_std"] = np.where(std_te <= thr, "N01", common_fault)
        out["B4_threshold_safety"] = np.where(std_te <= thr, "N01", "F03")
        out["B5_always_F03"] = "F03"
        pred_rows.append(out)
        print(f"seed {seed} fold {k} done", flush=True)

    pd.concat(pred_rows).to_csv(OUT / f"preds_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main()
