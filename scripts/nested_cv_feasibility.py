from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEV3 = ["F03", "F06"]


def counts(df: pd.DataFrame) -> dict:
    out = {}
    for c in SEV3:
        sub = df[df["Fault_Mode"] == c]
        out[f"{c}_cycles"] = len(sub)
        out[f"{c}_SN"] = sub["Serial_Number"].nunique()
    out["Sev3_cycles"] = out["F03_cycles"] + out["F06_cycles"]
    return out


def split(df: pd.DataFrame, n_splits: int, seed: int):
    X = np.arange(len(df))
    y = df["Fault_Mode"].values
    g = df["Serial_Number"].values
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, g))


def audit(train: pd.DataFrame, n_outer: int, n_inner: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, (otr, ote) in enumerate(split(train, n_outer, seed)):
        outer_train = train.iloc[otr].reset_index(drop=True)
        outer_test = train.iloc[ote]
        co_tr, co_te = counts(outer_train), counts(outer_test)
        row = {
            "outer": i, "level": "outer", "inner": "",
            **{f"train_{k}": v for k, v in co_tr.items()},
            **{f"val_{k}": v for k, v in co_te.items()},
        }
        row["ok_train_F03"] = co_tr["F03_SN"] >= 2
        row["ok_train_F06"] = co_tr["F06_SN"] >= 2
        row["ok_val_sev3"] = co_te["Sev3_cycles"] >= 1
        rows.append(row)
        for j, (itr, ite) in enumerate(split(outer_train, n_inner, seed)):
            ci_tr, ci_va = counts(outer_train.iloc[itr]), counts(outer_train.iloc[ite])
            r = {
                "outer": i, "level": "inner", "inner": j,
                **{f"train_{k}": v for k, v in ci_tr.items()},
                **{f"val_{k}": v for k, v in ci_va.items()},
            }
            r["ok_train_F03"] = ci_tr["F03_SN"] >= 2
            r["ok_train_F06"] = ci_tr["F06_SN"] >= 2
            r["ok_val_sev3"] = ci_va["Sev3_cycles"] >= 1
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "nested_cv_feasibility.csv")
    args = parser.parse_args()

    train = pd.read_csv(PROJECT_ROOT / "data" / "train_real.csv")
    print(f"Partição de treino: {len(train)} ciclos, {train['Serial_Number'].nunique()} SN")
    print("SN por classe:", train.groupby("Fault_Mode")["Serial_Number"].nunique().to_dict())

    all_rows = []
    for n_outer, n_inner in ((5, 5), (3, 3)):
        a = audit(train, n_outer, n_inner, args.seed)
        a.insert(0, "design", f"{n_outer}x{n_inner}")
        all_rows.append(a)
        ok_cols = ["ok_train_F03", "ok_train_F06", "ok_val_sev3"]
        for level in ("outer", "inner"):
            sub = a[a["level"] == level]
            print(f"\nDesenho {n_outer}x{n_inner}, nível {level} ({len(sub)} folds):")
            for c in ok_cols:
                print(f"  {c}: {int(sub[c].sum())}/{len(sub)} folds atendem")
            print(f"  todos os critérios: {int(sub[ok_cols].all(axis=1).sum())}/{len(sub)} folds")
    pd.concat(all_rows).to_csv(args.out, index=False)
    print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
