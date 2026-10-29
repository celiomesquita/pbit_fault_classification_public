from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MISSING_LIMIT = 0.5
CONSTANT_STD = 1e-6
DROP_THRESHOLD = 0.05
DROP_MIN_RUN = 10
DROP_MIN_MEDIAN = 1.0
NORMAL_CLASSES = {"N01", "N02"}
FAULT_ORDER = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]
D_SENS_EXTRA_EXCLUSION = {920}


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def audit_cycles(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, group in long_df.sort_values("Sample_Index").groupby("Experiment_Index"):
        x = group["Measurement"].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        xv = x[valid]
        n = len(x)
        std = float(np.std(xv)) if len(xv) > 1 else np.nan
        median_abs = float(np.median(np.abs(xv))) if len(xv) else np.nan
        low = np.zeros(n, dtype=bool)
        low[valid] = np.abs(xv) < DROP_THRESHOLD
        rows.append({
            "Experiment_Index": exp,
            "Serial_Number": group["Serial_Number"].iloc[0],
            "Fault_Mode": group["Fault_Mode"].iloc[0],
            "Severity_Index": int(group["Severity_Index"].iloc[0]),
            "n_samples": n,
            "n_valid": int(valid.sum()),
            "n_missing": int((~valid).sum()),
            "missing_frac": float((~valid).mean()),
            "signal_std": std,
            "signal_mean": float(np.mean(xv)) if len(xv) else np.nan,
            "low_run": longest_run(low),
            "median_abs": median_abs,
        })
    audit = pd.DataFrame(rows)
    audit["R1_missing_gt50"] = audit["missing_frac"] > MISSING_LIMIT
    audit["R2_dead_channel"] = (
        (audit["signal_std"] < CONSTANT_STD) & (audit["n_valid"] > 1)
    )
    audit["R3_drop_to_zero"] = (
        (audit["low_run"] >= DROP_MIN_RUN) & (audit["median_abs"] > DROP_MIN_MEDIAN)
    )
    dead_normal = audit["R2_dead_channel"] & audit["Fault_Mode"].isin(NORMAL_CLASSES)
    audit["in_D_primary"] = ~(audit["R1_missing_gt50"] | dead_normal)
    audit["in_D_sens"] = audit["in_D_primary"] & ~audit["Experiment_Index"].isin(
        D_SENS_EXTRA_EXCLUSION
    )
    return audit.drop(columns=["median_abs"])


def class_table(audit: pd.DataFrame) -> pd.DataFrame:
    def counts(mask):
        sub = audit[mask]
        return (
            sub.groupby("Fault_Mode")
            .agg(cycles=("Experiment_Index", "size"), units=("Serial_Number", "nunique"))
            .reindex(FAULT_ORDER)
        )

    raw = counts(audit.index == audit.index)
    prim = counts(audit["in_D_primary"])
    sens = counts(audit["in_D_sens"])
    out = pd.concat(
        {"raw": raw, "D_primary": prim, "D_sens": sens}, axis=1
    )
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


def to_markdown(df: pd.DataFrame) -> str:
    df = df.reset_index()
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PBIT cycles and define D_primary / D_sens.")
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "PBIT_atuador_DESCARAC_r1_long.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    long_df = pd.read_csv(args.input, encoding="utf-8-sig")
    audit = audit_cycles(long_df)
    audit.to_csv(args.out_dir / "audit_dataset_r1.csv", index=False)

    flagged = audit[
        audit["R1_missing_gt50"] | audit["R2_dead_channel"] | audit["R3_drop_to_zero"]
        | (audit["n_missing"] > 0)
    ]
    sn045 = audit[audit["Serial_Number"] == "SN 045"].groupby("Fault_Mode").size()
    units_per_class = audit.groupby("Fault_Mode")["Serial_Number"].nunique().reindex(FAULT_ORDER)

    lines = [
        "# Auditoria do conjunto de dados PBIT r1",
        "",
        f"Entrada: `{args.input.name}`. Ciclos: {len(audit)}. Medições ausentes: {int(audit['n_missing'].sum())}.",
        "",
        f"Regras: R1 ausência > {MISSING_LIMIT:.0%}; R2 desvio-padrão < {CONSTANT_STD:g} (canal morto); "
        f"R3 pelo menos {DROP_MIN_RUN} amostras consecutivas com |valor| < {DROP_THRESHOLD} em ciclo de mediana |valor| > {DROP_MIN_MEDIAN:g} (exploratória, sem exclusão).",
        "D_primary exclui R1 e canais mortos de classes normais; D_sens exclui ainda o experimento 920.",
        "",
        "## Ciclos e unidades por classe",
        "",
        to_markdown(class_table(audit)),
        "",
        "## Ciclos sinalizados",
        "",
        to_markdown(
            flagged[[
                "Experiment_Index", "Serial_Number", "Fault_Mode", "n_valid", "n_missing",
                "signal_std", "signal_mean", "low_run",
                "R1_missing_gt50", "R2_dead_channel", "R3_drop_to_zero",
                "in_D_primary", "in_D_sens",
            ]].set_index("Experiment_Index")
        ),
        "",
        "## SN 045",
        "",
        "Ciclos por classe (bruto): " + ", ".join(f"{k} = {v}" for k, v in sn045.items()),
        "",
        "## Unidades por classe (bruto)",
        "",
        ", ".join(f"{k} = {v}" for k, v in units_per_class.items()),
        "",
    ]
    (args.out_dir / "audit_dataset_r1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
