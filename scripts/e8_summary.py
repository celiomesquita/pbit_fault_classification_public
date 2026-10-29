"""Descriptive summary of E8 (SMOTEWB-LGBM of Li et al., 2024) under grouped and random partitions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nested_cv_summary import md  # noqa: E402
from pbit.modeling import FAULT_ORDER, SEVERITY_MAP  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "e8_smotewb"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
B = 10_000
CIDX = {c: i for i, c in enumerate(FAULT_ORDER)}
SEV = np.array([SEVERITY_MAP[c] for c in FAULT_ORDER])
WEIGHTS = ["none", "freq", "sev", "freq_sev"]
KEYS = ["CMR", "FA0", "DR", "R3", "macro_F1", "micro_F1", "weighted_P", "rec_F04", "rec_F05", "rec_F06"]
REFERENCE = ["lgbm_freq_sev_gaussian", "lgbm_freq_sev_smote", "lgbm_freq_sev_none"]
CONTRASTS = [
    ("G1", "li_tuned_grouped", "lgbm_freq_sev_gaussian", ["CMR", "FA0", "macro_F1"]),
    ("G2", "lgbm_freq_sev_smotewb", "lgbm_freq_sev_smote_bal", ["CMR", "FA0", "macro_F1"]),
    ("G3", "lgbm_freq_sev_smotewb", "lgbm_freq_sev_gaussian", ["CMR", "FA0", "macro_F1"]),
    ("L", "li_tuned_random", "li_tuned_grouped", ["macro_F1", "micro_F1", "CMR", "FA0"]),
    ("L", "li_fixed_random", "li_fixed", ["macro_F1", "micro_F1", "CMR", "FA0"]),
]


def batch_metrics(cm):
    """Metrics of nested_cv_summary/imbalance_summary plus Li's micro-F1 and weighted precision, over (..., 8, 8)."""
    cm = np.asarray(cm, dtype=float)
    rows, cols, tp = cm.sum(axis=-1), cm.sum(axis=-2), np.diagonal(cm, axis1=-2, axis2=-1)
    n = cm.sum(axis=(-2, -1))
    s0, s3 = SEV == 0, SEV == 3
    sev3 = cm[..., s3, :].sum(axis=(-2, -1))
    normal = cm[..., s0, :].sum(axis=(-2, -1))
    down = (cm * (SEV[None, :] < SEV[:, None])).sum(axis=(-2, -1))
    fp, fn = cols - tp, rows - tp
    present = (rows + cols) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    precision = np.where(cols > 0, tp / np.maximum(cols, 1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "CMR": cm[..., s3, :][..., s0].sum(axis=(-2, -1)) / sev3,
            "FA0": cm[..., s0, :][..., SEV >= 1].sum(axis=(-2, -1)) / normal,
            "DR": down / n,
            "R3": cm[..., s3, :][..., s3].sum(axis=(-2, -1)) / sev3,
            "macro_F1": (f1 * present).sum(axis=-1) / present.sum(axis=-1),
            "micro_F1": tp.sum(axis=-1) / n,
            "weighted_P": (precision * rows).sum(axis=-1) / n,
        }
        for c in ("F04", "F05", "F06"):
            i = CIDX[c]
            out[f"rec_{c}"] = cm[..., i, i] / cm[..., i, :].sum(axis=-1)
    return out


def load():
    grouped = pd.concat([pd.read_csv(OUT / "grouped" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    rand = pd.concat([pd.read_csv(OUT / "random" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    ablation = pd.concat([pd.read_csv(DATA / "imbalance" / f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    key = ["seed", "Experiment_Index"]
    merged = grouped.merge(ablation[key + ["fold", "Fault_Mode"] + REFERENCE], on=key, suffixes=("", "_abl"))
    assert len(merged) == len(grouped) == len(ablation) == 10 * 1196
    assert (merged["fold"] == merged["fold_abl"]).all() and (merged["Fault_Mode"] == merged["Fault_Mode_abl"]).all()
    rand = rand.rename(columns={"fold": "fold_random"})
    merged = merged.merge(rand[key + ["fold_random", "Fault_Mode", "li_fixed_random", "li_tuned_random"]],
                          on=key, suffixes=("", "_rand"))
    assert len(merged) == len(grouped) and (merged["Fault_Mode"] == merged["Fault_Mode_rand"]).all()
    return merged


def confusions(d, arms):
    sns = sorted(d["Serial_Number"].unique())
    sidx = {s: i for i, s in enumerate(sns)}
    cms = {}
    for arm in arms:
        cm = np.zeros((len(SEEDS), len(sns), 8, 8))
        for si, seed in enumerate(SEEDS):
            g = d[d["seed"] == seed]
            np.add.at(cm[si], (g["Serial_Number"].map(sidx).values, g["Fault_Mode"].map(CIDX).values,
                               g[arm].map(CIDX).values), 1)
        cms[arm] = cm
    return cms, len(sns)


def summarise(cms, n_sn):
    weights = np.random.default_rng(42).multinomial(n_sn, np.ones(n_sn) / n_sn, size=B).astype(float)
    point, boot = {}, {}
    for arm, cm in cms.items():
        per_seed = batch_metrics(cm.sum(axis=1))
        point[arm] = {k: float(np.nanmean(per_seed[k])) for k in KEYS}
        draws = [batch_metrics(np.tensordot(weights, cm[si], axes=(1, 0))) for si in range(len(SEEDS))]
        boot[arm] = {k: np.nanmean(np.stack([d[k] for d in draws]), axis=0) for k in KEYS}
    return point, boot


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return f"[{lo:.3f}; {hi:.3f}]"


def arm_table(point, boot, arms):
    rows = []
    for arm in arms:
        p, b = point[arm], boot[arm]
        rows.append({"braço": arm, "CMR": f"{p['CMR']:.3f}", "CMR_IC95": ci(b["CMR"]), "FA0": f"{p['FA0']:.3f}",
                     "DR": f"{p['DR']:.3f}", "R3": f"{p['R3']:.3f}", "macro_F1": f"{p['macro_F1']:.3f}",
                     "macro_F1_IC95": ci(b["macro_F1"]), "micro_F1": f"{p['micro_F1']:.3f}",
                     "weighted_P": f"{p['weighted_P']:.3f}", "rec_F04": f"{p['rec_F04']:.3f}",
                     "rec_F05": f"{p['rec_F05']:.3f}", "rec_F06": f"{p['rec_F06']:.3f}"})
    return md(pd.DataFrame(rows).set_index("braço"))


def resampling_table():
    rows = []
    for partition in ("grouped", "random"):
        for seed in SEEDS:
            for log in json.loads((OUT / partition / f"resampling_seed{seed}.json").read_text(encoding="utf-8")):
                for arm in ("smotewb", "smote_bal", "li_fixed", "li_tuned"):
                    for label, rec in log.get(arm, {}).items():
                        rows.append(dict(partition=partition, arm=arm, cls=label, **rec))
    d = pd.DataFrame(rows)
    d = d[d["cls"].isin(["F04", "F05", "F06", "N02"])]
    g = d.groupby(["partition", "arm", "cls"]).agg(before=("before", "mean"), after=("after", "mean"),
                                                   generated=("generated", "sum"), copies=("exact_copies", "sum"),
                                                   untouched=("generated", lambda x: int((x == 0).sum())))
    g["copy_share"] = (g["copies"] / g["generated"].where(g["generated"] > 0)).round(3)
    g[["before", "after"]] = g[["before", "after"]].round(1)
    return md(g.reset_index().set_index("partition"))


def tuning_table():
    rows = []
    for partition in ("grouped", "random"):
        for seed in SEEDS:
            for log in json.loads((OUT / partition / f"resampling_seed{seed}.json").read_text(encoding="utf-8")):
                rows.append(dict(partition=partition, inner_micro_F1=log["li_tuned_inner_micro_f1"],
                                 seconds=log["seconds"], **log["li_tuned_best"]))
    d = pd.DataFrame(rows)
    q = d.groupby("partition").quantile([0.0, 0.5, 1.0]).round(3)
    q.index = [f"{p} q{int(100 * v)}" for p, v in q.index]
    return md(q), d


def main():
    d = load()
    arms = (REFERENCE + [f"{m}_{w}_{a}" for a in ("smotewb", "smote_bal") for m in ("lgbm", "rf") for w in WEIGHTS]
            + ["li_fixed", "li_tuned_grouped", "li_fixed_random", "li_tuned_random"])
    cms, n_sn = confusions(d, arms)
    point, boot = summarise(cms, n_sn)
    rows = []
    for name, a, b, metrics in CONTRASTS:
        for m in metrics:
            x = boot[a][m] - boot[b][m]
            rows.append({"contraste": name, "A − B": f"{a} − {b}", "métrica": m,
                         "dif": f"{point[a][m] - point[b][m]:+.4f}", "IC95_descritivo": ci(x)})
    tuning, tuned = tuning_table()
    manifests = sorted(OUT.glob("*/manifest_seed*.json"))
    freeze = json.loads((OUT / "protocol_freeze.json").read_text(encoding="utf-8-sig"))
    out = ["# E8 — SMOTEWB-LGBM (Li et al., 2024) sob partição por SN e aleatória", "",
           f"Protocolo `docs/protocolos/E8.md`, SHA-256 `{freeze['protocol_sha256']}`, congelado em {freeze['frozen_utc']}.",
           f"{len(manifests)} manifestos; 10 sementes × 5 dobras × 1.196 ciclos por partição; {n_sn} SN.",
           f"Médias sobre sementes das confusões agregadas; IC por bootstrap de SN ({B} sorteios, rng 42), "
           "descritivos e não calibrados (E3/E3b não validaram procedimento). Sem testes nem veredictos.",
           "Braços `lgbm_*_gaussian/smote/none` vêm de `data/imbalance/`, com as mesmas dobras (conferidas).", "",
           "## Partição por SN — referência da ablação, SMOTEWB, SMOTE balanceado e Li et al.", "",
           arm_table(point, boot, [a for a in arms if not a.endswith("_random")]), "",
           "## Partição aleatória (StratifiedKFold) — pipeline de Li et al.", "",
           arm_table(point, boot, ["li_fixed_random", "li_tuned_random"]), "",
           "## Contrastes pré-especificados (descritivos)", "", md(pd.DataFrame(rows).set_index("contraste")), "",
           "## Reamostragem: contagens médias por dobra e cópias exactas geradas", "", resampling_table(), "",
           "## Reajuste TPE (100 tentativas por dobra externa): melhor configuração e custo", "", tuning, "",
           f"Tempo total por dobra externa: mediana {tuned['seconds'].median():.0f} s.", ""]
    path = DATA / "e8_smotewb_summary.md"
    path.write_text("\n".join(out), encoding="utf-8")
    (OUT / "summary_point.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
    print("\n".join(out))
    print("summary sha256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
