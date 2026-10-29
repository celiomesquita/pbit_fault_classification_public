"""E9: fixed-model leakage contrast, following docs/protocolos/E9.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1_external_analysis import FAULT_ORDER
from e7_compare import all_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
OUT = DATA/"e9_fixed_model"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
DRAWS, BOOT_SEED = 10000, 20261027
METRICS = ["Accuracy", "macro_F1", "CMR", "FA0", "DR", "SE", "R3"]
MODELS = {"lgbm": "LightGBM", "rf": "Random Forest"}


def tensor(frame, column, units):
    index = {u: i for i, u in enumerate(units)}
    cls = {c: i for i, c in enumerate(FAULT_ORDER)}
    result = np.zeros((len(units), 8, 8))
    np.add.at(result, ([index[u] for u in frame.Serial_Number], [cls[t] for t in frame.Fault_Mode],
                       [cls[p] for p in frame[column]]), 1)
    return result


def balanced(pooled, mask):
    rows = pooled.sum(axis=2)
    recall = np.where(rows > 0, np.einsum("dtt->dt", pooled)/np.maximum(rows, 1), np.nan)
    return np.nanmean(np.where(mask[None, :], recall, np.nan), axis=1)


def seed_arrays(frame, column, units, weights):
    """Metric arrays (seeds, 1+D) plus the per-seed pooled confusion (D,8,8)."""
    out, pooled = {}, []
    for seed in SEEDS:
        part = frame[frame.seed == seed]
        t = tensor(part, column, units)
        for name, value in all_metrics(t, weights).items():
            out.setdefault(name, []).append(value)
        pooled.append(np.einsum("dg,gtp->dtp", weights, t))
    return {k: np.array(v) for k, v in out.items()}, np.array(pooled)       # pooled: (seeds, D, 8, 8)


def main():
    split = pd.concat([pd.read_csv(OUT/f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    leak_paper = pd.concat([pd.read_csv(DATA/"leakage"/f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)
    units = sorted(set(split.Serial_Number))
    assert len(units) == 54
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.vstack([np.ones((1, 54)), rng.multinomial(54, np.full(54, 1/54), size=DRAWS).astype(float)])
    rows, summary = [], {}
    for model, name in MODELS.items():
        a, pa = seed_arrays(split[split.split == "leak"], model, units, weights)
        b, pb = seed_arrays(split[split.split == "unseen"], model, units, weights)
        rnd, _ = seed_arrays(leak_paper[leak_paper.arm == "random_stratified"], model, units, weights)
        grp, _ = seed_arrays(leak_paper[leak_paper.arm == "grouped_faultmode"], model, units, weights)
        # balanced accuracy over the classes present in both pooled evaluation sets
        mask = (pa[:, 0].sum(axis=0).sum(axis=1) > 0) & (pb[:, 0].sum(axis=0).sum(axis=1) > 0)
        ba_a, ba_b = balanced(pa.sum(axis=0), mask), balanced(pb.sum(axis=0), mask)
        a["BalancedAcc"], b["BalancedAcc"] = ba_a[None, :], ba_b[None, :]
        for metric in METRICS + ["BalancedAcc"]:
            la, lb = a[metric].mean(axis=0), b[metric].mean(axis=0)
            diff = la-lb
            gap = rnd[metric].mean(axis=0)-grp[metric].mean(axis=0) if metric in METRICS else None
            row = dict(model=name, metric=metric, leak=la[0], unseen=lb[0], diff=diff[0],
                       diff_lo=np.percentile(diff[1:], 2.5), diff_hi=np.percentile(diff[1:], 97.5))
            if gap is not None:
                row.update(paper_random=rnd[metric].mean(axis=0)[0], paper_grouped=grp[metric].mean(axis=0)[0],
                           paper_gap=gap[0], paper_gap_lo=np.percentile(gap[1:], 2.5),
                           paper_gap_hi=np.percentile(gap[1:], 97.5),
                           fraction=diff[0]/gap[0] if gap[0] != 0 else np.nan)
            rows.append(row)
        summary[name] = dict(common_classes=[c for c, m in zip(FAULT_ORDER, mask) if m])
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"contrast.csv", index=False)
    infos = [json.loads((OUT/f"info_seed{s}.json").read_text()) for s in SEEDS]
    dropped = int(sum(sum(i["dropped_reserved_cycles"]) for i in infos))
    counts = split.groupby("split").size().to_dict()
    primary = table[table.metric == "Accuracy"].set_index("model")
    fractions = {m: float(primary.loc[m, "fraction"]) for m in MODELS.values()}
    def reading(f):
        return "f >= 0,8" if f >= 0.8 else "0,4 <= f < 0,8" if f >= 0.4 else "f < 0,4"
    verdict = dict(fractions=fractions, readings={m: reading(f) for m, f in fractions.items()},
                   dropped_reserved_cycles=dropped, rows_by_split={k: int(v) for k, v in counts.items()}, **summary)
    (OUT/"verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(table, verdict)
    write_tex(table)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def write_md(table, verdict):
    lines = ["# E9 — Contraste de vazamento com modelo fixo", "",
             "Protocolo `docs/protocolos/E9.md`. Mesmo modelo avaliado em ciclos reservados de unidades vistas "
             "(vazamento) e em unidades não vistas; dez sementes; intervalos de SN descritivos e não calibrados.", "",
             f"Ciclos reservados excluídos por a unidade não ter ciclo no ajuste: {verdict['dropped_reserved_cycles']}. "
             f"Linhas por conjunto: {verdict['rows_by_split']}.", "",
             "| modelo | métrica | vazamento | não vistas | diferença [IC] | aleatório − agrupado (manuscrito) [IC] | fração f |",
             "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        gap = "—" if pd.isna(getattr(r, "paper_gap", np.nan)) else f"{f3(r.paper_gap)} [{f3(r.paper_gap_lo)}; {f3(r.paper_gap_hi)}]"
        frac = "—" if pd.isna(getattr(r, "fraction", np.nan)) else f"{r.fraction:.2f}"
        lines.append(f"| {r.model} | {r.metric} | {f3(r.leak)} | {f3(r.unseen)} | {f3(r.diff)} [{f3(r.diff_lo)}; {f3(r.diff_hi)}] | {gap} | {frac} |")
    lines += ["", "## Leitura pré-especificada (acurácia)", ""]
    for m, f in verdict["fractions"].items():
        lines.append(f"- {m}: f = {f:.2f} → {verdict['readings'][m]}")
    (DATA/"e9_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def write_tex(table):
    def n(x):
        return f"{x:.3f}".replace("-", "$-$")
    rows = []
    for r in table.itertuples():
        gap = "---" if pd.isna(getattr(r, "paper_gap", np.nan)) else n(r.paper_gap)
        frac = "---" if pd.isna(getattr(r, "fraction", np.nan)) else f"{r.fraction:.2f}"
        metric = r.metric.replace("_", r"\_")
        rows.append(f"{r.model} & {metric} & {n(r.leak)} & {n(r.unseen)} & "
                    f"{n(r.diff)} [{n(r.diff_lo)}; {n(r.diff_hi)}] & {gap} & {frac} \\\\")
    text = ("% Source: data/e9_fixed_model/contrast.csv (E9.md).\n" + r"\begin{tabular}{llcccrc}" + "\n" + r"\toprule" + "\n"
            + r"Model & Metric & Seen units & Unseen units & Difference [interval] & Manuscript gap & Fraction \\" + "\n"
            + r"\midrule" + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    (ROOT/"paper/supplement/tables/tab_e9_fixed_model.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
