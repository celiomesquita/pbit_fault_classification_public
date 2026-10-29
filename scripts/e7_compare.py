"""E7: historical vs audited population comparison, following docs/protocolos/E7.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from e1_external_analysis import FAULT_ORDER, SEVERITY, confusion_by_unit, metrics_from

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
OUT = DATA/"e7_audited"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
DRAWS, BOOT_SEED = 10000, 20261027
MARGIN = {"CMR": 0.02, "DR": 0.02, "FA0": 0.03, "macro_F1": 0.03, "Accuracy": 0.03, "SE": 0.03,
          "R3": 0.03, "Lbar": 0.02, "FA0_unit": 0.03}
LEVEL_METRICS = ["Accuracy", "macro_F1", "CMR", "FA0", "DR", "SE", "R3"]


def block_frame(population, block):
    base = DATA if population == "historical" else OUT
    return pd.concat([pd.read_csv(base/block/f"preds_seed{s}.csv") for s in SEEDS], ignore_index=True)


def all_metrics(tensor, weights):
    values = metrics_from(tensor, weights)
    pooled = np.einsum("dg,gtp->dtp", weights, tensor)
    total = pooled.sum(axis=(1, 2))
    values["Accuracy"] = np.einsum("dtt->d", pooled)/total
    values["SE"] = (pooled*(SEVERITY[:, None] != SEVERITY[None, :])).sum(axis=(1, 2))/total
    return values


class Cache:
    def __init__(self, weights):
        self.weights, self.frames, self.store = weights, {}, {}

    def series(self, population, block, column, arm=None):
        key = (population, block, column, arm)
        if key not in self.store:
            frame = self.frames.setdefault((population, block), block_frame(population, block))
            if arm is not None:
                frame = frame[frame.arm == arm]
            runs = {}
            for seed, part in frame.groupby("seed"):
                tensor, _ = confusion_by_unit(part.Fault_Mode.tolist(), part[column].tolist(),
                                              part.Serial_Number.tolist())
                for name, value in all_metrics(tensor, self.weights).items():
                    runs.setdefault(name, []).append(value)
            self.store[key] = {k: np.array(v) for k, v in runs.items()}   # (seeds, 1+D)
        return self.store[key]


def category(lo, hi, margin):
    if lo > -margin and hi < margin:
        return "within_margin"
    return "straddles_zero" if lo <= 0 <= hi else "excludes_zero"


def levels_spec():
    spec = []
    for model, name in (("lgbm", "LightGBM"), ("rf", "Random Forest")):
        spec += [("leakage", f"{name}, grouped", model, "grouped_faultmode"),
                 ("leakage", f"{name}, random", model, "random_stratified")]
    spec += [("families", "XGBoost", "xgb_F246", None), ("families", "Gradient Boosting", "gb_F246", None)]
    for fs in ("F246", "F115", "F18"):
        spec += [("ablation", f"LightGBM {fs}", f"lgbm_{fs}", None), ("ablation", f"Random Forest {fs}", f"rf_{fs}", None)]
    spec += [("ablation", "B4 threshold", "B4_threshold_safety", None)]
    cols = pd.read_csv(DATA/"imbalance/preds_seed42.csv", nrows=1).columns
    for c in cols:
        if c.startswith(("lgbm_", "rf_")):
            spec.append(("imbalance", c, c, None))
    return spec


def contrast_spec():
    c = []
    for model, name in (("lgbm", "LightGBM"), ("rf", "Random Forest")):
        for metric in ("Accuracy", "macro_F1", "CMR", "FA0", "DR"):
            c.append(("leakage", f"{name}: random - grouped", (model, "random_stratified"), (model, "grouped_faultmode"), metric))
    for other, label in (("lgbm_F246", "LightGBM"), ("xgb_F246", "XGBoost")):
        for metric in ("CMR", "FA0"):
            c.append(("families", f"Random Forest - {label}", ("rf_F246", None, "ablation" if other.startswith("lgbm") else "ablation"),
                      (other, None, "ablation" if other.startswith("lgbm") else "families"), metric))
    for model, name in (("lgbm", "LightGBM"), ("rf", "Random Forest")):
        for other in ("F115", "F18"):
            for metric in ("CMR", "FA0", "DR", "macro_F1"):
                c.append(("ablation", f"{name}: F246 - {other}", (f"{model}_F246", None), (f"{model}_{other}", None), metric))
    c += [("imbalance", "LightGBM: product - frequency weights (Gaussian)", ("lgbm_freq_sev_gaussian", None),
           ("lgbm_freq_gaussian", None), "CMR"),
          ("imbalance", "LightGBM: Gaussian - none (product weights)", ("lgbm_freq_sev_gaussian", None),
           ("lgbm_freq_sev_none", None), "macro_F1"),
          ("imbalance", "LightGBM: Gaussian - SMOTE (product weights)", ("lgbm_freq_sev_gaussian", None),
           ("lgbm_freq_sev_smote", None), "macro_F1")]
    return c


def main():
    units = sorted(set(block_frame("historical", "ablation").Serial_Number))
    assert units == sorted(set(block_frame("audited", "ablation").Serial_Number)) and len(units) == 54
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.vstack([np.ones((1, len(units))),
                         rng.multinomial(len(units), np.full(len(units), 1/len(units)), size=DRAWS).astype(float)])
    cache = Cache(weights)
    levels = []
    for block, label, column, arm in levels_spec():
        h = cache.series("historical", block, column, arm)
        a = cache.series("audited", block, column, arm)
        for metric in LEVEL_METRICS:
            hs, as_ = h[metric][:, 0], a[metric][:, 0]
            levels.append(dict(block=block, quantity=label, metric=metric, hist_mean=hs.mean(), hist_min=hs.min(),
                               hist_max=hs.max(), aud_mean=as_.mean(), diff=as_.mean()-hs.mean(),
                               inside_hist_range=bool(hs.min()-1e-12 <= as_.mean() <= hs.max()+1e-12)))
    # families RF/LightGBM live in the ablation file, XGBoost/GB in the families file
    contrasts = []
    for block, label, a_spec, b_spec, metric in contrast_spec():
        row = dict(block=block, contrast=label, metric=metric)
        for population in ("historical", "audited"):
            def get(spec):
                column, arm = spec[0], spec[1]
                source = spec[2] if len(spec) > 2 else block
                return cache.series(population, source, column, arm)[metric].mean(axis=0)
            diff = get(a_spec)-get(b_spec)
            lo, hi = np.percentile(diff[1:], [2.5, 97.5])
            tag = "hist" if population == "historical" else "aud"
            row.update({f"{tag}_diff": diff[0], f"{tag}_lo": lo, f"{tag}_hi": hi,
                        f"{tag}_category": category(lo, hi, MARGIN[metric])})
        row["same_sign"] = bool(np.sign(row["hist_diff"]) == np.sign(row["aud_diff"]))
        row["same_category"] = row["hist_category"] == row["aud_category"]
        contrasts.append(row)
    levels, contrasts = pd.DataFrame(levels), pd.DataFrame(contrasts)
    levels.to_csv(OUT/"levels_comparison.csv", index=False)
    contrasts.to_csv(OUT/"contrasts_comparison.csv", index=False)
    verdict = {}
    for block in ("leakage", "families", "ablation", "imbalance"):
        lv, ct = levels[levels.block == block], contrasts[contrasts.block == block]
        verdict[block] = dict(values_stable=bool(lv.inside_hist_range.all()),
                              conclusions_stable=bool((ct.same_sign & ct.same_category).all()),
                              out_of_range=int((~lv.inside_hist_range).sum()), of=int(len(lv)),
                              changed_contrasts=int((~(ct.same_sign & ct.same_category)).sum()), contrasts=int(len(ct)))
    (OUT/"verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_summary(levels, contrasts, verdict)
    print(json.dumps(verdict, indent=2))


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def write_summary(levels, contrasts, verdict):
    lines = ["# E7 — Comparações principais: população histórica (1.196) vs auditada (1.195)", "",
             "Protocolo `docs/protocolos/E7.md`. Dez sementes, mesmas configurações; única mudança: exclusão do "
             "experimento 1107. Intervalos: reamostragem pareada de SN, descritivos e não calibrados.", "",
             "## Veredito por bloco (regra fixada antes dos resultados)", "",
             "| bloco | valores dentro da amplitude histórica por semente | conclusões dos contrastes mantidas |",
             "|---|---|---|"]
    for block, v in verdict.items():
        lines.append(f"| {block} | {'sim' if v['values_stable'] else 'NÃO'} ({v['of']-v['out_of_range']}/{v['of']}) | "
                     f"{'sim' if v['conclusions_stable'] else 'NÃO'} ({v['contrasts']-v['changed_contrasts']}/{v['contrasts']}) |")
    lines += ["", "## Contrastes", "",
              "| bloco | contraste | métrica | histórico [IC] | cat. | auditado [IC] | cat. | mantido |", "|---|---|---|---|---|---|---|---|"]
    for r in contrasts.itertuples():
        lines.append(f"| {r.block} | {r.contrast} | {r.metric} | {f3(r.hist_diff)} [{f3(r.hist_lo)}; {f3(r.hist_hi)}] | "
                     f"{r.hist_category} | {f3(r.aud_diff)} [{f3(r.aud_lo)}; {f3(r.aud_hi)}] | {r.aud_category} | "
                     f"{'sim' if r.same_sign and r.same_category else 'NÃO'} |")
    lines += ["", "## Níveis fora da amplitude histórica por semente", "",
              "| bloco | quantidade | métrica | histórico (média; mín–máx) | auditado | diferença |", "|---|---|---|---|---|---|"]
    out = levels[~levels.inside_hist_range]
    for r in out.itertuples():
        lines.append(f"| {r.block} | {r.quantity} | {r.metric} | {f3(r.hist_mean)} ({f3(r.hist_min)}–{f3(r.hist_max)}) | "
                     f"{f3(r.aud_mean)} | {f3(r.diff)} |")
    if out.empty:
        lines.append("| — | nenhum | — | — | — | — |")
    lines += ["", "## Maiores diferenças de nível (auditado − histórico)", "",
              "| bloco | quantidade | métrica | histórico | auditado | diferença |", "|---|---|---|---|---|---|"]
    top = levels.reindex(levels["diff"].abs().sort_values(ascending=False).index).head(12)
    for r in top.itertuples():
        lines.append(f"| {r.block} | {r.quantity} | {r.metric} | {f3(r.hist_mean)} | {f3(r.aud_mean)} | {f3(r.diff)} |")
    (DATA/"e7_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")




def write_tex(contrasts):
    def n(x):
        return f"{x:.3f}".replace("-", "$-$")
    rows = []
    for r in contrasts.itertuples():
        rows.append(f"{r.block} & {r.contrast.replace('_', chr(92)+'_')} & {r.metric.replace('_', chr(92)+'_')} & "
                    f"{n(r.hist_diff)} [{n(r.hist_lo)}; {n(r.hist_hi)}] & {n(r.aud_diff)} [{n(r.aud_lo)}; {n(r.aud_hi)}] & "
                    f"{'yes' if r.same_sign and r.same_category else 'no'} \\\\")
    text = ("% Source: data/e7_audited/contrasts_comparison.csv (E7.md).\n"
            + r"\begin{tabular}{lllccc}" + "\n" + r"\toprule" + "\n"
            + r"Block & Contrast & Metric & Historical [interval] & Audited [interval] & Same \\" + "\n"
            + r"\midrule" + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    (ROOT/"paper/supplement/tables/tab_e7_contrasts.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
    write_tex(pd.read_csv(OUT/"contrasts_comparison.csv"))
