"""E1 external analysis (descriptive), following docs/protocolos/E1_externo.md."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/protocolos/E1_externo.md"
FREEZE = ROOT / "data/e1_external_freeze/protocol_freeze.json"
OUT = ROOT / "data/e1_external"
BATCH = ROOT / "data/e1_full_v1"
FAULT_ORDER = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]
SEVERITY = np.array([0, 0, 1, 2, 3, 2, 2, 3])
ARMS = ["untuned", "macro_F1_fixed", "macro_F1_optimized", "CMR_fixed", "CMR_optimized",
        "minimum_cost", "metacost"]
REFERENCE = "untuned"
BUDGETS = [0.40, 0.30, 0.50]
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
DRAWS = 10000
BOOT_SEED = 20261027
LOW_M, LOW_F = 0.02, 0.03
EPS = 1e-12
METRICS = ["CMR", "Lbar", "FA0", "FA0_unit", "DR", "R3", "macro_F1"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suffix(budget):
    return f"b{int(round(budget*100))}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def classify(delta_main, delta_fa0):
    """Descriptive label from point differences (protocol table)."""
    m = "menor" if delta_main <= -LOW_M+EPS else "maior" if delta_main >= LOW_M-EPS else "semelhante"
    f = "menor" if delta_fa0 <= -LOW_F+EPS else "maior" if delta_fa0 >= LOW_F-EPS else "semelhante"
    if m == "menor" and f == "maior":
        label = "Deslocamento ao longo do compromisso"
    elif (m == "menor" and f in ("semelhante", "menor")) or (m == "semelhante" and f == "menor"):
        label = "Melhoria aparente do compromisso (não confirmada)"
    elif m == "semelhante" and f == "semelhante":
        label = "Sem diferença apreciável"
    elif m in ("semelhante", "maior") and f == "maior":
        label = "Piora aparente"
    else:
        label = "Deslocamento inverso"
    return label, m, f


def confusion_by_unit(truth, pred, units):
    """(G,8,8) counts per SN, SN order = sorted unique units."""
    order = sorted(set(units))
    g = {u: i for i, u in enumerate(order)}
    index = {c: i for i, c in enumerate(FAULT_ORDER)}
    tensor = np.zeros((len(order), 8, 8))
    np.add.at(tensor, ([g[u] for u in units], [index[t] for t in truth], [index[p] for p in pred]), 1)
    return tensor, order


def metrics_from(tensor, weights):
    """Metrics for weights (D,G): pooled and unit-level, from per-SN confusion tensors."""
    crit, norm = SEVERITY == 3, SEVERITY == 0
    crit_n = tensor[:, crit, :].sum(axis=(1, 2))
    crit_miss = tensor[:, crit][:, :, SEVERITY == 0].sum(axis=(1, 2))
    crit_hit3 = tensor[:, crit][:, :, SEVERITY == 3].sum(axis=(1, 2))
    norm_n = tensor[:, norm, :].sum(axis=(1, 2))
    norm_alarm = tensor[:, norm][:, :, SEVERITY >= 1].sum(axis=(1, 2))
    total = tensor.sum(axis=(1, 2))
    down = (tensor * (SEVERITY[None, None, :] < SEVERITY[None, :, None])).sum(axis=(1, 2))
    has_c, has_n = crit_n > 0, norm_n > 0
    loss = np.where(has_c, crit_miss/np.maximum(crit_n, 1), 0.0)
    alarm = np.where(has_n, norm_alarm/np.maximum(norm_n, 1), 0.0)
    pooled = np.einsum("dg,gtp->dtp", weights, tensor)
    tp = np.einsum("dtt->dt", pooled)
    rows, cols = pooled.sum(axis=2), pooled.sum(axis=1)
    present = (rows+cols) > 0
    f1 = np.where(present, 2*tp/np.maximum(rows+cols, 1e-300), 0.0)
    return {"CMR": weights@crit_miss/(weights@crit_n), "Lbar": weights@loss/(weights@has_c),
            "FA0": weights@norm_alarm/(weights@norm_n), "FA0_unit": weights@alarm/(weights@has_n),
            "DR": weights@down/(weights@total), "R3": weights@crit_hit3/(weights@crit_n),
            "macro_F1": f1.sum(axis=1)/present.sum(axis=1)}


def refuse_if_computed():
    assert not OUT.exists(), "data/e1_external/ already exists: external metrics were computed before"


def check_freeze():
    freeze = read_json(FREEZE)
    assert freeze["protocol_sha256"] == sha(PROTOCOL), "E1_externo.md changed after freezing"
    assert freeze["completion_manifest_sha256"] == sha(BATCH / "completion_manifest.json")
    assert freeze["batch_freeze_sha256"] == sha(BATCH / "batch_freeze.json")
    return freeze


def load_predictions():
    manifest = read_json(BATCH / "completion_manifest.json")
    assert len(manifest["completed"]) == 50
    frames, decisions, hashes = {}, [], {}
    for item in manifest["completed"]:
        folder = ROOT / item["output"]
        for key, name in [("run_manifest_sha256", "run_manifest.json"),
                          ("verification_sha256", "verification.json"),
                          ("model_verification_sha256", "model_verification.json")]:
            assert sha(folder / name) == item[key], f"{name} differs in {folder}"
        data = pd.read_csv(folder / "external_predictions.csv")
        data["seed"], data["outer"] = item["seed"], item["outer"]
        probability = np.load(folder / "untuned_external_probability.npy")
        assert probability.shape == (len(data), 8)
        assert (np.array(FAULT_ORDER)[probability.argmax(axis=1)] == data["untuned_base"].to_numpy()).all()
        data[[f"p_{c}" for c in FAULT_ORDER]] = probability
        frames.setdefault(item["seed"], []).append(data)
        for row in read_json(folder / "decisions.json"):
            decisions.append(dict(seed=item["seed"], outer=item["outer"], arm=row["arm"],
                                  budget=row["budget"], failed=bool(row["budget_failed"])))
        hashes[item["output"]] = item["run_manifest_sha256"]
    pooled = {}
    for seed, parts in frames.items():
        data = pd.concat(parts, ignore_index=True)
        assert len(parts) == 5 and data.Experiment_Index.is_unique, f"seed {seed}"
        assert data.groupby("Serial_Number").outer.nunique().eq(1).all(), f"SN across folds, seed {seed}"
        pooled[seed] = data
    assert sorted(pooled) == sorted(SEEDS)
    return pooled, pd.DataFrame(decisions), hashes


def policy_column(arm, budget):
    return f"{arm}_{suffix(budget)}"


def retrospective_curve_cmr(data, fa0_limit):
    """Lowest CMR on the untuned threshold curve (external labels) with FA0 <= limit."""
    truth = data.Fault_Mode.to_numpy()
    sev_true = SEVERITY[[FAULT_ORDER.index(t) for t in truth]]
    base = data["untuned_base"].to_numpy()
    probs = data[[f"p_{c}" for c in FAULT_ORDER]].to_numpy()
    score = probs[:, 4]+probs[:, 7]
    critical = np.where(probs[:, 4] >= probs[:, 7], "F03", "F06")
    taus = np.concatenate([[np.inf], np.unique(score)[::-1], [0.0]])
    sev_base = SEVERITY[[FAULT_ORDER.index(p) for p in base]]
    sev_critical = np.full(len(data), 3)
    crit, norm = sev_true == 3, sev_true == 0
    best = np.inf
    for tau in taus:
        promote = score >= tau
        sev_pred = np.where(promote, sev_critical, sev_base)
        fa0 = ((sev_pred >= 1) & norm).sum()/norm.sum()
        if fa0 <= fa0_limit+EPS:
            best = min(best, ((sev_pred == 0) & crit).sum()/crit.sum())
    return best


def run():
    refuse_if_computed()
    freeze = check_freeze()
    pooled, decisions, hashes = load_predictions()
    OUT.mkdir(parents=True)
    units = sorted(set(pooled[SEEDS[0]].Serial_Number))
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.multinomial(len(units), np.full(len(units), 1/len(units)), size=DRAWS).astype(float)
    weights = np.vstack([np.ones((1, len(units))), draws])
    cost_arms = ("minimum_cost", "metacost")  # no threshold-free base column in external_predictions.csv
    keys = [(arm, col) for arm in ARMS for col in ([] if arm in cost_arms else ["base"])+[suffix(b) for b in BUDGETS]]
    values = {}
    seed_rows = []
    retro = []
    for seed in SEEDS:
        data = pooled[seed]
        for arm, col in keys:
            tensor, order = confusion_by_unit(data.Fault_Mode.tolist(), data[f"{arm}_{col}"].tolist(),
                                              data.Serial_Number.tolist())
            assert order == units
            m = metrics_from(tensor, weights)
            for name in METRICS:
                values.setdefault((arm, col, name), []).append(m[name])
            seed_rows.append(dict(seed=seed, arm=arm, policy=col, **{k: float(v[0]) for k, v in m.items()}))
            if col != "base" and arm != REFERENCE:
                retro.append(dict(seed=seed, arm=arm, policy=col,
                                  curve_CMR=retrospective_curve_cmr(data, float(m["FA0"][0]))))
    by_seed = pd.DataFrame(seed_rows)
    stacked = {k: np.array(v) for k, v in values.items()}       # (S, 1+D)
    mean = {k: v.mean(axis=0) for k, v in stacked.items()}
    availability, results, labels = [], [], []
    for arm in ARMS:
        for budget in BUDGETS:
            sub = decisions[(decisions.arm == arm) & np.isclose(decisions.budget, budget)]
            assert len(sub) == 50
            col = suffix(budget)
            seeds_over = int((by_seed[(by_seed.arm == arm) & (by_seed.policy == col)].FA0 > budget+EPS).sum())
            availability.append(dict(arm=arm, budget=budget, failed_folds=int(sub.failed.sum()),
                                     seeds_FA0_above_budget=seeds_over,
                                     mostly_unavailable=bool(sub.failed.sum() >= 25)))
    availability = pd.DataFrame(availability)
    retro = pd.DataFrame(retro)
    for arm in ARMS:
        for budget in BUDGETS:
            col = suffix(budget)
            for name in METRICS:
                level = mean[(arm, col, name)]
                row = dict(arm=arm, budget=budget, metric=name, value=float(level[0]),
                           value_lo=float(np.percentile(level[1:], 2.5)),
                           value_hi=float(np.percentile(level[1:], 97.5)),
                           seed_min=float(stacked[(arm, col, name)][:, 0].min()),
                           seed_max=float(stacked[(arm, col, name)][:, 0].max()))
                if arm != REFERENCE:
                    diff = level - mean[(REFERENCE, col, name)]
                    seed_diff = stacked[(arm, col, name)][:, 0]-stacked[(REFERENCE, col, name)][:, 0]
                    row.update(diff=float(diff[0]), diff_lo=float(np.percentile(diff[1:], 2.5)),
                               diff_hi=float(np.percentile(diff[1:], 97.5)),
                               seeds_same_sign=int((np.sign(seed_diff) == np.sign(diff[0])).sum()
                                                   if diff[0] != 0 else 0))
                results.append(row)
            if arm == REFERENCE:
                continue
            table = pd.DataFrame([r for r in results if r["arm"] == arm and r["budget"] == budget]).set_index("metric")
            for main in ("CMR", "Lbar"):
                label, m_class, f_class = classify(table.loc[main, "diff"], table.loc["FA0", "diff"])
                flags = []
                for endpoint, cls in ((main, m_class), ("FA0", f_class)):
                    if cls != "semelhante":
                        r = table.loc[endpoint]
                        flags.append(r.seeds_same_sign >= 8 and (r.diff_lo > 0 or r.diff_hi < 0))
                consistency = "—" if label == "Sem diferença apreciável" else (
                    "consistente" if flags and all(flags) else "instável")
                labels.append(dict(arm=arm, budget=budget, main_endpoint=main, label=label,
                                   consistency=consistency, delta_main=float(table.loc[main, "diff"]),
                                   delta_FA0=float(table.loc["FA0", "diff"])))
    results, labels = pd.DataFrame(results), pd.DataFrame(labels)
    by_seed.to_csv(OUT/"metrics_by_seed.csv", index=False)
    results.to_csv(OUT/"metrics_and_contrasts.csv", index=False)
    labels.to_csv(OUT/"labels.csv", index=False)
    availability.to_csv(OUT/"availability.csv", index=False)
    retro.to_csv(OUT/"retrospective_curve.csv", index=False)
    outputs = {p.name: sha(p) for p in sorted(OUT.glob("*.csv"))}
    (OUT/"run_manifest.json").write_text(json.dumps(dict(
        item="E1_externo", finished_utc=datetime.now(timezone.utc).isoformat(), protocol=freeze,
        draws=DRAWS, boot_seed=BOOT_SEED, input_run_manifest_sha256=hashes, output_sha256=outputs,
        status="complete"), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = retro_summary(by_seed, retro)
    summary.to_csv(OUT/"retrospective_summary.csv", index=False)
    write_report(results, labels, availability, summary)
    write_table(results, labels, availability)


def retro_summary(by_seed, retro):
    """Policy CMR minus the retrospective curve, over seeds where the curve is defined.

    The curve starts at the untuned model's own FA0 (no threshold) and can only add alarms, so it
    is undefined for seeds where the policy FA0 is lower than that starting point.
    """
    merged = retro.merge(by_seed[["seed", "arm", "policy", "CMR"]], on=["seed", "arm", "policy"])
    merged["defined"] = np.isfinite(merged.curve_CMR)
    rows = []
    for (arm, policy), g in merged.groupby(["arm", "policy"], sort=False):
        ok = g[g.defined]
        rows.append(dict(arm=arm, budget=int(policy[1:])/100, seeds_defined=int(len(ok)),
                         CMR_minus_curve=float((ok.CMR-ok.curve_CMR).mean()) if len(ok) else float("nan")))
    return pd.DataFrame(rows)


def rebuild():
    """Rebuild retrospective summary, report, table and manifest from the saved CSVs (no new metrics)."""
    freeze = check_freeze()
    labels = pd.read_csv(OUT/"labels.csv")
    labels = labels.drop(columns=[c for c in ("retro_curve_CMR", "CMR_minus_curve") if c in labels])
    results = pd.read_csv(OUT/"metrics_and_contrasts.csv")
    availability = pd.read_csv(OUT/"availability.csv")
    by_seed = pd.read_csv(OUT/"metrics_by_seed.csv")
    retro = pd.read_csv(OUT/"retrospective_curve.csv")
    summary = retro_summary(by_seed, retro)
    labels.to_csv(OUT/"labels.csv", index=False)
    summary.to_csv(OUT/"retrospective_summary.csv", index=False)
    manifest = read_json(OUT/"run_manifest.json")
    manifest["post_result_amendment"] = (
        "27/10/2026, after seeing results: the retrospective curve is undefined when the policy FA0 is "
        "below the untuned no-threshold FA0; the first run averaged an infinite value. Averages now use "
        "defined seeds only and report their count. No primary metric, label or interval changed.")
    manifest["output_sha256"] = {p.name: sha(p) for p in sorted(OUT.glob("*.csv"))}
    manifest["protocol"] = freeze
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(results, labels, availability, summary)
    write_table(results, labels, availability)


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def f3t(x):
    return f"{x:.3f}".replace("-", "$-$")


def write_report(results, labels, availability, summary):
    lines = ["# E1 — Análise externa descritiva", "",
             "Emenda `docs/protocolos/E1_externo.md`. Sem testes, valores-p, Holm, equivalência ou "
             "superioridade. Intervalos: reamostragem pareada de SN (10.000 sorteios), descritivos e "
             "não calibrados (E3/E3b/E3c). Médias sobre dez sementes; sementes não são réplicas independentes.", ""]
    for budget in BUDGETS:
        lines += [f"## Orçamento interno de FA0 {budget:.2f}" + (" (primário)" if budget == 0.40 else ""), "",
                  "| braço | pol. inviável (de 50) | FA0 > orç. (de 10 sem.) | CMR | L̄ | FA0 | ΔCMR [IC] | ΔL̄ [IC] | ΔFA0 [IC] | rótulo (CMR) | consist. | rótulo (L̄) | consist. |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            r = results[(results.arm == arm) & (results.budget == budget)].set_index("metric")
            a = availability[(availability.arm == arm) & np.isclose(availability.budget, budget)].iloc[0]
            def d(name):
                return "—" if arm == REFERENCE else f"{f3(r.loc[name, 'diff'])} [{f3(r.loc[name, 'diff_lo'])}; {f3(r.loc[name, 'diff_hi'])}]"
            lab = labels[(labels.arm == arm) & np.isclose(labels.budget, budget)].set_index("main_endpoint")
            cells = ["—", "—", "—", "—"] if arm == REFERENCE else [
                lab.loc["CMR", "label"], lab.loc["CMR", "consistency"], lab.loc["Lbar", "label"], lab.loc["Lbar", "consistency"]]
            flag = " (majoritariamente indisponível)" if a.mostly_unavailable else ""
            lines.append(f"| {arm}{flag} | {a.failed_folds} | {a.seeds_FA0_above_budget} | "
                         f"{f3(r.loc['CMR', 'value'])} | {f3(r.loc['Lbar', 'value'])} | {f3(r.loc['FA0', 'value'])} | "
                         f"{d('CMR')} | {d('Lbar')} | {d('FA0')} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Referência retrospectiva (usa rótulos externos)", "",
              "CMR da política menos a menor CMR da curva de limiares do `untuned` com FA0 não superior; "
              "valores negativos = abaixo da curva. Serve só para situar a política; nunca para declarar melhoria.", "",
              "A curva só acrescenta alarmes a partir do `untuned` sem limiar; é indefinida nas sementes "
              "em que a FA0 da política é menor que esse ponto (desvio pós-resultado registrado no "
              "manifesto). A média usa só as sementes definidas.", "",
              "| braço | orçamento | sementes definidas (de 10) | CMR − curva |", "|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.arm} | {r.budget:.2f} | {r.seeds_defined} | "
                     f"{'n/d' if r.seeds_defined == 0 else f3(r.CMR_minus_curve)} |")
    (ROOT/"data/e1_external_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


LABEL_EN = {"Deslocamento ao longo do compromisso": "Shift along the trade-off",
            "Melhoria aparente do compromisso (não confirmada)": "Apparent improvement (unconfirmed)",
            "Sem diferença apreciável": "No appreciable difference", "Piora aparente": "Apparent deterioration",
            "Deslocamento inverso": "Inverse shift"}
CONSISTENCY_EN = {"consistente": "consistent", "instável": "unstable", "—": "n/a"}


def esc(name):
    return name.replace("_", "\\_")


def write_table(results, labels, availability):
    budget = 0.40
    rows = []
    for arm in ARMS:
        r = results[(results.arm == arm) & np.isclose(results.budget, budget)].set_index("metric")
        a = availability[(availability.arm == arm) & np.isclose(availability.budget, budget)].iloc[0]
        head = (f"{esc(arm)} & {a.failed_folds} & {f3t(r.loc['CMR', 'value'])} & {f3t(r.loc['Lbar', 'value'])} & "
                f"{f3t(r.loc['FA0', 'value'])} & ")
        if arm == REFERENCE:
            rows.append(head + "--- & --- & --- \\\\")
        else:
            lab = labels[(labels.arm == arm) & np.isclose(labels.budget, budget)
                         & (labels.main_endpoint == "CMR")].iloc[0]
            rows.append(head + f"{f3t(r.loc['CMR', 'diff'])} & {f3t(r.loc['FA0', 'diff'])} & "
                        f"{LABEL_EN[lab.label]} ({CONSISTENCY_EN[lab.consistency]}) \\\\")
    body = "\n\t".join(rows)
    text = ("% Source: data/e1_external/metrics_and_contrasts.csv, labels.csv (E1_externo.md); budget 0.40.\n"
            "\\begin{tabular}{lccccccp{4.2cm}}\n\t\\toprule\n"
            "\t\\textbf{Policy} & \\makecell{\\textbf{Fallback}\\\\\\textbf{folds}} & \\textbf{CMR} & $\\bar L$ & "
            "\\textbf{FA0} & $\\Delta$CMR & $\\Delta$FA0 & \\textbf{Descriptive label} \\\\\n\t\\midrule\n\t"
            + body + "\n\t\\bottomrule\n\\end{tabular}\n")
    (ROOT/"paper/manuscript/tables/tab_independent_policies.tex").write_text(text, encoding="utf-8")
    write_all_budgets(results, labels, availability)


def write_all_budgets(results, labels, availability):
    rows = []
    for budget in BUDGETS:
        for arm in ARMS:
            r = results[(results.arm == arm) & np.isclose(results.budget, budget)].set_index("metric")
            a = availability[(availability.arm == arm) & np.isclose(availability.budget, budget)].iloc[0]

            def d(metric):
                if arm == REFERENCE:
                    return "---"
                return (f"{f3t(r.loc[metric, 'diff'])} [{f3t(r.loc[metric, 'diff_lo'])}; "
                        f"{f3t(r.loc[metric, 'diff_hi'])}]")
            if arm == REFERENCE:
                lab = "---"
            else:
                lab = LABEL_EN[labels[(labels.arm == arm) & np.isclose(labels.budget, budget)
                                      & (labels.main_endpoint == "CMR")].iloc[0].label]
            rows.append(f"{budget:.2f} & {esc(arm)} & {a.failed_folds} & {a.seeds_FA0_above_budget} & "
                        f"{f3t(r.loc['CMR', 'value'])} & {f3t(r.loc['Lbar', 'value'])} & "
                        f"{f3t(r.loc['FA0', 'value'])} & {d('CMR')} & {d('Lbar')} & {d('FA0')} & {lab} \\\\")
        rows.append("\\midrule")
    rows.pop()
    text = ("% Source: data/e1_external/metrics_and_contrasts.csv, labels.csv, availability.csv (E1_externo.md).\n"
            "\\begin{tabular}{llccccccccl}\n\\toprule\n"
            "Budget & Policy & \\makecell{Fallback\\\\folds} & \\makecell{FA0 $>$\\\\budget} & CMR & $\\bar L$ & FA0 & "
            "$\\Delta$CMR [interval] & $\\Delta\\bar L$ [interval] & $\\Delta$FA0 [interval] & Label (CMR) \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (ROOT/"paper/supplement/tables/tab_e1_all_budgets.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    rebuild() if parser.parse_args().rebuild else run()
