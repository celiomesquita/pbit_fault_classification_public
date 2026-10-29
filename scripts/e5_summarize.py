"""E5: summary of the fully nested leave-one-unit-out runs (docs/protocolos/E5.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"data/e5_nested_loso"
TARGETS = ["SN 045", "SN 052", "SN 054", "SN 056"]
OSCILLATORY = ["SN 052", "SN 054", "SN 056"]
SEEDS = [42, 0, 7, 13, 21]
ARMS = ["untuned", "F1_tuned", "CMR_tuned"]
SEV3 = {"F03", "F06"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    freeze = json.loads((OUT/"protocol_freeze.json").read_text(encoding="utf-8-sig"))
    for rel, digest in freeze["sha256"].items():
        assert sha(ROOT/rel) == digest, f"{rel} changed after freezing"
    audit = pd.read_csv(ROOT/"data/audit_dataset_r1.csv")
    keep = set(audit.loc[audit["in_D_primary"], "Experiment_Index"])
    features = pd.read_csv(ROOT/"data/PBIT_features_r1.csv", usecols=["Experiment_Index", "Serial_Number", "Fault_Mode"])
    features = features[features.Experiment_Index.isin(keep)]
    rows, params, checks = [], [], []
    for target in TARGETS:
        for seed in SEEDS:
            run = OUT/"runs"/f"{target.replace(' ', '')}_seed{seed}"
            manifest = json.loads((run/"run_manifest.json").read_text(encoding="utf-8"))
            assert manifest["target"] == target and manifest["seed"] == seed and not manifest["smoke"]
            assert manifest["trials"] == 100 and manifest["folds"] == 5
            assert manifest["checks"]["target_absent_from_all_training_and_selection"]
            for rel, digest in manifest["source_sha256"].items():
                if rel != "scripts/loso_f06_nested.py":     # frozen separately above
                    assert sha(ROOT/rel) == digest or rel == "docs/protocolos/E5.md", rel
            pred = pd.read_csv(run/"predictions.csv")
            expected = set(features[features.Serial_Number == target].Experiment_Index)
            assert set(pred.Experiment_Index) == expected and len(pred) == len(expected)
            assert manifest["training"]["cycles"] == 1487-len(expected)
            checks.append(dict(target=target, seed=seed, training_cycles=manifest["training"]["cycles"],
                               F06_units_in_training=manifest["training"]["F06_units"]))
            for arm in ARMS:
                for r in pred.itertuples():
                    rows.append(dict(target=target, seed=seed, arm=arm, Experiment_Index=r.Experiment_Index,
                                     pred=getattr(r, arm)))
            for s in manifest["selections"]:
                params.append(dict(target=target, seed=seed, arm=s["arm"], search_failed=s["search_failed"],
                                   feasible_trials=s["feasible_trials"], sev3_boost=s["sev3_boost"],
                                   f05_boost=s["f05_boost"], n_estimators=s["parameters"]["n_estimators"],
                                   num_leaves=s["parameters"]["num_leaves"],
                                   learning_rate=s["parameters"]["learning_rate"]))
    frame = pd.DataFrame(rows)
    # the endpoint concerns the F06 cycles of each unit (the unit also holds cycles of other classes)
    f06_cycles = set(features[features.Fault_Mode == "F06"].Experiment_Index)
    frame = frame[frame.Experiment_Index.isin(f06_cycles)]
    frame["is_F06"] = frame.pred == "F06"
    frame["is_sev3"] = frame.pred.isin(SEV3)
    by_seed = frame.groupby(["target", "arm", "seed"]).agg(F06=("is_F06", "mean"), sev3=("is_sev3", "mean")).reset_index()
    by_seed.to_csv(OUT/"summary_by_seed.csv", index=False)
    result = []
    for (target, arm), g in by_seed.groupby(["target", "arm"]):
        classes = frame[(frame.target == target) & (frame.arm == arm)].pred.value_counts().to_dict()
        result.append(dict(target=target, arm=arm, seeds_F06=float(g.F06.mean()), seeds_sev3=float(g.sev3.mean()),
                           seeds_with_majority_sev3=int((g.sev3 >= 0.5).sum()), classes=classes))
    result = pd.DataFrame(result)
    result.to_csv(OUT/"summary_by_target_arm.csv", index=False)
    pd.DataFrame(params).to_csv(OUT/"selected_parameters.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT/"training_composition.csv", index=False)
    rule = {}
    for arm in ["F1_tuned", "CMR_tuned"]:
        sub = result[(result.arm == arm) & result.target.isin(OSCILLATORY)]
        rule[arm] = {r.target: int(r.seeds_with_majority_sev3) for r in sub.itertuples()}
    preserved = all(v >= 4 for arm in rule.values() for v in arm.values())
    f06_max = int(round(5*result[result.arm.isin(["F1_tuned", "CMR_tuned"])].seeds_F06.max()))
    verdict = dict(runs=len(checks), all_checks_passed=True, severity3_seed_counts_oscillatory=rule,
                   preserved_in_at_least_4_of_5=preserved, max_F06_exact_seed_equivalents=f06_max,
                   search_failures=int(pd.DataFrame(params).search_failed.sum()))
    (OUT/"verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_md(result, pd.DataFrame(params), verdict)
    write_tex(result)
    print(json.dumps(verdict, indent=2))


def write_md(result, params, verdict):
    lines = ["# E5 — LOSO de F06 totalmente aninhado", "",
             "Protocolo `docs/protocolos/E5.md`. 4 unidades × 5 sementes, 100 tentativas por braço, "
             "alvo ausente de todo treino e seleção. Frações agrupadas sobre os ciclos F06 de cada unidade e médias "
             "sobre as cinco sementes; sem testes nem intervalos.", "",
             "| unidade | braço | F06 (média das sementes) | severidade 3 (média) | sementes com severidade 3 em >= 50% dos ciclos | classes previstas (todos os ciclos × sementes) |",
             "|---|---|---|---|---|---|"]
    for r in result.itertuples():
        lines.append(f"| {r.target} | {r.arm} | {r.seeds_F06:.2f} | {r.seeds_sev3:.2f} | {r.seeds_with_majority_sev3}/5 | {r.classes} |")
    lines += ["", "## Regra pré-especificada", "",
              f"Sementes (de 5) com severidade 3 em cada unidade oscilatória: {verdict['severity3_seed_counts_oscillatory']}.",
              f"Preservação em pelo menos 4 de 5 sementes em todas as unidades oscilatórias e braços: "
              f"{'SIM' if verdict['preserved_in_at_least_4_of_5'] else 'NÃO'}.",
              f"Falhas de busca (nenhuma tentativa elegível): {verdict['search_failures']} de {len(params)} seleções.", ""]
    summary = params[params.arm != "untuned"].groupby("arm").agg(
        sev3_boost_med=("sev3_boost", "median"), sev3_boost_min=("sev3_boost", "min"), sev3_boost_max=("sev3_boost", "max"),
        feasible_trials_med=("feasible_trials", "median"))
    lines += ["## Parâmetros escolhidos (resumo)", "", summary.round(2).to_markdown(), ""]
    (ROOT/"data/e5_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def write_tex(result):
    names = {"SN 045": r"SN~045 (flat, 4 cycles)", "SN 052": r"SN~052 (oscillatory)",
             "SN 054": r"SN~054 (oscillatory)", "SN 056": r"SN~056 (oscillatory)"}
    rows = []
    for target in TARGETS:
        cells = []
        for arm in ARMS:
            r = result[(result.target == target) & (result.arm == arm)].iloc[0]
            cells.append(f"{r.seeds_F06:.2f} / {r.seeds_sev3:.2f}")
        rows.append(f"{names[target]} & " + " & ".join(cells) + r" \\")
    header = (r"\makecell{\textbf{Held-out unit}\\\textbf{(archetype)}} & "
              r"\makecell{\textbf{LightGBM}\\\textbf{untuned}} & "
              r"\makecell{\textbf{LightGBM}\\\textbf{F1-tuned, nested}} & "
              r"\makecell{\textbf{LightGBM}\\\textbf{CMR-tuned, nested}} \\")
    text = ("% Source: data/e5_summary.md (scripts/loso_f06_nested.py, scripts/e5_summarize.py); each cell: fraction of F06 "
            "cycles predicted F06 / predicted F03 or F06 (Sev3), mean over five seeds.\n"
            + r"\begin{tabular}{lccc}" + "\n" + r"\toprule" + "\n" + header + "\n" + r"\midrule" + "\n"
            + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    (ROOT/"paper/manuscript/tables/tab_loso_nested.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
