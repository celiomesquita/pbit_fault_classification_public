"""E3b: frozen coverage of restricted wild and jackknife ratio inference."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from cluster_coverage_simulation import (MARGINS, ROOT, generate, markdown, scenarios, setup, sha,
                                         utc, wilson, write_json)
from cluster_ratio_wild import (BOOTSTRAP_METHODS, METHODS, abstentions, confidence_set,
                                one_sided, rademacher, two_sided)

OUT = ROOT / "data/cluster_coverage_e3b"
PROTOCOL = ROOT / "docs/protocolos/E3b.md"
FAMILIES = [1, 3, 4, 6, 8]
HOLM = [3, 4, 6, 8]
STAGES = {"pilot": dict(seed=2026101400, replicates=32, resamples=199, width=8),
          "A": dict(seed=2026101401, replicates=5000, resamples=1999, width=500),
          "B": dict(seed=2026101402, replicates=5000, resamples=1999, width=500)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stage_dir(stage):
    return OUT / ("pilot" if stage == "pilot" else f"stage_{stage}")


def stage_methods(stage):
    if stage != "B":
        return list(METHODS)
    chosen = read_json(stage_dir("A") / "selection.json")["selected"]
    assert chosen is not None, "Stage A selected no procedure; stage B is not run"
    return [chosen]


def inputs_and_profiles():
    freeze = read_json(OUT / "protocol_freeze.json")
    assert freeze["protocol_sha256"] == sha(PROTOCOL), "E3b protocol changed after freezing"
    assert freeze["e3_simulation_sha256"] == sha(ROOT / "scripts/cluster_coverage_simulation.py")
    calibration, profiles, exposures, inputs = setup()
    for path in [PROTOCOL, OUT / "protocol_freeze.json", Path(__file__),
                 ROOT / "scripts/cluster_ratio_wild.py", ROOT / "tests/test_cluster_wild.py",
                 ROOT / "data/cluster_coverage/protocol_freeze.json"]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return calibration, profiles, exposures, inputs


def simulate(cfg, calibration, profile, stage, methods, batch):
    settings = STAGES[stage]
    replicates, resamples = settings["replicates"], settings["resamples"]
    data_rng = np.random.default_rng([settings["seed"], cfg["id"], 0])
    boot_rng = np.random.default_rng([settings["seed"], cfg["id"], 1])
    n = profile["n"].astype(float)
    contrasts = cfg["contrasts"]
    counts = generate(data_rng, replicates, contrasts, profile["n"], calibration, cfg["delta"], cfg["rho"])
    flat = counts.reshape(-1, len(n))
    hs = np.array([cfg["delta"], 0.0, -cfg["margin"], cfg["margin"]])
    arrays = {f"{m}_{key}": [] for m in methods for key in ("upper", "lower", "abstain", "internal_bad")}
    widths = {m: [] for m in methods}
    for first in range(0, len(flat), batch):
        rows = flat[first:first+batch]
        v = rademacher(boot_rng, len(rows), resamples, len(n))
        replicate = np.arange(first, first+len(rows)) // contrasts
        wanted = (np.arange(first, first+len(rows)) % contrasts == 0) & (replicate < settings["width"])
        for method in methods:
            upper, lower, bad = one_sided(method, rows, n, hs, v)
            arrays[f"{method}_upper"].append(upper)
            arrays[f"{method}_lower"].append(lower)
            arrays[f"{method}_abstain"].append(abstentions(method, rows, n))
            arrays[f"{method}_internal_bad"].append(bad)
            if wanted.any():
                bounds = confidence_set(method, rows[wanted], n, v[wanted], iterations=30)
                widths[method].append(bounds[:, 1] - bounds[:, 0])
    arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
    arrays = {key: value.reshape(replicates, contrasts, *value.shape[1:]) for key, value in arrays.items()}
    arrays["estimate"] = (flat.sum(axis=1) / n.sum()).reshape(replicates, contrasts)
    arrays["h"] = hs
    records = []
    for method in methods:
        abstain = arrays[f"{method}_abstain"]
        upper = np.where(abstain[..., None], 1.0, arrays[f"{method}_upper"])
        lower = np.where(abstain[..., None], 1.0, arrays[f"{method}_lower"])
        p_two = two_sided(upper, lower)
        width = np.concatenate(widths[method])
        widths_ok = width[np.isfinite(width)]
        arrays[f"{method}_width"] = width
        events = {"coverage95": p_two[:, 0, 0] > 0.05, "reject_zero": p_two[:, 0, 1] <= 0.05,
                  "abstain": abstain[:, 0]}
        for size in FAMILIES:
            events[f"equivalence_m{size}"] = (upper[:, 0, 2] <= 0.05/size) & (lower[:, 0, 3] <= 0.05/size)
        if cfg["delta"] == 0:
            for size in HOLM:
                events[f"holm_fwer_m{size}"] = p_two[:, :size, 1].min(axis=1) <= 0.05/size
        for event, flags in events.items():
            proportion, low, high, se = wilson(int(flags.sum()), replicates)
            records.append({**cfg, "method": method, "event": event, "count": int(flags.sum()),
                            "replicates": replicates, "resamples": resamples, "rate": proportion,
                            "wilson_low": low, "wilson_high": high, "MC_SE": se,
                            "median_finite_width": float(np.median(widths_ok)) if len(widths_ok) else None,
                            "unbounded_width_share": float(1 - len(widths_ok)/len(width)) if len(width) else None,
                            "internal_bad_mean": float(arrays[f"{method}_internal_bad"][:, 0].mean())})
    return pd.DataFrame(records), arrays


def run_shard(stage, shard, shards, batch):
    directory = stage_dir(stage)
    directory.mkdir(parents=True, exist_ok=True)
    calibration, profiles, _, inputs = inputs_and_profiles()
    methods = stage_methods(stage)
    if stage == "B":
        inputs[str((stage_dir("A") / "selection.json").relative_to(ROOT))] = sha(stage_dir("A") / "selection.json")
    configs = scenarios()
    configs = [configs[0], configs[3]] if stage == "pilot" else configs
    for cfg in configs[shard::shards]:
        stem = directory / f"scenario_{cfg['id']:02d}"
        record_path = stem.with_suffix(".json")
        if record_path.exists():
            record = read_json(record_path)
            assert record["input_sha256"] == inputs, f"Inputs changed since {record_path.name}"
            if all(sha(ROOT / p) == h for p, h in record["output_sha256"].items()):
                continue
        started = time.perf_counter()
        records, arrays = simulate(cfg, calibration[cfg["metric"]], profiles[cfg["profile"], cfg["metric"]],
                                   stage, methods, batch)
        records.to_csv(stem.with_suffix(".csv"), index=False)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        record = dict(item="E3b", stage=stage, scenario=cfg, methods=methods, **STAGES[stage],
                      seconds=round(time.perf_counter()-started, 2), finished_utc=utc(),
                      input_sha256=inputs, python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__,
                      output_sha256={str(p.relative_to(ROOT)): sha(p)
                                     for p in [stem.with_suffix(".csv"), stem.with_suffix(".npz")]})
        write_json(record_path, record)
        print(json.dumps(dict(stage=stage, scenario=cfg["id"], seconds=record["seconds"])), flush=True)


def select(rows):
    summaries = []
    for method, data in rows.groupby("method", sort=False):
        cover = data[data.event.eq("coverage95")]
        boundary = data[np.isclose(data.delta.abs(), data.margin) & data.event.str.startswith("equivalence")]
        fwer = data[data.event.str.startswith("holm_fwer")]
        abstain = data[data.event.eq("abstain")]
        eligible = (cover.rate.min() >= 0.93 and boundary.rate.max() <= 0.05
                    and fwer.rate.max() <= 0.05 and abstain.rate.max() <= 0.01)
        summaries.append(dict(method=method, min_coverage=float(cover.rate.min()),
                              max_coverage=float(cover.rate.max()),
                              max_boundary_equivalence=float(boundary.rate.max()),
                              max_holm_fwer=float(fwer.rate.max()), max_abstention=float(abstain.rate.max()),
                              max_coverage_deviation=float((cover.rate-0.95).abs().max()),
                              mean_null_power_m4=float(data[data.delta.eq(0) & data.event.eq("equivalence_m4")].rate.mean()),
                              eligible=bool(eligible)))
    summary = pd.DataFrame(summaries)
    candidates = summary[summary.eligible]
    if not len(candidates):
        return summary, None
    candidates = candidates.assign(order=candidates.method.map({m: i for i, m in enumerate(METHODS)}))
    chosen = candidates.sort_values(["max_coverage_deviation", "mean_null_power_m4", "order"],
                                    ascending=[True, False, True]).iloc[0].method
    return summary, chosen


def finalise(stage):
    directory = stage_dir(stage)
    _, _, exposures, inputs = inputs_and_profiles()
    if stage == "B":
        inputs[str((stage_dir("A") / "selection.json").relative_to(ROOT))] = sha(stage_dir("A") / "selection.json")
    configs = scenarios()
    configs = [configs[0], configs[3]] if stage == "pilot" else configs
    outputs, frames, seconds = {}, [], 0.0
    for cfg in configs:
        record = read_json(directory / f"scenario_{cfg['id']:02d}.json")
        assert record["input_sha256"] == inputs, f"Inputs differ in scenario {cfg['id']}"
        assert record["scenario"] == cfg and record["methods"] == stage_methods(stage)
        for path, digest in record["output_sha256"].items():
            assert sha(ROOT / path) == digest, f"Output changed: {path}"
        outputs.update(record["output_sha256"])
        outputs[str((directory / f"scenario_{cfg['id']:02d}.json").relative_to(ROOT))] = sha(
            directory / f"scenario_{cfg['id']:02d}.json")
        frames.append(pd.read_csv(directory / f"scenario_{cfg['id']:02d}.csv"))
        seconds += record["seconds"]
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(directory / "summary.csv", index=False)
    exposures.to_csv(directory / "exposures.csv", index=False)
    table, chosen = select(rows)
    table.to_csv(directory / "method_selection.csv", index=False)
    if stage == "B":
        chosen = chosen if bool(table.eligible.iloc[0]) else None
        write_json(directory / "confirmation.json", dict(
            candidate=stage_methods("B")[0], confirmed=chosen is not None, scope="CMR_FA0_only",
            criteria="docs/protocolos/E3b.md"))
    else:
        write_json(directory / "selection.json", dict(selected=chosen, scope="CMR_FA0_only",
                                                      criteria="docs/protocolos/E3b.md"))
    for name in ["summary.csv", "exposures.csv", "method_selection.csv",
                 "confirmation.json" if stage == "B" else "selection.json"]:
        outputs[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    write_json(directory / "run_manifest.json", dict(
        item="E3b", stage=stage, finished_utc=utc(), **STAGES[stage], methods=stage_methods(stage),
        scenarios=len(configs), scenario_seconds=round(seconds, 2), input_sha256=inputs,
        protocol_sha256=sha(PROTOCOL), output_sha256=outputs, status="complete"))
    print(table.to_string(index=False), flush=True)
    print(json.dumps(dict(stage=stage, selected_or_confirmed=chosen)), flush=True)
    if stage != "pilot" and (stage == "B" or chosen is None):
        report()


def report():
    a = pd.read_csv(stage_dir("A") / "summary.csv")
    table_a = pd.read_csv(stage_dir("A") / "method_selection.csv")
    chosen = read_json(stage_dir("A") / "selection.json")["selected"]
    lines = ["# E3b — Wild bootstrap restrito e jackknife por SN para CMR e FA0", "",
             "Protocolo `docs/protocolos/E3b.md`; gerador, calibração e 40 cenários do E3 sem mudanças; "
             "5.000 réplicas e B = 1.999 pesos Rademacher por réplica.", "",
             "## Etapa A — seleção", "", markdown(table_a), ""]
    if chosen is None:
        decision = ("Nenhum dos quatro procedimentos cumpriu todos os critérios na etapa A. "
                    "Não há procedimento validado; E1/E4/E5/E7 mantêm a pendência inferencial.")
    else:
        b = read_json(stage_dir("B") / "confirmation.json")
        table_b = pd.read_csv(stage_dir("B") / "method_selection.csv")
        lines += ["## Etapa B — confirmação com semente independente", "", markdown(table_b), ""]
        decision = (f"Procedimento confirmado para CMR/FA0: {chosen}. Não valida DR nem macro-F1."
                    if b["confirmed"] else
                    f"{chosen} foi escolhido na etapa A, mas falhou na confirmação. Não há procedimento validado.")
    lines[2:2] = [f"**Decisão.** {decision}", ""]
    cover = a[a.event.eq("coverage95")]
    lines += ["## Cobertura por cenário (etapa A)", "", markdown(cover.pivot_table(
                  index=["profile", "metric", "rho", "delta"], columns="method", values="rate").reset_index()), "",
              "## FWER Holm sob o nulo global (etapa A)", "", markdown(a[a.event.str.startswith("holm")].pivot_table(
                  index=["profile", "metric", "rho", "event"], columns="method", values="rate").reset_index()), "",
              "## Falsa equivalência nas fronteiras (etapa A, máximo sobre m)", "", markdown(
                  a[np.isclose(a.delta.abs(), a.margin) & a.event.str.startswith("equivalence")].groupby(
                      ["profile", "metric", "rho", "delta", "method"]).rate.max().unstack().reset_index()), "",
              "## Limitações", "",
              "Mesmo gerador semissintético do E3: condiciona às exposições, sorteia probabilidades por SN "
              "de uma beta ajustada e não representa todo o treino em CV. Este é um segundo exame da mesma "
              "grade depois de um resultado negativo conhecido; a etapa B controla o acaso Monte Carlo da "
              "seleção, não a especificidade do gerador. Neste modelo só com intercepto, WCR-S e WCR-B "
              "coincidem com WCR-C e WCR-V. Amplitudes são descritivas (500 réplicas por cenário). "
              "DR e macro-F1 não foram avaliados.", ""]
    path = ROOT / "data/cluster_coverage_e3b_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {path.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--finalise", action="store_true")
    args = parser.parse_args()
    if args.finalise:
        finalise(args.stage)
    else:
        run_shard(args.stage, args.shard, args.shards, args.batch)


if __name__ == "__main__":
    main()
