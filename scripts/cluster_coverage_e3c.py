"""E3c: frozen coverage of unit-mean (L-bar) inference for CMR and WCR-C confirmation for FA0."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from cluster_coverage_simulation import (ROOT, generate, markdown, scenarios, setup, sha, utc,
                                         wilson, write_json)
from cluster_ratio_wild import abstentions, one_sided, rademacher, two_sided
from cluster_unit_mean import L_METHODS, lbar_abstentions, lbar_estimate, lbar_one_sided

OUT = ROOT / "data/cluster_coverage_e3c"
PROTOCOL = ROOT / "docs/protocolos/E3c.md"
FAMILIES = [1, 3, 4, 6, 8]
HOLM = [3, 4, 6, 8]
ORDER = list(L_METHODS)
STAGES = {"pilot": dict(seed=2026102700, replicates=32, resamples=199),
          "A": dict(seed=2026102701, replicates=5000, resamples=1999),
          "B": dict(seed=2026102702, replicates=5000, resamples=1999)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stage_dir(stage):
    return OUT / ("pilot" if stage == "pilot" else f"stage_{stage}")


def jobs(stage):
    """(scenario, endpoint, methods) triples for a stage."""
    result = []
    for cfg in scenarios():
        if stage == "pilot" and cfg["id"] not in (0, 3, 10, 13):
            continue
        if cfg["metric"] == "CMR":
            if stage == "B":
                chosen = read_json(stage_dir("A") / "selection.json")["selected"]
                assert chosen is not None, "Stage A selected no procedure; stage B is not run for Lbar"
                result.append((cfg, "Lbar", [chosen]))
            else:
                result.append((cfg, "Lbar", list(ORDER)))
        elif stage in ("B", "pilot"):
            result.append((cfg, "FA0", ["WCR-C"]))
    return result


def inputs_and_profiles():
    freeze = read_json(OUT / "protocol_freeze.json")
    assert freeze["protocol_sha256"] == sha(PROTOCOL), "E3c protocol changed after freezing"
    assert freeze["e3_simulation_sha256"] == sha(ROOT / "scripts/cluster_coverage_simulation.py")
    assert freeze["e3b_wild_sha256"] == sha(ROOT / "scripts/cluster_ratio_wild.py")
    calibration, profiles, exposures, inputs = setup()
    for path in [PROTOCOL, OUT / "protocol_freeze.json", Path(__file__),
                 ROOT / "scripts/cluster_ratio_wild.py", ROOT / "scripts/cluster_unit_mean.py",
                 ROOT / "tests/test_cluster_unit_mean.py"]:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return calibration, profiles, exposures, inputs


def simulate(cfg, endpoint, methods, calibration, profile, stage, batch):
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
    for first in range(0, len(flat), batch):
        rows = flat[first:first+batch]
        v = rademacher(boot_rng, len(rows), resamples, len(n))
        for method in methods:
            if endpoint == "Lbar":
                upper, lower, bad = lbar_one_sided(method, rows, n, hs, v)
                abstain = lbar_abstentions(method, rows, n)
            else:
                upper, lower, bad = one_sided("wcr_c", rows, n, hs, v)
                abstain = abstentions("wcr_c", rows, n)
            arrays[f"{method}_upper"].append(upper)
            arrays[f"{method}_lower"].append(lower)
            arrays[f"{method}_abstain"].append(abstain)
            arrays[f"{method}_internal_bad"].append(bad)
    arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
    arrays = {key: value.reshape(replicates, contrasts, *value.shape[1:]) for key, value in arrays.items()}
    estimate = lbar_estimate(flat, n) if endpoint == "Lbar" else flat.sum(axis=1)/n.sum()
    arrays["estimate"] = estimate.reshape(replicates, contrasts)
    arrays["h"] = hs
    records = []
    for method in methods:
        abstain = arrays[f"{method}_abstain"]
        upper = np.where(abstain[..., None], 1.0, arrays[f"{method}_upper"])
        lower = np.where(abstain[..., None], 1.0, arrays[f"{method}_lower"])
        p_two = two_sided(upper, lower)
        events = {"coverage95": p_two[:, 0, 0] > 0.05, "reject_zero": p_two[:, 0, 1] <= 0.05,
                  "abstain": abstain[:, 0]}
        for size in FAMILIES:
            events[f"equivalence_m{size}"] = (upper[:, 0, 2] <= 0.05/size) & (lower[:, 0, 3] <= 0.05/size)
        if cfg["delta"] == 0:
            for size in HOLM:
                events[f"holm_fwer_m{size}"] = p_two[:, :size, 1].min(axis=1) <= 0.05/size
        for event, flags in events.items():
            proportion, low, high, se = wilson(int(flags.sum()), replicates)
            records.append({**cfg, "endpoint": endpoint, "method": method, "event": event,
                            "count": int(flags.sum()), "replicates": replicates, "resamples": resamples,
                            "rate": proportion, "wilson_low": low, "wilson_high": high, "MC_SE": se,
                            "internal_bad_mean": float(arrays[f"{method}_internal_bad"][:, 0].mean())})
    return pd.DataFrame(records), arrays


def run_shard(stage, shard, shards, batch):
    directory = stage_dir(stage)
    directory.mkdir(parents=True, exist_ok=True)
    calibration, profiles, _, inputs = inputs_and_profiles()
    if stage == "B":
        inputs[str((stage_dir("A") / "selection.json").relative_to(ROOT))] = sha(stage_dir("A") / "selection.json")
    for cfg, endpoint, methods in jobs(stage)[shard::shards]:
        stem = directory / f"scenario_{cfg['id']:02d}"
        record_path = stem.with_suffix(".json")
        if record_path.exists():
            record = read_json(record_path)
            assert record["input_sha256"] == inputs, f"Inputs changed since {record_path.name}"
            if all(sha(ROOT / p) == h for p, h in record["output_sha256"].items()):
                continue
        started = time.perf_counter()
        records, arrays = simulate(cfg, endpoint, methods, calibration[cfg["metric"]],
                                   profiles[cfg["profile"], cfg["metric"]], stage, batch)
        records.to_csv(stem.with_suffix(".csv"), index=False)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        record = dict(item="E3c", stage=stage, scenario=cfg, endpoint=endpoint, methods=methods,
                      **STAGES[stage], seconds=round(time.perf_counter()-started, 2), finished_utc=utc(),
                      input_sha256=inputs, python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__,
                      output_sha256={str(p.relative_to(ROOT)): sha(p)
                                     for p in [stem.with_suffix(".csv"), stem.with_suffix(".npz")]})
        write_json(record_path, record)
        print(json.dumps(dict(stage=stage, scenario=cfg["id"], endpoint=endpoint, seconds=record["seconds"])),
              flush=True)


def evaluate(rows):
    """Eligibility per (endpoint, method) under the frozen criteria."""
    summaries = []
    for (endpoint, method), data in rows.groupby(["endpoint", "method"], sort=False):
        cover = data[data.event.eq("coverage95")]
        boundary = data[np.isclose(data.delta.abs(), data.margin) & data.event.str.startswith("equivalence")]
        fwer = data[data.event.str.startswith("holm_fwer")]
        abstain = data[data.event.eq("abstain")]
        eligible = (cover.rate.min() >= 0.93 and boundary.rate.max() <= 0.05
                    and fwer.rate.max() <= 0.05 and abstain.rate.max() <= 0.01)
        summaries.append(dict(endpoint=endpoint, method=method, min_coverage=float(cover.rate.min()),
                              max_coverage=float(cover.rate.max()),
                              max_boundary_equivalence=float(boundary.rate.max()),
                              max_holm_fwer=float(fwer.rate.max()), max_abstention=float(abstain.rate.max()),
                              max_coverage_deviation=float((cover.rate-0.95).abs().max()),
                              mean_null_power_m4=float(data[data.delta.eq(0) & data.event.eq("equivalence_m4")].rate.mean()),
                              eligible=bool(eligible)))
    return pd.DataFrame(summaries)


def choose(table):
    lbar = table[(table.endpoint == "Lbar") & table.eligible]
    if not len(lbar):
        return None
    lbar = lbar.assign(order=lbar.method.map({m: i for i, m in enumerate(ORDER)}))
    return lbar.sort_values(["max_coverage_deviation", "mean_null_power_m4", "order"],
                            ascending=[True, False, True]).iloc[0].method


def finalise(stage):
    directory = stage_dir(stage)
    _, _, exposures, inputs = inputs_and_profiles()
    if stage == "B":
        inputs[str((stage_dir("A") / "selection.json").relative_to(ROOT))] = sha(stage_dir("A") / "selection.json")
    outputs, frames, seconds, count = {}, [], 0.0, 0
    for cfg, endpoint, methods in jobs(stage):
        stem = directory / f"scenario_{cfg['id']:02d}"
        record = read_json(stem.with_suffix(".json"))
        assert record["input_sha256"] == inputs, f"Inputs differ in scenario {cfg['id']}"
        assert record["scenario"] == cfg and record["methods"] == methods and record["endpoint"] == endpoint
        for path, digest in record["output_sha256"].items():
            assert sha(ROOT / path) == digest, f"Output changed: {path}"
        outputs.update(record["output_sha256"])
        outputs[str(stem.with_suffix(".json").relative_to(ROOT))] = sha(stem.with_suffix(".json"))
        frames.append(pd.read_csv(stem.with_suffix(".csv")))
        seconds += record["seconds"]
        count += 1
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(directory / "summary.csv", index=False)
    exposures.to_csv(directory / "exposures.csv", index=False)
    table = evaluate(rows)
    table.to_csv(directory / "method_selection.csv", index=False)
    chosen = None
    if stage == "A":
        chosen = choose(table)
        write_json(directory / "selection.json", dict(selected=chosen, scope="Lbar_CMR_only",
                                                      criteria="docs/protocolos/E3c.md"))
        final_name = "selection.json"
    else:
        confirmed = {f"{r.endpoint}:{r.method}": bool(r.eligible) for r in table.itertuples()}
        write_json(directory / "confirmation.json", dict(confirmed=confirmed, scope="Lbar_and_FA0_cycle",
                                                         criteria="docs/protocolos/E3c.md"))
        final_name = "confirmation.json"
    for name in ["summary.csv", "exposures.csv", "method_selection.csv", final_name]:
        outputs[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    write_json(directory / "run_manifest.json", dict(
        item="E3c", stage=stage, finished_utc=utc(), **STAGES[stage], scenarios=count,
        scenario_seconds=round(seconds, 2), input_sha256=inputs, protocol_sha256=sha(PROTOCOL),
        output_sha256=outputs, status="complete"))
    print(table.to_string(index=False), flush=True)
    if stage == "A":
        print(json.dumps(dict(stage="A", selected=chosen)), flush=True)
    if stage == "B" or (stage == "A" and chosen is None):
        report()


def report():
    a = pd.read_csv(stage_dir("A") / "summary.csv")
    table_a = pd.read_csv(stage_dir("A") / "method_selection.csv")
    chosen = read_json(stage_dir("A") / "selection.json")["selected"]
    lines = ["# E3c — Perda média por unidade (L̄) e confirmação do WCR-C para FA0", "",
             "Protocolo `docs/protocolos/E3c.md`; gerador, calibração e cenários do E3 sem mudanças; "
             "5.000 réplicas e B = 1.999 pesos Rademacher por réplica.", "",
             "## Etapa A — seleção para L̄ (20 cenários de CMR)", "", markdown(table_a), ""]
    if chosen is None:
        decision = ("Nenhum dos quatro procedimentos para L̄ cumpriu todos os critérios na etapa A. "
                    "Não há procedimento validado para o endpoint crítico; a etapa B não foi executada "
                    "e, conforme o protocolo, não haverá nova tentativa (plano R3). A FA0 por ciclo "
                    "(WCR-C) não foi confirmada nesta rodada.")
    else:
        table_b = pd.read_csv(stage_dir("B") / "method_selection.csv")
        lines += ["## Etapa B — confirmação com semente independente", "", markdown(table_b), ""]
        parts = []
        for r in table_b.itertuples():
            name = "L̄" if r.endpoint == "Lbar" else "FA0 por ciclo"
            parts.append(f"{name} ({r.method}): {'confirmado' if r.eligible else 'NÃO confirmado'}")
        decision = "; ".join(parts) + ". Não valida a CMR por ciclo, DR nem macro-F1."
    lines[2:2] = [f"**Decisão.** {decision}", ""]
    cover = a[a.event.eq("coverage95")]
    lines += ["## Cobertura por cenário (etapa A, L̄)", "", markdown(cover.pivot_table(
                  index=["profile", "rho", "delta"], columns="method", values="rate").reset_index()), "",
              "## FWER Holm sob o nulo global (etapa A, L̄)", "", markdown(a[a.event.str.startswith("holm")].pivot_table(
                  index=["profile", "rho", "event"], columns="method", values="rate").reset_index()), "",
              "## Falsa equivalência nas fronteiras (etapa A, máximo sobre m)", "", markdown(
                  a[np.isclose(a.delta.abs(), a.margin) & a.event.str.startswith("equivalence")].groupby(
                      ["profile", "rho", "delta", "method"]).rate.max().unstack().reset_index()), "",
              "## Limitações", "",
              "Mesmo gerador semissintético do E3. Terceiro exame da mesma grade depois de dois resultados "
              "negativos conhecidos; a escolha do estimando L̄ e a restrição da FA0 ao WCR-C são "
              "pós-resultado. A etapa B controla o acaso Monte Carlo da seleção, não a especificidade do "
              "gerador. L̄ dá peso igual a cada SN crítica (inclusive a de um único ciclo) e responde a "
              "uma pergunta diferente da CMR por ciclo.", ""]
    path = ROOT / "data/cluster_coverage_e3c_summary.md"
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
