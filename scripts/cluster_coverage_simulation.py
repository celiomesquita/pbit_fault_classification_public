"""E3: frozen semisynthetic coverage experiment for CMR/FA0 ratio inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
from scipy.stats import beta

from cluster_ratio_methods import FAMILIES, METHODS, infer_batch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cluster_coverage"
PROTOCOL = ROOT / "docs/protocolos/E3.md"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
MARGINS = {"CMR": 0.02, "FA0": 0.03}
META = ["Experiment_Index", "Serial_Number", "Fault_Mode"]
BASE_SEED = 2026101103


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def wilson(count, size):
    p, z = count/size, 1.959963984540054
    centre = (p+z*z/(2*size))/(1+z*z/size)
    radius = z*np.sqrt(p*(1-p)/size+z*z/(4*size*size))/(1+z*z/size)
    return p, max(0, centre-radius), min(1, centre+radius), np.sqrt(p*(1-p)/size)


def shifted_probability(p, mu, delta):
    assert 0 <= mu+delta <= 1
    return p+delta*(1-p)/(1-mu) if delta >= 0 else p*(1+delta/mu)


def setup():
    inputs = [PROTOCOL, Path(__file__), ROOT / "scripts/cluster_ratio_methods.py",
              ROOT / "src/pbit/inference.py", ROOT / "references/t0344.pdf",
              ROOT / "data/train_real.csv", ROOT / "data/test_real.csv"]
    train, test = [pd.read_csv(p, usecols=META) for p in inputs[-2:]]
    assert set(train.Serial_Number).isdisjoint(test.Serial_Number)
    frames = []
    for seed in SEEDS:
        path = ROOT / "data/ablation" / f"preds_seed{seed}.csv"
        inputs.append(path)
        df = pd.read_csv(path)
        assert df.seed.eq(seed).all() and df.Experiment_Index.is_unique
        pd.testing.assert_frame_equal(df[META].sort_values("Experiment_Index").reset_index(drop=True),
                                      train.sort_values("Experiment_Index").reset_index(drop=True))
        assert df.groupby("Serial_Number").fold.nunique().eq(1).all()
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    observed = {}
    for metric in MARGINS:
        mask = full.Fault_Mode.isin(["F03", "F06"] if metric == "CMR" else ["N01", "N02"])
        loss = full.lgbm_F246.isin(["N01", "N02"])
        if metric == "FA0":
            loss = ~loss
        unit_loss = full.assign(loss=loss.astype(int))[mask].groupby("Serial_Number").loss.mean()
        mu, variance = float(unit_loss.mean()), float(unit_loss.var(ddof=1))
        assert 0 < variance < mu*(1-mu)
        concentration = mu*(1-mu)/variance-1
        observed[metric] = dict(mu=mu, variance=variance, a=mu*concentration,
                                b=(1-mu)*concentration, units=len(unit_loss))
    historical = frames[0][META+['fold']].copy()
    audited = pd.concat([train.assign(fold="training"), test.assign(fold="heldout")], ignore_index=True)
    audited = audited[audited.Experiment_Index.ne(1107)]
    assert len(historical) == 1196 and historical.Serial_Number.nunique() == 54
    assert len(audited) == 1487 and audited.Serial_Number.nunique() == 65  # verified below if source differs
    profiles, profile_rows = {}, []
    for name, frame in [("historical", historical), ("audited", audited)]:
        for metric in MARGINS:
            sns = sorted(frame.Serial_Number.unique())
            endpoint = frame.Fault_Mode.isin(["F03", "F06"] if metric == "CMR" else ["N01", "N02"])
            exposure = frame[endpoint].groupby("Serial_Number").size().reindex(sns, fill_value=0)
            profiles[name, metric] = dict(units=len(sns), n=exposure[exposure > 0].to_numpy(dtype=int),
                                         sns=exposure[exposure > 0].index.tolist())
            for sn, count in exposure.items():
                fold = frame.loc[frame.Serial_Number.eq(sn), "fold"].iloc[0]
                profile_rows.append(dict(profile=name, metric=metric, Serial_Number=sn, cycles=count, fold=fold))
    return observed, profiles, pd.DataFrame(profile_rows), {str(p.relative_to(ROOT)): sha(p) for p in inputs}


def generate(rng, replicates, contrasts, n, calibration, delta, rho):
    """Shared reference B; exact Bernoulli marginal counts over ten paired seeds."""
    g = len(n)
    zb = rng.normal(size=(replicates, g))
    za = rho*zb[:, None, :] + np.sqrt(1-rho*rho)*rng.normal(size=(replicates, contrasts, g))
    pb = beta.ppf(ndtr(zb), calibration["a"], calibration["b"])
    pa0 = beta.ppf(ndtr(za), calibration["a"], calibration["b"])
    pa = shifted_probability(pa0, calibration["mu"], delta)
    assert np.isfinite(pa).all() and np.all((pa >= 0) & (pa <= 1))
    joint11 = 0.8*np.minimum(pa, pb[:, None, :]) + 0.2*pa*pb[:, None, :]
    q1 = np.divide(joint11, pb[:, None, :], out=np.zeros_like(pa), where=pb[:, None, :] > 0)
    q0 = np.divide(pa-joint11, 1-pb[:, None, :], out=np.zeros_like(pa), where=pb[:, None, :] < 1)
    common_n = rng.binomial(n, 0.5, size=(replicates, g))
    independent_n = (n-common_n)*len(SEEDS)
    b_common = rng.binomial(common_n, pb)
    b_independent = rng.binomial(independent_n, pb)
    a_common = (rng.binomial(b_common[:, None, :], np.clip(q1, 0, 1))
                + rng.binomial((common_n-b_common)[:, None, :], np.clip(q0, 0, 1)))
    a_independent = (rng.binomial(b_independent[:, None, :], np.clip(q1, 0, 1))
                     + rng.binomial((independent_n-b_independent)[:, None, :], np.clip(q0, 0, 1)))
    return (a_common-b_common[:, None, :]) + (a_independent-b_independent[:, None, :])/len(SEEDS)


def scenarios():
    result = []
    for profile in ["historical", "audited"]:
        for metric, margin in MARGINS.items():
            for rho, effects in [(0.8, [0, -0.01, 0.01, -margin, margin, -2*margin, 2*margin]),
                                 (0.2, [0, -margin, margin])]:
                for delta in effects:
                    result.append(dict(id=len(result), profile=profile, metric=metric, margin=margin,
                                       rho=rho, delta=delta, contrasts=8 if delta == 0 else 1))
    return result


def simulate(cfg, calibration, profile, replicates, resamples, batch):
    rng = np.random.default_rng(BASE_SEED+cfg["id"])
    contrast_count = cfg["contrasts"]
    counts = generate(rng, replicates, contrast_count, profile["n"], calibration, cfg["delta"], cfg["rho"])
    flat = counts.reshape(-1, len(profile["n"]))
    estimates = np.empty(len(flat))
    collected = {m: dict(bounds=[], p=[], abstain=[], internal_bad=[]) for m in METHODS}
    for first in range(0, len(flat), batch):
        last = min(first+batch, len(flat))
        est, results = infer_batch(flat[first:last], profile["n"], profile["units"], rng, resamples)
        estimates[first:last] = est
        for method, values in results.items():
            for key, value in values.items():
                collected[method][key].append(value)
    records, arrays = [], {"estimate": estimates.reshape(replicates, contrast_count)}
    for method in METHODS:
        values = {k: np.concatenate(v) for k, v in collected[method].items()}
        for key, value in values.items():
            arrays[f"{method}_{key}"] = value.reshape(replicates, contrast_count, *value.shape[1:])
        # Coverage and equivalence use the first contrast only, not 8R nominal replicates.
        bounds = arrays[f"{method}_bounds"][:, 0]
        p = arrays[f"{method}_p"][:, 0]
        events = {"coverage95": (bounds[:, 0] <= cfg["delta"]) & (cfg["delta"] <= bounds[:, 1]),
                  "reject_zero": p <= 0.05, "abstain": arrays[f"{method}_abstain"][:, 0]}
        for i, size in enumerate(FAMILIES):
            events[f"equivalence_m{size}"] = ((bounds[:, 2+2*i] > -cfg["margin"])
                                               & (bounds[:, 3+2*i] < cfg["margin"]))
        if cfg["delta"] == 0:
            all_p = arrays[f"{method}_p"]
            for size in [3, 4, 8]:
                # Under the global null Holm rejects anything iff min(p) <= alpha/m.
                events[f"holm_fwer_m{size}"] = all_p[:, :size].min(axis=1) <= 0.05/size
        width = bounds[:, 1]-bounds[:, 0]
        median_width = float(np.median(width[np.isfinite(width)])) if np.isfinite(width).any() else None
        for event, flags in events.items():
            proportion, low, high, se = wilson(int(flags.sum()), replicates)
            records.append({**cfg, "method": method, "event": event, "count": int(flags.sum()),
                            "replicates": replicates, "resamples": resamples, "rate": proportion,
                            "wilson_low": low, "wilson_high": high, "MC_SE": se,
                            "median_finite_width": median_width,
                            "internal_bad_mean": float(arrays[f"{method}_internal_bad"][:, 0].mean())})
    return pd.DataFrame(records), arrays


def select(rows):
    summaries = []
    for method, data in rows.groupby("method", sort=False):
        cover = data[data.event.eq("coverage95")]
        boundary = data[np.isclose(data.delta.abs(), data.margin) & data.event.str.startswith("equivalence")]
        fwer = data[data.event.str.startswith("holm_fwer")]
        abstain = data[data.event.eq("abstain")]
        eligible = (cover.rate.min() >= 0.93 and boundary.rate.max() <= 0.05
                    and fwer.rate.max() <= 0.05 and abstain.rate.max() <= 0.01 and method != "sign_swap")
        summaries.append(dict(method=method, min_coverage=float(cover.rate.min()),
                              max_boundary_equivalence=float(boundary.rate.max()),
                              max_holm_fwer=float(fwer.rate.max()), max_abstention=float(abstain.rate.max()),
                              max_coverage_deviation=float((cover.rate-0.95).abs().max()),
                              mean_null_power_m4=float(data[data.delta.eq(0) & data.event.eq("equivalence_m4")].rate.mean()),
                              eligible=bool(eligible)))
    summary = pd.DataFrame(summaries)
    candidates = summary[summary.eligible]
    if "percentile" in set(candidates.method):
        chosen = "percentile"
    elif len(candidates):
        candidates = candidates.assign(order=candidates.method.map({m: i for i, m in enumerate(METHODS)}))
        chosen = candidates.sort_values(["max_coverage_deviation", "mean_null_power_m4", "order"],
                                        ascending=[True, False, True]).iloc[0].method
    else:
        chosen = None
    return summary, chosen


def markdown(df):
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"]*len(df.columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def summarise(run):
    rows = pd.read_csv(run / "summary.csv")
    table, chosen = select(rows)
    table.to_csv(run / "method_selection.csv", index=False)
    write_json(run / "selection.json", dict(selected=chosen, scope="CMR_FA0_only", criteria="docs/protocolos/E3.md"))
    if chosen is None:
        decision = "Nenhum procedimento cumpriu todos os critérios fixados. Não há procedimento validado para substituir a inferência histórica."
    else:
        decision = f"Procedimento seleccionado para CMR/FA0: {chosen}. Esta selecção não valida DR ou macro-F1."
    report = ["# E3 — Cobertura dos procedimentos de inferência por unidade", "", decision, "",
              "40 cenários; 5.000 réplicas e 2.000 sorteios internos por procedimento/réplica. "
              "CMR e FA0; populações histórica e auditada; cópula latente 0,8 e sensibilidade 0,2. "
              "As famílias Holm têm referência comum e 3/4/8 contrastes de taxas homogéneas. "
              "Não são as famílias mistas exactas de DR/macro-F1 do manuscrito.", "",
              "## Decisão global", "", markdown(table), "",
              "## Cobertura por cenário", "", markdown(rows[rows.event.eq("coverage95")][
                  ["profile", "metric", "rho", "delta", "method", "rate", "wilson_low", "wilson_high", "MC_SE", "median_finite_width", "internal_bad_mean"]]), "",
              "## Fronteiras de equivalência (erro tipo I)", "", markdown(rows[
                  np.isclose(rows.delta.abs(), rows.margin) & rows.event.str.startswith("equivalence")][
                  ["profile", "metric", "rho", "delta", "method", "event", "rate", "wilson_low", "wilson_high"]]), "",
              "## Holm sob o nulo global", "", markdown(rows[rows.event.str.startswith("holm_fwer")][
                  ["profile", "metric", "rho", "method", "event", "rate", "wilson_low", "wilson_high"]]), "",
              "## Limitações", "", "O gerador condiciona às exposições e sortea probabilidades por SN "
              "de uma beta ajustada a perdas observadas, não corrige a estimação dessa beta nem "
              "representa todo o processo de treino em CV. Preserva folds como metadados; a inferência "
              "é sobre predições externas agrupadas. Seeds são dez repetições pareadas dependentes, não clusters. "
              "Troca de sinal requer simetria adicional. Wild-t é uma adaptação não restrita para razões, "
              "não o algoritmo de regressão restrito ao nulo de Cameron et al. "
              "Nenhuma calibração observada constitui garantia fora dos cenários. "
              "DR e macro-F1 não foram validados. Wilson mede a incerteza Monte Carlo, não a do gerador.", "",
              "## Rastreabilidade", "", "Protocolo `docs/protocolos/E3.md`; calibração e exposições em "
              "`data/cluster_coverage/run/`; valores por réplica nos NPZ por cenário. "
              "Manifesto inclui hashes de entradas, código, protocolo e saídas. Piloto separado em `pilot/`.", ""]
    report_path = ROOT / "data/cluster_coverage_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    labels = {"percentile": "Percentile", "pairs_t": "Pairs bootstrap-t", "wild_rademacher": "Wild-t Rademacher",
              "wild_webb": "Wild-t Webb", "sign_swap": "Sign swap", "bca": "BCa"}
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Procedure & Min. coverage & Max. boundary Eq. & Max. Holm FWER & Max. abstention \\", r"\midrule"]
    for row in table.itertuples(index=False):
        tex.append(f"{labels[row.method]} & {row.min_coverage:.3f} & {row.max_boundary_equivalence:.3f} & "
                   f"{row.max_holm_fwer:.3f} & {row.max_abstention:.3f}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    table_path = ROOT / "paper/supplement/tables/tab_cluster_coverage.tex"
    table_path.write_text("\n".join(tex)+"\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(decision, flush=True)
    return [run / "method_selection.csv", run / "selection.json", report_path, table_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "run"], default="run")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    run = OUT / args.mode
    run.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((OUT / "protocol_freeze.json").read_text(encoding="utf-8-sig"))
    assert freeze["protocol_sha256"] == sha(PROTOCOL)
    calibration, profiles, exposure_table, inputs = setup()
    inputs["data/cluster_coverage/protocol_freeze.json"] = sha(OUT / "protocol_freeze.json")
    reps, resamples = (32, 200) if args.mode == "pilot" else (5000, 2000)
    configs = scenarios() if args.mode == "run" else [scenarios()[0], scenarios()[3]]
    manifest_path = run / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["input_sha256"] == inputs, "Inputs/code changed since run started"
        assert manifest["replicates"] == reps and manifest["resamples"] == resamples
        for path, digest in manifest.get("output_sha256", {}).items():
            assert sha(ROOT / path) == digest, f"Output changed: {path}"
    else:
        manifest = dict(item="E3", mode=args.mode, started_utc=utc(), input_sha256=inputs,
                        protocol_sha256=sha(PROTOCOL), replicates=reps, resamples=resamples,
                        base_seed=BASE_SEED, scenarios=configs, completed=[], output_sha256={},
                        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                        scipy=scipy.__version__, deviations=[], status="running")
        write_json(run / "calibration.json", calibration)
        exposure_table.to_csv(run / "exposures.csv", index=False)
        write_json(manifest_path, manifest)
    for cfg in configs:
        if cfg["id"] in manifest["completed"]:
            continue
        started = time.perf_counter()
        records, arrays = simulate(cfg, calibration[cfg["metric"]], profiles[cfg["profile"], cfg["metric"]],
                                   reps, resamples, args.batch)
        csv = run / f"scenario_{cfg['id']:02d}.csv"
        npz = run / f"scenario_{cfg['id']:02d}.npz"
        records.to_csv(csv, index=False)
        np.savez_compressed(npz, **arrays)
        manifest["completed"].append(cfg["id"])
        manifest["output_sha256"].update({str(p.relative_to(ROOT)): sha(p) for p in [csv, npz]})
        write_json(manifest_path, manifest)
        print(json.dumps(dict(scenario=cfg, seconds=round(time.perf_counter()-started, 2),
                              completed=len(manifest["completed"]), total=len(configs))), flush=True)
    combined = pd.concat([pd.read_csv(run / f"scenario_{cfg['id']:02d}.csv") for cfg in configs], ignore_index=True)
    combined.to_csv(run / "summary.csv", index=False)
    outputs = [run / "summary.csv", run / "calibration.json", run / "exposures.csv"]
    if args.mode == "run":
        outputs.extend(summarise(run))
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, name
    manifest["output_sha256"].update({str(p.relative_to(ROOT)): sha(p) for p in outputs})
    manifest.update(status="complete", finished_utc=utc())
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
