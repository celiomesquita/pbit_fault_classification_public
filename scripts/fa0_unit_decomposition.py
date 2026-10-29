"""E2: descriptive unit decomposition of saved normal-cycle predictions."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "fa0_unit_decomposition"
PROTOCOL = ROOT / "docs/protocolos/E2.md"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
META = ["Experiment_Index", "Serial_Number", "Fault_Mode"]
CLASSES = ["N01", "N02", "F01", "F02", "F03", "F04", "F05", "F06"]
B = 100_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(df):
    def fmt(v):
        return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join([
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
        *["| " + " | ".join(map(fmt, row)) + " |" for row in df.itertuples(index=False, name=None)],
    ])


def validate(df, reference):
    assert not df[META].isna().any().any()
    assert not df.Experiment_Index.duplicated().any()
    got = df[META].sort_values("Experiment_Index").reset_index(drop=True)
    want = reference[META].sort_values("Experiment_Index").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want)


def decompose(df, columns, source):
    normal = df.loc[df.Fault_Mode.isin(["N01", "N02"])]
    tables, destinations = [], []
    for col in columns:
        assert df[col].isin(CLASSES).all(), (source, col)
        d = normal[["seed", "Serial_Number", "Fault_Mode", col]].copy()
        d["alarm"] = d[col].isin(CLASSES[2:]).astype(int)
        unit_seed = d.groupby(["seed", "Serial_Number", "Fault_Mode"]).agg(
            cycles=("alarm", "size"), alarms=("alarm", "sum"))
        assert unit_seed.groupby(["Serial_Number", "Fault_Mode"]).cycles.nunique().eq(1).all()
        units = unit_seed.groupby(["Serial_Number", "Fault_Mode"]).agg(
            cycles=("cycles", "first"), alarms=("alarms", "mean"), seeds=("alarms", "size")).reset_index()
        assert units.seeds.eq(df.seed.nunique()).all()
        units["FA0"] = units.alarms / units.cycles
        units.insert(0, "arm", col)
        units.insert(0, "source", source)
        tables.append(units)
        counts = d.groupby(["Fault_Mode", col]).size().reset_index(name="total")
        counts = counts.rename(columns={col: "predicted_class"})
        counts["mean_count"] = counts.total / df.seed.nunique()
        counts.insert(0, "arm", col)
        counts.insert(0, "source", source)
        assert np.isclose(counts.mean_count.sum(), units.cycles.sum())
        assert np.isclose(counts.loc[counts.predicted_class.isin(CLASSES[2:]), "mean_count"].sum(), units.alarms.sum())
        destinations.append(counts)
    return pd.concat(tables, ignore_index=True), pd.concat(destinations, ignore_index=True)


def summarise(units):
    rows = []
    for (source, arm), g in units.groupby(["source", "arm"], sort=False):
        by_sn = g.groupby("Serial_Number").agg(cycles=("cycles", "sum"), alarms=("alarms", "sum")).reset_index()
        by_sn["FA0"] = by_sn.alarms / by_sn.cycles
        n01, n02 = g[g.Fault_Mode == "N01"], g[g.Fault_Mode == "N02"]
        sn045 = g[g.Serial_Number == "SN 045"]
        other = g[g.Serial_Number != "SN 045"]
        top = by_sn.sort_values(["alarms", "Serial_Number"], ascending=[False, True]).head(3)
        rows.append(dict(
            source=source, arm=arm, units=len(by_sn), cycles=int(g.cycles.sum()),
            mean_alarms=g.alarms.sum(), FA0_cycle=g.alarms.sum() / g.cycles.sum(),
            FA0_unit_mean=by_sn.FA0.mean(), FA0_unit_median=by_sn.FA0.median(),
            FA0_unit_min=by_sn.FA0.min(), FA0_unit_max=by_sn.FA0.max(),
            FA0_N01=n01.alarms.sum() / n01.cycles.sum(),
            FA0_N02=n02.alarms.sum() / n02.cycles.sum() if len(n02) else np.nan,
            SN045_alarm_share=sn045.alarms.sum() / g.alarms.sum() if g.alarms.sum() else np.nan,
            FA0_without_SN045=other.alarms.sum() / other.cycles.sum(),
            top3_alarm_share=top.alarms.sum() / g.alarms.sum() if g.alarms.sum() else np.nan,
            top3_cycle_share=top.cycles.sum() / g.cycles.sum(),
            top3_units=", ".join(top.Serial_Number),
        ))
    return pd.DataFrame(rows)


def resample(units):
    cv = units[(units.source != "test") & (units.Fault_Mode == "N01")]
    sns = sorted(cv.Serial_Number.unique())
    rng = np.random.default_rng(20261011)
    indices = {
        "with_replacement": rng.integers(len(sns), size=(B, 7)),
        "without_replacement": np.argsort(rng.random((B, len(sns))), axis=1)[:, :7],
    }
    rows = []
    for (source, arm), g in cv.groupby(["source", "arm"], sort=False):
        g = g.set_index("Serial_Number").loc[sns]
        for sampling, ix in indices.items():
            values = {
                "unit_mean": g.FA0.to_numpy()[ix].mean(axis=1),
                "cycle_weighted": g.alarms.to_numpy()[ix].sum(axis=1) / g.cycles.to_numpy()[ix].sum(axis=1),
            }
            for weighting, rates in values.items():
                count = int(np.count_nonzero(rates <= 0.10))
                p = count / B
                lo, median, hi = np.quantile(rates, [0.025, 0.5, 0.975])
                rows.append(dict(source=source, arm=arm, sampling=sampling, weighting=weighting,
                                 n_pool=len(sns), draws=B, count_le_010=count, probability=p,
                                 MC_SE=np.sqrt(p * (1 - p) / B), q025=lo, q50=median, q975=hi))
    return pd.DataFrame(rows)


def main():
    started = datetime.now(timezone.utc).isoformat()
    freeze_path = OUT / "protocol_freeze.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    assert sha(PROTOCOL) == frozen["protocol_sha256"], "Protocol changed since freeze"
    inputs = [PROTOCOL, freeze_path, Path(__file__), DATA / "train_real.csv", DATA / "test_real.csv"]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    train = pd.read_csv(DATA / "train_real.csv", usecols=META)
    test = pd.read_csv(DATA / "test_real.csv", usecols=META)
    assert set(train.Serial_Number).isdisjoint(test.Serial_Number)
    assert set(train.Experiment_Index).isdisjoint(test.Experiment_Index)
    units, destinations = [], []
    for source in ["ablation", "families", "imbalance"]:
        frames = []
        for seed in SEEDS:
            path = DATA / source / f"preds_seed{seed}.csv"
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
            df = pd.read_csv(path)
            assert df.seed.eq(seed).all()
            validate(df, train)
            if frames:
                assert list(df.columns) == list(frames[0].columns)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        columns = [c for c in df.columns if c not in [*META, "seed", "fold"]]
        u, d = decompose(df, columns, source)
        units.append(u)
        destinations.append(d)
    path = DATA / "final_test_predictions.csv"
    input_hashes[str(path.relative_to(ROOT))] = sha(path)
    df = pd.read_csv(path)
    validate(df, test)
    df["seed"] = 42
    u, d = decompose(df, [c for c in df if c.startswith("pred_")], "test")
    assert u[u.Fault_Mode == "N01"].groupby("arm").size().eq(7).all()
    units.append(u)
    destinations.append(d)
    units = pd.concat(units, ignore_index=True)
    destinations = pd.concat(destinations, ignore_index=True)
    summary = summarise(units)
    sampling = resample(units)
    outputs = []
    for name, table in [("units", units), ("summary", summary), ("destinations", destinations), ("resampling", sampling)]:
        path = OUT / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs.append(path)
    main_arms = ["rf_F246", "lgbm_F246", "xgb_F246", "gb_F246"]
    primary = summary[summary.arm.isin(main_arms) | summary.source.eq("test")]
    sampled_primary = sampling[sampling.arm.isin(main_arms)]
    lgbm = summary[summary.arm.eq("lgbm_F246")].iloc[0]
    test_f1 = summary[summary.arm.eq("pred_main_F1-tuned")].iloc[0]
    table = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Units & Cycle FA0 & Unit mean & Unit median & Without SN 045 \\", r"\midrule"]
    labels = {"lgbm_F246": "LightGBM (CV)", "rf_F246": "Random Forest (CV)",
              "xgb_F246": "XGBoost (CV)", "gb_F246": "Gradient Boosting (CV)",
              "pred_main_F1-tuned": "F1-tuned (test)", "pred_main_CMR-tuned": "CMR-tuned (test)"}
    for row in primary.itertuples(index=False):
        if row.arm in labels:
            table.append(f"{labels[row.arm]} & {row.units} & {row.FA0_cycle:.3f} & "
                         f"{row.FA0_unit_mean:.3f} & {row.FA0_unit_median:.3f} & "
                         + (f"{row.FA0_without_SN045:.3f}" if row.source != "test" else "--") + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    table_path = ROOT / "paper/supplement/tables/tab_fa0_unit_decomposition.tex"
    table_path.write_text("\n".join(table) + "\n", encoding="utf-8")
    outputs.append(table_path)
    report = [
        "# E2 — Decomposição do FA0 por unidade", "",
        f"Protocolo: `docs/protocolos/E2.md`; SHA-256 `{sha(PROTOCOL)}`. Nenhum novo ajuste.", "",
        f"População histórica: {len(train)} ciclos de treino, {train.Serial_Number.nunique()} SN; "
        f"{len(test)} ciclos de teste, {test.Serial_Number.nunique()} SN. "
        f"Ciclo 1107 no treino: {1107 in set(train.Experiment_Index)}. Dez sementes na CV; uma no teste.", "",
        "## Interpretação", "",
        f"No LightGBM F246, FA0 passa de {lgbm.FA0_cycle:.6f} para {lgbm.FA0_without_SN045:.6f} "
        f"sem SN 045; essa unidade representa {100*lgbm.SN045_alarm_share:.2f}% dos alarmes. "
        f"As três unidades com mais alarmes concentram {100*lgbm.top3_alarm_share:.2f}% dos alarmes "
        f"em {100*lgbm.top3_cycle_share:.2f}% dos ciclos normais. A concentração existe, mas "
        "a retirada da SN 045 não aproxima a CV do FA0 por ciclo do teste.", "",
        f"No teste F1-tuned, FA0 por ciclo = {test_f1.FA0_cycle:.6f}, média por unidade = "
        f"{test_f1.FA0_unit_mean:.6f} e mediana = {test_f1.FA0_unit_median:.6f}. "
        "SN 031 tem 3 alarmes em 4 ciclos e SN 065 tem 2 em 2. "
        "Logo, o baixo FA0 por ciclo não implica sete unidades uniformemente fáceis. "
        "A versão ponderada e a não ponderada devem ser reportadas juntas; "
        "o mecanismo da diferença entre modelos/populações permanece confundido.", "",
        "## Braços principais e teste", "", markdown(primary), "",
        "FA0_cycle pondera ciclos; FA0_unit_mean dá peso igual às unidades. "
        "Alarmes e classes preditas são médias sobre sementes. Top-3 ordenado por número de alarmes, não por taxa.", "",
        "## Sete unidades N01 — reamostragem descritiva", "", markdown(sampled_primary), "",
        "Probabilidade de FA0 <= 0,10 na população empírica de treino. "
        "MC_SE mede apenas erro Monte Carlo; não é incerteza da população. "
        "Quantis descrevem sorteios de sete unidades, não IC de uma diferença. "
        "Com reposição é a análise principal; sem reposição, sensibilidade.", "",
        "## Taxas por unidade nos braços principais e no teste", "",
        markdown(units[units.arm.isin(main_arms) | units.source.eq("test")]), "",
        "## Classes preditas nos alarmes dos braços principais e do teste", "",
        markdown(destinations[(destinations.arm.isin(main_arms) | destinations.source.eq("test")) &
                              destinations.predicted_class.isin(CLASSES[2:])]), "",
        "## Limitações", "",
        "N02 está restrita às SN 030 e 045; SN 030 também contém N01. A CV avalia modelos fixos e o teste avalia "
        "modelos ajustados: o contraste não identifica isoladamente facilidade das unidades de teste. "
        "As distribuições empíricas não constituem validação externa nem inferência confirmatória. "
        "Arquivos completos incluem todos os braços e controlos. E3 permanece necessário para inferência.", "",
        "## Verificações", "",
        "Identidade de ciclos/metadados com train_real/test_real; dez sementes completas; "
        "ausência de SN e ciclos compartilhados entre treino e teste; classes conhecidas; "
        "exposição constante por unidade entre sementes; conservação de ciclos e alarmes "
        "nas decomposições; sete unidades N01 no teste. Todas satisfeitas.", "",
    ]
    report_path = DATA / "fa0_unit_decomposition.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    outputs.append(report_path)
    for name, digest in input_hashes.items():
        assert sha(ROOT / name) == digest, f"Input changed during execution: {name}"
    manifest = dict(item="E2", started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                    protocol_sha256=sha(PROTOCOL), freeze=frozen, input_sha256=input_hashes,
                    output_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    resampling_draws=B, random_seed=20261011,
                    deviations=["Emenda anterior aos resultados: N02 em SN 030 e 045; agregar N01/N02 por SN nas estatísticas globais. V1 e seu congelamento preservados."])
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(primary[["source", "arm", "FA0_cycle", "FA0_N01", "FA0_N02", "FA0_without_SN045", "SN045_alarm_share"]].to_string(index=False))
    print(sampled_primary[sampled_primary.sampling.eq("with_replacement")].to_string(index=False))


if __name__ == "__main__":
    main()
