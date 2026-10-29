"""A1: analytic independent-unit requirements, with separate evaluation/CRC roles."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/unit_evidence_requirements'
PROTOCOL = ROOT / 'docs/protocolos/A1.md'
TARGETS = ['0.30', '0.20', '0.10', '0.05', '0.02', '0.01']
COUNTS = [2, 4, 7, 9, 11, 12, 14, 16]
META = ['Experiment_Index', 'Serial_Number', 'Fault_Mode']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_event_upper(n):
    return -math.expm1(math.log(0.025)/n)


def crc_minimum(alpha):
    with localcontext() as context:
        context.prec = 50
        a = Decimal(alpha)
        n = int((1/a - 1).to_integral_value(rounding=ROUND_CEILING))
        assert Decimal(1)/Decimal(n+1) <= a
        assert n == 0 or Decimal(1)/Decimal(n) > a
    return n


def markdown(df):
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| ' + ' | '.join(df.columns) + ' |',
                      '| ' + ' | '.join(['---']*len(df.columns)) + ' |',
                      *['| ' + ' | '.join(map(fmt, row)) + ' |' for row in df.itertuples(index=False, name=None)]])


def main():
    started = datetime.now(timezone.utc).isoformat()
    freeze_path = OUT / 'protocol_freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8-sig'))
    assert freeze['protocol_sha256'] == sha(PROTOCOL)
    inputs = [PROTOCOL, freeze_path, Path(__file__), ROOT / 'data/train_real.csv',
              ROOT / 'data/test_real.csv', ROOT / 'data/restart/inference/comparisons.csv',
              ROOT / 'data/cluster_coverage/run/selection.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    train, test = [pd.read_csv(p, usecols=META) for p in inputs[3:5]]
    assert train.Experiment_Index.is_unique and test.Experiment_Index.is_unique
    assert set(train.Experiment_Index).isdisjoint(test.Experiment_Index)
    assert set(train.Serial_Number).isdisjoint(test.Serial_Number)
    population = pd.concat([train, test], ignore_index=True)
    population = population[population.Experiment_Index.ne(1107)]
    assert len(population) == 1487
    critical = population[population.Fault_Mode.isin(['F03', 'F06'])]
    units = critical.Serial_Number.nunique()
    class_units = critical.groupby('Fault_Mode').Serial_Number.nunique().to_dict()
    assert units == 16 and class_units == {'F03': 12, 'F06': 4}
    selection = json.loads(inputs[-1].read_text(encoding='utf-8'))
    assert selection['selected'] is None, 'Update A1 protocol if a calibrated method is subsequently selected'
    rows = []
    for target in TARGETS:
        p = float(target)
        n = math.ceil(math.log(0.025)/math.log1p(-p))
        assert zero_event_upper(n) <= p and zero_event_upper(n-1) > p
        np.testing.assert_allclose(zero_event_upper(n), beta.ppf(0.975, 1, n), rtol=1e-12)
        crc_n = crc_minimum(target)
        rows.append(dict(target=p, n_evaluation_zero_events=n, CP_upper_at_minimum=zero_event_upper(n),
                         deficit_evaluation_vs_16=max(0, n-units), n_calibration_CRC=crc_n,
                         CRC_floor_at_minimum=1/(crc_n+1), deficit_calibration_vs_16=max(0, crc_n-units)))
    requirements = pd.DataFrame(rows)
    availability = pd.DataFrame([dict(units=n, CP_upper_if_zero_events=zero_event_upper(n),
                                      CRC_floor_if_zero_calibration_loss=1/(n+1)) for n in COUNTS])
    np.testing.assert_allclose(availability.CP_upper_if_zero_events,
                               beta.ppf(0.975, 1, availability.units), rtol=1e-12)
    historical = pd.read_csv(inputs[-2])
    historical = historical[historical.family.isin(['families', 'feature_reduction', 'imbalance'])].copy()
    assert len(historical) == 15
    np.testing.assert_allclose(historical.delta_min,
                               np.maximum(historical.equivalence_low.abs(), historical.equivalence_high.abs()),
                               rtol=1e-12)
    historical['historical_margin_ratio'] = historical.delta_min/historical.margin
    historical['interpretation'] = 'descriptive_interval_containment_not_power_or_sample_size'
    comparison = historical[['family', 'A', 'B', 'metric', 'family_size', 'equivalence_level',
                             'equivalence_low', 'equivalence_high', 'margin', 'delta_min',
                             'historical_margin_ratio', 'interpretation']]
    outputs = []
    for name, df in [('requirements', requirements), ('availability', availability), ('historical_precision', comparison)]:
        path = OUT / f'{name}.csv'
        df.to_csv(path, index=False)
        outputs.append(path)
    labels = {'families': 'Family', 'feature_reduction': 'Features', 'imbalance': 'Imbalance'}
    tex = [r'\begin{tabular}{rrrrr}', r'\toprule',
           r'Target & $n_{\rm eval}$ & Deficit vs 16 & $n_{\rm cal}$ & Deficit vs 16 \\', r'\midrule']
    for row in requirements.itertuples(index=False):
        tex.append(f'{row.target:.2f} & {row.n_evaluation_zero_events} & {row.deficit_evaluation_vs_16} & '
                   f'{row.n_calibration_CRC} & {row.deficit_calibration_vs_16}' + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    table_path = ROOT / 'paper/supplement/tables/tab_unit_evidence_requirements.tex'
    table_path.write_text('\n'.join(tex)+'\n', encoding='utf-8')
    outputs.append(table_path)
    tex = [r'\begin{tabular}{llrrrr}', r'\toprule',
           r'Contrast & Metric & $m$ & Level & Margin & Historical $\delta_{\min}$ \\', r'\midrule']
    for row in comparison.itertuples(index=False):
        if row.family == 'families':
            label = 'RF $-$ ' + ('LightGBM' if row.B.startswith('lgbm') else 'XGBoost')
        elif row.family == 'feature_reduction':
            label = ('LightGBM' if row.A.startswith('lgbm') else 'RF') + ': F246 $-$ F115'
        else:
            label = ('P1: weights' if row.metric == 'CMR' else
                     ('P2: Gaussian $-$ none' if row.B.endswith('none') else 'P3: Gaussian $-$ SMOTE'))
        metric = 'Macro-F1' if row.metric == 'macro_F1' else row.metric
        tex.append(f'{label} & {metric} & {row.family_size} & {100*row.equivalence_level:.2f}\\% & '
                   f'{row.margin:.3f} & {row.delta_min:.3f}' + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    precision_path = ROOT / 'paper/supplement/tables/tab_unit_evidence_precision.tex'
    precision_path.write_text('\n'.join(tex)+'\n', encoding='utf-8')
    outputs.append(precision_path)
    report = ['# A1 — Requisitos de evidência por unidade', '',
              f'Protocolo `docs/protocolos/A1.md`; SHA-256 `{sha(PROTOCOL)}`. Cálculo analítico, sem novos ajustes.', '',
              '## Unidades disponíveis', '',
              f'População auditada: {len(population)} ciclos, {population.Serial_Number.nunique()} SN, '
              f'{units} SN críticas ({class_units["F03"]} F03, {class_units["F06"]} F06). '
              'Treino e teste têm SN/ciclos disjuntos. A exclusão de 1107 afecta um ciclo normal.', '',
              '## Requisitos por alvo — papéis e garantias distintos', '', markdown(requirements), '',
              'Avaliação: U(n)=1−0,025^(1/n), limite superior Clopper–Pearson bilateral de 95% '
              'com zero unidades apresentando algum miss. n=ceil[ln(0,025)/ln(1−p0)]. '
              'Pressupõe unidades independentes com probabilidade comum e exposição definida; '
              'não é limite de CMR ponderado por ciclos, nem cálculo de poder. '
              'Cada alvo é individual, sem garantia simultânea para a tabela.', '',
              'CRC: com perda média de calibração igual a zero, o piso do critério é 1/(n+1); '
              'n=ceil(1/alpha−1). A perda é a fracção de ciclos críticos com miss na unidade e '
              'a garantia refere-se à perda média esperada de uma nova unidade permutável. '
              'Os mínimos são apenas condição necessária para admissibilidade, não garantia de '
              'utilidade nem dimensão total de estudo. O fallback alarme para todos continua '
              'possível abaixo dos mínimos. Não comparar os dois n como se controlassem a mesma garantia.', '',
              'Déficits em relação a 16 são contas idealizadas para cada papel isoladamente. '
              'Não autorizam usar as mesmas 16 unidades em treino, calibração e avaliação; '
              'a obtenção de zero perdas também não é assegurada.', '',
              '## Limites para contagens disponíveis ou planeadas', '', markdown(availability), '',
              'Os limites nesta tabela são condicionais a zero eventos/perda. Não se afirma '
              'que as 16 unidades disponíveis, ou os subgrupos F03/F06, tenham apresentado zero misses. '
              'O teste histórico possui apenas duas unidades críticas; a sua exposição desigual '
              '(73 e um ciclo crítico) impede transferir o limite binário para risco por ciclo '
              'ou para uma exposição futura padronizada.', '',
              '## Precisão histórica e dimensionamento de comparações', '', markdown(comparison), '',
              'delta_min=max(|lo|,|hi|) é a margem necessária para conter o intervalo histórico. '
              'Para satisfazer o critério de equivalência estrito, a margem teria de ser maior '
              'do que esse valor. Não é mínimo efeito detectável, diferença com poder fixado '
              'ou requisito de tamanho amostral. As 15 linhas conservam os cálculos históricos.', '',
              'O E3 não seleccionou procedimento calibrado para CMR/FA0 e não avaliou DR/macro-F1. '
              'Consequentemente não se estima n por extrapolação quadrática desses intervalos. '
              'O dimensionamento confirmatório de comparações requer procedimento válido, '
              'efeito de interesse, poder, alocação independente de SN, exposições e '
              'heterogeneidade definidos num novo protocolo.', '',
              '## Verificação', '',
              'Limites binomiais coincidem com quantis beta do SciPy; cada n é mínimo '
              '(n satisfaz o alvo e n−1 não); CRC verificado em Decimal; 15 delta_min '
              'conferidos contra extremos dos intervalos. Todas as verificações passaram.', '']
    report_path = ROOT / 'data/unit_evidence_requirements.md'
    report_path.write_text('\n'.join(report), encoding='utf-8')
    outputs.append(report_path)
    for name, digest in hashes.items():
        assert sha(ROOT / name) == digest, name
    manifest = dict(item='A1', started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                    input_sha256=hashes, output_sha256={str(p.relative_to(ROOT)): sha(p) for p in outputs},
                    protocol_sha256=sha(PROTOCOL), freeze=freeze, python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    critical_units=units, units_by_critical_class=class_units,
                    checks='CP_beta_quantiles; integer_minimality; CRC_Decimal; historical_delta_min; disjoint_metadata',
                    deviations=[], status='complete')
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(requirements.to_string(index=False))
    print(availability.to_string(index=False))


if __name__ == '__main__':
    main()
