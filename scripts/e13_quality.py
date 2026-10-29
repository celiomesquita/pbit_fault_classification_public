"""Class-independent raw-signal audit on historical development units only."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/e13_quality_v3'
FROZEN = ROOT / 'data/ress_revision/freeze_v1'
META = ['Experiment_Index', 'Serial_Number', 'Serial_Run_Index', 'Fault_Mode', 'Severity_Index']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def longest_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def parse_source_values(tokens):
    values=pd.to_numeric(pd.Series(tokens),errors='coerce').to_numpy(float)
    invalid=[token for token,value in zip(tokens,values) if token.strip() and not np.isfinite(value)]
    return values,invalid


def measurement_tokens(rows, column):
    assert len(rows)==127 and all(not token.strip() for token in rows[3])
    return [rows[i][column] for i in range(4,127)]


def signal_quality(indices, values):
    """Input intentionally contains no labels, unit identity or predictions."""
    indices = np.asarray(indices)
    values = np.asarray(values, dtype=float)
    order = np.argsort(indices, kind='stable')
    indices, values = indices[order], values[order]
    expected = np.arange(1, 124)
    unique = np.unique(indices)
    structural = len(indices) != 123 or not np.array_equal(np.sort(indices), expected)
    finite = np.isfinite(values)
    d = np.diff(values)
    adjacent = np.diff(indices) == 1
    good_d = np.isfinite(d) & adjacent
    flat_edges = good_d & (np.abs(d) <= 1e-9)
    max_flat = longest_run(flat_edges) + 1 if finite.any() else 0
    zero = finite & (np.abs(values) < .05)
    # Index gaps interrupt runs even when numeric values coincide.
    zero_run = longest_run(np.r_[False, zero[1:] & zero[:-1] & adjacent])+1 if zero.any() else 0
    valid_d = d[good_d]
    mad = float(np.median(np.abs(valid_d-np.median(valid_d)))) if len(valid_d) else np.nan
    sigma = 1.4826*mad
    isolated = np.zeros(max(0, len(d)-1), dtype=bool)
    spike_defined = bool(np.isfinite(sigma) and sigma > 1e-12)
    if spike_defined:
        isolated = (good_d[:-1] & good_d[1:] & (np.sign(d[:-1]) != np.sign(d[1:]))
                    & (np.abs(d[:-1]) > 8*sigma) & (np.abs(d[1:]) > 8*sigma))
    std = float(np.std(values[finite])) if finite.any() else np.nan
    missing = 123 - len(set(indices[finite]).intersection(expected))
    coverage_bad = missing > 123/2
    constant = bool(np.isfinite(std) and std < 1e-6)
    suspicious = missing > 0 or constant or max_flat >= 10 or isolated.any()
    category = ('structural_problem' if structural else 'insufficient_coverage' if coverage_bad
                else 'suspect' if suspicious else 'no_flag_available_rules')
    return dict(n_rows=len(indices), n_finite=int(finite.sum()), n_missing_positions=int(missing),
                duplicate_indices=len(indices)-len(unique), out_of_range=int((~np.isin(indices, expected)).sum()),
                structural_problem=bool(structural), insufficient_coverage=bool(coverage_bad),
                signal_std=std, amplitude=float(np.ptp(values[finite])) if finite.any() else np.nan,
                median_abs=float(np.median(np.abs(values[finite]))) if finite.any() else np.nan,
                near_constant=constant, longest_constant_run=max_flat, longest_zero_run=zero_run,
                plateau_flag=bool(max_flat >= 10), max_jump=float(np.max(np.abs(valid_d))) if len(valid_d) else np.nan,
                difference_robust_sigma=sigma, spike_rule_defined=spike_defined,
                isolated_spike_count=int(isolated.sum()), isolated_spike_flag=bool(isolated.any()),
                quality_category=category)


def raw_original(raw, ids):
    with (ROOT / 'data/PBIT_atuador_DESCARAC_r1.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    columns = {int(x): i for i, x in enumerate(rows[1]) if x}
    raw_matrix = raw.pivot(index='Experiment_Index', columns='Sample_Index', values='Measurement')
    records = []
    for exp in ids:
        col = columns[int(exp)]
        source,tokens = parse_source_values(measurement_tokens(rows,col))
        actual = raw_matrix.loc[exp, np.arange(1,124)].to_numpy(float)
        finite = np.isfinite(source) & np.isfinite(actual)
        delta = float(np.max(np.abs(source[finite]-actual[finite]))) if finite.any() else 0.
        records.append(dict(Experiment_Index=exp, missing_mask_equal=bool(np.array_equal(np.isnan(source), np.isnan(actual))),
                            max_absolute_delta=delta, source_error_cells=len(tokens),source_error_tokens=';'.join(sorted(set(tokens))),
                            values_match=bool(np.allclose(source, actual, equal_nan=True, rtol=0, atol=1e-12))))
    return pd.DataFrame(records)


def main():
    inputs = [ROOT/'data/PBIT_atuador_DESCARAC_r1_long.csv', ROOT/'data/PBIT_atuador_DESCARAC_r1.csv',
              ROOT/'data/PBIT_atuador_DESCARAC_r1_long_clean.csv', FROZEN/'train_primary.csv',
              FROZEN/'long_primary.csv', FROZEN/'freeze_manifest.json', ROOT/'Orange/criar_atributos_segmentos.py']
    sources = [Path(__file__), ROOT/'tests/test_e13_quality.py', ROOT/'docs/protocolos/E13.md',
               ROOT/'docs/protocolos/E13_emenda1.md',ROOT/'docs/protocolos/E13_emenda2.md']
    assert not OUT.exists(), 'Preserve this audit; select a new version for an amendment.'
    previous = json.loads((FROZEN/'freeze_manifest.json').read_text(encoding='utf-8'))
    for name in ('train_primary.csv', 'long_primary.csv'):
        p=FROZEN/name
        assert sha(p) == previous['output_sha256'][p.relative_to(ROOT).as_posix()]
    OUT.mkdir()
    for p in sources:
        dest = OUT/'frozen_sources'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p,dest)
    write(OUT/'freeze.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
          input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
          source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
          environment={n:importlib.metadata.version(n) for n in ('numpy','pandas','matplotlib')},
          classifier_fits=0, physical_phase_map='awaiting explicit boundaries/variable confirmation'))
    historical = pd.read_csv(FROZEN/'train_primary.csv')
    development_units = set(historical.Serial_Number)
    raw_all = pd.read_csv(inputs[0])
    raw = raw_all[raw_all.Serial_Number.isin(development_units)].copy()
    del raw_all
    assert raw.Serial_Number.nunique() == 54
    meta = raw.groupby('Experiment_Index')[META[1:]].first().reset_index()
    assert raw.groupby('Experiment_Index')[META[1:]].nunique().le(1).all().all()
    records=[]
    matrix=[]
    ids=[]
    for exp,g in raw.groupby('Experiment_Index',sort=True):
        records.append(dict(Experiment_Index=exp, **signal_quality(g.Sample_Index, g.Measurement)))
        assert not g.Sample_Index.duplicated().any()
        matrix.append(g.set_index('Sample_Index').Measurement.reindex(range(1,124)).to_numpy(float))
        ids.append(exp)
    signals=np.asarray(matrix)
    audit=pd.DataFrame(records).merge(meta,on='Experiment_Index',validate='one_to_one')
    duplicate_rows=[]
    for decimals, tag in ((None,'exact'),(6,'rounded6')):
        buckets={}
        for exp,x in zip(ids,signals):
            v=x.copy() if decimals is None else np.round(x,decimals)
            v[v==0]=0. # identical +0/-0; missingness represented separately
            payload=np.isfinite(v).tobytes()+np.nan_to_num(v,nan=0.).astype('<f8').tobytes()
            key=hashlib.sha256(payload).hexdigest()
            buckets.setdefault(key,[]).append(exp)
        membership={}
        for key,group in buckets.items():
            if len(group)>1:
                for exp in group:
                    duplicate_rows.append(dict(precision=tag, group_hash=key, Experiment_Index=exp,group_size=len(group)))
                    membership[exp]=len(group)
        audit[f'duplicate_{tag}_size']=audit.Experiment_Index.map(membership).fillna(1).astype(int)
    duplicate=pd.DataFrame(duplicate_rows,columns=['precision','group_hash','Experiment_Index','group_size'])
    audit.loc[(audit.quality_category=='no_flag_available_rules') & (audit.duplicate_rounded6_size>1),'quality_category']='suspect'
    audit['historical_primary']=audit.Experiment_Index.isin(historical.Experiment_Index)
    audit['processable_candidate']=~(audit.structural_problem|audit.insufficient_coverage)
    audit['any_flag']=audit.quality_category!='no_flag_available_rules'
    audit.to_csv(OUT/'cycle_quality.csv',index=False)
    duplicate.to_csv(OUT/'duplicate_signals.csv',index=False)
    original=raw_original(raw,ids)
    original.to_csv(OUT/'original_comparison.csv',index=False)
    assert original.values_match.all() and original.missing_mask_equal.all()
    clean=pd.read_csv(inputs[2])
    clean=clean[clean.Serial_Number.isin(development_units)]
    joined=raw[['Experiment_Index','Sample_Index','Measurement']].merge(
        clean[['Experiment_Index','Sample_Index','Measurement']],on=['Experiment_Index','Sample_Index'],
        how='left',suffixes=('_raw','_clean'),validate='one_to_one',indicator=True)
    joined['imputed']=joined.Measurement_raw.isna() & joined.Measurement_clean.notna()
    joined['changed_finite']=joined.Measurement_raw.notna() & joined.Measurement_clean.notna() & (
        abs(joined.Measurement_raw-joined.Measurement_clean)>1e-12)
    joined[(joined.Measurement_raw.isna())|joined.changed_finite].to_csv(OUT/'raw_clean_differences.csv',index=False)
    aggregations=dict(n_cycles=('Experiment_Index','size'),n_flagged=('any_flag','sum'),
        n_missing_cycles=('n_missing_positions',lambda x:int((x>0).sum())),
        n_near_constant=('near_constant','sum'),n_plateaus=('plateau_flag','sum'),
        n_spikes=('isolated_spike_flag','sum'),n_processable=('processable_candidate','sum'),
        n_historical=('historical_primary','sum'))
    for group,name in (('Serial_Number','unit_quality'),('Fault_Mode','class_quality')):
        table=audit.groupby(group).agg(**aggregations).reset_index()
        table['flagged_fraction']=table.n_flagged/table.n_cycles
        table.to_csv(OUT/f'{name}.csv',index=False)
    position=raw.groupby('Sample_Index').Measurement.agg(n_values='count',median='median',q05=lambda x:x.quantile(.05),q95=lambda x:x.quantile(.95)).reset_index()
    position['n_missing']=len(ids)-position.n_values
    position.to_csv(OUT/'position_quality.csv',index=False)
    audit[META+['historical_primary','processable_candidate','quality_category']].to_csv(OUT/'population_manifest.csv',index=False)
    write(OUT/'phase_map_pending.json',dict(status='not confirmed',n_positions=123,
        author_information='fixed extension/contraction sequence with identifiable reversal',
        historical_source='Orange/criar_atributos_segmentos.py:14-20',
        historical_intervals=[dict(start=1,end=62,phase='extension'),dict(start=63,end=123,phase='contraction')],
        reversal_interval=None,measured_variable=None,unit=None,phase_features_executed=False))
    fig,ax=plt.subplots(figsize=(9,4))
    ax.fill_between(position.Sample_Index,position.q05,position.q95,alpha=.2,label='5th–95th percentiles')
    ax.plot(position.Sample_Index,position['median'],label='Median')
    ax.set(xlabel='Measurement index',ylabel='Recorded signal (unit unconfirmed)',title='Development signals — labels concealed')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'blind_signal_envelope.png',dpi=160);plt.close(fig)
    # Predefined examples: first ID for each flag; plots contain no class/SN/error data.
    chosen=[]
    for flag in ('n_missing_positions','near_constant','plateau_flag','isolated_spike_flag'):
        subset=audit[audit[flag].astype(float)>0].sort_values('Experiment_Index')
        if len(subset): chosen.append(int(subset.iloc[0].Experiment_Index))
    chosen=list(dict.fromkeys(chosen))
    if chosen:
        fig,axes=plt.subplots(len(chosen),1,figsize=(9,2.6*len(chosen)),squeeze=False)
        for i,exp in enumerate(chosen):
            ax=axes[i,0];ax.plot(range(1,124),signals[ids.index(exp)],linewidth=.9)
            ax.set(title=f'Anonymous trace {i+1}',xlabel='Measurement index',ylabel='Recorded signal')
        fig.tight_layout();fig.savefig(OUT/'blind_flagged_examples.png',dpi=160);plt.close(fig)
    pd.DataFrame({'anonymous_trace':range(1,len(chosen)+1),'Experiment_Index':chosen}).to_csv(OUT/'blind_example_key.csv',index=False)
    summary=dict(n_cycles=len(audit),n_units=len(development_units),historical_cycles=len(historical),
        added_for_independent_audit=audit.loc[~audit.historical_primary,'Experiment_Index'].tolist(),
        raw_missing_measurements=int(raw.Measurement.isna().sum()),
        cycles_with_missing=int((audit.n_missing_positions>0).sum()),
        structural_problems=int(audit.structural_problem.sum()),insufficient_coverage=int(audit.insufficient_coverage.sum()),
        near_constant_cycles=int(audit.near_constant.sum()),plateau_cycles=int(audit.plateau_flag.sum()),
        spike_flagged_cycles=int(audit.isolated_spike_flag.sum()),spike_rule_undefined=int((~audit.spike_rule_defined).sum()),
        categories={str(k):int(v) for k,v in audit.quality_category.value_counts().items()},
        processable_candidate_cycles=int(audit.processable_candidate.sum()),
        exact_duplicate_groups=int(duplicate[duplicate.precision=='exact'].group_hash.nunique()),
        rounded_duplicate_groups=int(duplicate[duplicate.precision=='rounded6'].group_hash.nunique()),
        original_records_verified=len(original),source_error_cells=int(original.source_error_cells.sum()),
        source_records_with_errors=int((original.source_error_cells>0).sum()),imputed_cells_in_clean=int(joined.imputed.sum()),
        finite_cells_changed_in_clean=int(joined.changed_finite.sum()),
        raw_records_absent_in_clean=int((joined['_merge']=='left_only').sum()),
        physically_invalid_confirmed=0,units_excluded=0,classifier_fits=0,
        physical_phase_map='pending explicit confirmation',unit_exclusion_threshold='not established')
    write(OUT/'summary.json',summary)
    outputs={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='freeze.json'}
    write(OUT/'completion.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
         freeze_sha256=sha(OUT/'freeze.json'),output_sha256=outputs,complete_global_audit=True,complete_physical_phase_audit=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
