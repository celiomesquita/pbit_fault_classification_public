"""Independent verification of the E13 raw audit; does not import its producer."""
from itertools import groupby
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e13_quality_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    complete=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    assert sha(OUT/'freeze.json')==complete['freeze_sha256']
    for mapping in ('input_sha256','source_sha256'):
        for name,h in freeze[mapping].items():
            assert sha(ROOT/name)==h,name
    for name,h in complete['output_sha256'].items():
        assert sha(OUT/name)==h,name
    old=pd.read_csv(ROOT/'data/ress_revision/freeze_v1/train_primary.csv')
    raw=pd.read_csv(ROOT/'data/PBIT_atuador_DESCARAC_r1_long.csv')
    raw=raw[raw.Serial_Number.isin(old.Serial_Number.unique())]
    audit=pd.read_csv(OUT/'cycle_quality.csv').set_index('Experiment_Index')
    assert set(audit.index)==set(raw.Experiment_Index)
    assert set(raw.Serial_Number)==set(old.Serial_Number)
    checks=0
    for exp,g in raw.groupby('Experiment_Index'):
        g=g.sort_values('Sample_Index')
        x=g.Measurement.to_numpy(float)
        a=audit.loc[exp]
        finite=np.isfinite(x)
        std=np.nanstd(x)
        missing=int((~finite).sum())
        assert missing==a.n_missing_positions
        assert finite.sum()==a.n_finite
        assert np.isclose(std,a.signal_std,atol=1e-14,rtol=1e-12)
        assert bool(std<1e-6)==bool(a.near_constant)
        assert not g.Sample_Index.duplicated().any()
        assert set(g.Sample_Index)==set(range(1,124))
        same=np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (abs(np.diff(x))<=1e-9)
        flat=max([len(list(v))+1 for k,v in groupby(same) if k]+[1 if finite.any() else 0])
        zero=max([len(list(v)) for k,v in groupby(finite & (abs(x)<.05)) if k]+[0])
        assert flat==a.longest_constant_run and zero==a.longest_zero_run
        assert (missing<=61)==bool(a.processable_candidate)
        assert a.Serial_Number==g.Serial_Number.iloc[0] and a.Fault_Mode==g.Fault_Mode.iloc[0]
        assert bool(exp in set(old.Experiment_Index))==bool(a.historical_primary)
        checks+=1
    # Independent source parser: pandas skips the actual four metadata/separator rows.
    source=pd.read_csv(ROOT/'data/PBIT_atuador_DESCARAC_r1.csv',header=None,skiprows=4)
    identifiers=pd.read_csv(ROOT/'data/PBIT_atuador_DESCARAC_r1.csv',header=None,nrows=2).iloc[1].astype(int)
    source.columns=identifiers
    maximum=0.
    for exp,g in raw.groupby('Experiment_Index'):
        s=pd.to_numeric(source[exp],errors='coerce').to_numpy(float)
        x=g.sort_values('Sample_Index').Measurement.to_numpy(float)
        assert np.array_equal(np.isnan(s),np.isnan(x))
        assert np.allclose(s,x,atol=1e-12,rtol=0,equal_nan=True)
        d=abs(s-x);maximum=max(maximum,float(np.nanmax(d)))
    units=pd.read_csv(OUT/'unit_quality.csv').set_index('Serial_Number')
    for sn,g in audit.groupby('Serial_Number'):
        assert len(g)==units.loc[sn,'n_cycles']
        assert g.any_flag.sum()==units.loc[sn,'n_flagged']
        assert np.isclose(g.any_flag.mean(),units.loc[sn,'flagged_fraction'])
    result=dict(passed=True,input_hashes=len(freeze['input_sha256']),source_hashes=len(freeze['source_sha256']),
        output_hashes=len(complete['output_sha256']),cycles_independently_checked=checks,
        units_independently_checked=len(units),original_values_independently_checked=len(raw),
        max_original_absolute_delta=maximum,classifier_fits=0,
        scope='raw metrics/metadata/coverage/source correspondence and hashes; no physical phase or instrumental validation',
        completion_sha256=sha(OUT/'completion.json'),verifier_sha256=sha(Path(__file__)))
    target=OUT/'verification.json'
    assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
