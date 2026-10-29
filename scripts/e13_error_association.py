"""Post-audit descriptive association; quality flags are never modified."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e13_error_association'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    inputs=[ROOT/'data/e13_quality_v3/cycle_quality.csv',ROOT/'data/e12_diagnostic/cycle_errors.csv',
            ROOT/'data/e13_quality_v3/verification.json',ROOT/'data/e12_diagnostic/verification.json']
    sources=[Path(__file__),ROOT/'docs/protocolos/E13d.md']
    assert not OUT.exists();OUT.mkdir()
    write(OUT/'freeze.json',dict(utc=datetime.now(timezone.utc).isoformat(),
         input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
         source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
    for p in sources:
        snapshot=OUT/'frozen_sources'/p.relative_to(ROOT);snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(p.read_bytes())
    quality=pd.read_csv(inputs[0]);errors=pd.read_csv(inputs[1])
    selected=quality[quality.historical_primary][['Experiment_Index','any_flag','quality_category']]
    joined=errors.merge(selected,on='Experiment_Index',validate='many_to_one')
    assert len(joined)==len(errors) and joined.seed_count.eq(10).all()
    records=[]
    for (arm,cls),part in joined.groupby(['arm','Fault_Mode']):
        total=part.miss_fraction.sum()
        for flagged,g in part.groupby('any_flag'):
            records.append(dict(arm=arm,Fault_Mode=cls,flagged=bool(flagged),n_cycles=len(g),n_units=g.Serial_Number.nunique(),
                mean_recall=float(g.detect_recall.mean()),mean_misses_per_seed=float(g.miss_fraction.sum()),
                miss_share=float(g.miss_fraction.sum()/total) if total else None,exposure_share=len(g)/len(part)))
    grouped=pd.DataFrame(records);grouped.to_csv(OUT/'class_flag_association.csv',index=False)
    units=joined.groupby(['arm','Fault_Mode','Serial_Number','any_flag']).agg(n_cycles=('Experiment_Index','size'),
        mean_recall=('detect_recall','mean'),mean_misses_per_seed=('miss_fraction','sum')).reset_index()
    units.to_csv(OUT/'unit_flag_association.csv',index=False)
    checks=0
    for r in grouped.itertuples():
        subset=joined[(joined.arm==r.arm)&(joined.Fault_Mode==r.Fault_Mode)&(joined.any_flag==r.flagged)]
        recall=1-float(subset.miss_count.to_numpy().sum())/(10*len(subset))
        assert np.isclose(recall,r.mean_recall,atol=1e-12,rtol=0)
        assert np.isclose(subset.miss_count.sum()/10,r.mean_misses_per_seed)
        checks+=1
    write(OUT/'completion.json',dict(utc=datetime.now(timezone.utc).isoformat(),classifier_fits=0,verified_groups=checks,
        output_sha256={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    print(grouped.to_string(index=False))


if __name__=='__main__':main()
