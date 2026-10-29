"""Descriptive summaries of the fixed E14.e batch, including all scenarios."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pbit.triage_policy import evaluate
OUT=ROOT/'data/e14_empirical_v2'


def summarize():
    completion=json.loads((OUT/'completion.json').read_text())
    frames=[];selections=[];bounds=[]
    for item in completion['completed']:
        folder=OUT/f"seed_{item['seed']}_outer_{item['outer']}"
        frames.append(pd.read_csv(folder/'external_predictions.csv'))
        for s in json.loads((folder/'inner_selections.json').read_text()):selections.append(dict(seed=item['seed'],outer=item['outer'],**s))
        for b in json.loads((folder/'boundaries.json').read_text()):bounds.append(dict(seed=item['seed'],outer=item['outer'],role=b['role'],first=b['boundaries'][0],second=b['boundaries'][1],objective=b['objective']))
    pred=pd.concat(frames,ignore_index=True)
    rows=[]
    for (arm,target,budget,seed),g in pred.groupby(['arm','target','budget','seed']):
        assert len(g)==1195 and g.Experiment_Index.nunique()==1195
        result=evaluate(g.Fault_Mode.to_numpy(),g.prediction.to_numpy())
        rows.append(dict(arm=arm,target=target,budget=budget,seed=seed,
            external_feasible=bool(result['min_recall']>=target-1e-12 and result['FA0']<=budget+1e-12),**result))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'seed_metrics.csv',index=False)
    inner=pd.DataFrame(selections)
    inner[['seed','outer','arm','target','budget','feasible','status','FA0','min_recall']].to_csv(OUT/'inner_admissibility.csv',index=False)
    pd.DataFrame(bounds).to_csv(OUT/'region_boundaries.csv',index=False)
    columns=['FA0','min_recall','detect_F01','detect_F02','detect_F03','detect_F04','detect_F05','detect_F06','exact_F06','FA0_N01','FA0_N02','macro_F1']
    summary=metrics.groupby(['arm','target','budget'])[columns].mean().reset_index()
    counts=metrics.groupby(['arm','target','budget']).external_feasible.sum().rename('external_feasible_seeds').reset_index()
    internal=inner.groupby(['arm','target','budget']).feasible.sum().rename('inner_feasible_folds').reset_index()
    summary=summary.merge(counts).merge(internal);summary.to_csv(OUT/'summary.csv',index=False)
    primary=pred[np.isclose(pred.target,.90)&np.isclose(pred.budget,.40)].copy()
    primary['detected']=primary.prediction.str.startswith('F')
    unit=primary.groupby(['arm','Serial_Number','Fault_Mode']).agg(cycle_seed_rows=('Experiment_Index','size'),
         n_cycles=('Experiment_Index','nunique'),alarm_rate=('detected','mean')).reset_index()
    unit.to_csv(OUT/'primary_unit_class_metrics.csv',index=False)
    contrasts=[]
    for left,right in [('M2_empirical','M1_uniform'),('M3_combined','M0_F246')]:
        a=metrics[metrics.arm==left].set_index(['target','budget','seed'])
        b=metrics[metrics.arm==right].set_index(['target','budget','seed'])
        for key,row in (a[columns]-b[columns]).iterrows():
            contrasts.append(dict(contrast=f'{left} minus {right}',target=key[0],budget=key[1],seed=key[2],**row.to_dict()))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_seed_differences.csv',index=False)
    main=summary[np.isclose(summary.target,.90)&np.isclose(summary.budget,.40)]
    print(main[['arm','FA0','min_recall','detect_F02','detect_F05','inner_feasible_folds','external_feasible_seeds']].to_string(index=False),flush=True)


if __name__=='__main__':summarize()
