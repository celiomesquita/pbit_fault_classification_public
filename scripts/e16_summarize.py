"""All empirical-region ablations, without selecting an external winner."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pbit.triage_policy import evaluate
from e14_run import read
OUT=ROOT/'data/e16_empirical_v1'


def summarize():
    complete=read(OUT/'completion.json');preds=[];selections=[]
    for item in complete['completed']:
        folder=OUT/f"seed_{item['seed']}_outer_{item['outer']}";preds.append(pd.read_csv(folder/'external_predictions.csv'))
        selections.extend(dict(seed=item['seed'],outer=item['outer'],**s) for s in read(folder/'inner_selections.json'))
    pred=pd.concat(preds,ignore_index=True);rows=[]
    for (arm,target,budget,seed),g in pred.groupby(['arm','target','budget','seed']):
        assert len(g)==1195 and g.Experiment_Index.nunique()==1195
        result=evaluate(g.Fault_Mode.to_numpy(),g.prediction.to_numpy())
        rows.append(dict(arm=arm,target=target,budget=budget,seed=seed,
            external_feasible=bool(result['min_recall']>=target-1e-12 and result['FA0']<=budget+1e-12),**result))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'seed_metrics.csv',index=False);inner=pd.DataFrame(selections)
    inner[['arm','seed','outer','target','budget','feasible','status']].to_csv(OUT/'inner_admissibility.csv',index=False)
    columns=['FA0','min_recall','detect_F01','detect_F02','detect_F03','detect_F04','detect_F05','detect_F06','FA0_N01','FA0_N02']
    summary=metrics.groupby(['arm','target','budget'])[columns].mean().reset_index()
    summary=summary.merge(metrics.groupby(['arm','target','budget']).external_feasible.sum().rename('external_feasible_seeds').reset_index())
    summary=summary.merge(inner.groupby(['arm','target','budget']).feasible.sum().rename('inner_feasible_folds').reset_index());summary.to_csv(OUT/'summary.csv',index=False)
    baseline=metrics[metrics.arm=='M2_reference'].set_index(['seed','target','budget']);diffs=[]
    for arm in sorted(set(metrics.arm)-{'M2_reference'}):
        a=metrics[metrics.arm==arm].set_index(['seed','target','budget'])
        for key,row in (a[columns]-baseline[columns]).iterrows():diffs.append(dict(arm=arm,seed=key[0],target=key[1],budget=key[2],**row.to_dict()))
    pd.DataFrame(diffs).to_csv(OUT/'paired_seed_differences.csv',index=False)
    primary=pred[np.isclose(pred.target,.90)&np.isclose(pred.budget,.40)].copy();primary['alarm']=primary.prediction.str.startswith('F')
    primary.groupby(['arm','Serial_Number','Fault_Mode']).agg(n_cycles=('Experiment_Index','nunique'),alarm_rate=('alarm','mean')).reset_index().to_csv(OUT/'primary_unit_class_metrics.csv',index=False)
    print(summary[np.isclose(summary.target,.90)&np.isclose(summary.budget,.40)][['arm','FA0','detect_F02','detect_F05','inner_feasible_folds','external_feasible_seeds']].to_string(index=False),flush=True)


if __name__=='__main__':summarize()
