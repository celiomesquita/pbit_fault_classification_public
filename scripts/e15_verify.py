"""Independent E15.g role/filter/decision and metric verification."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from e14_verify import sha,read,independent_selection,scalar_metrics,CLASSES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e15_global_v1'


def main():
    freeze=read(OUT/'freeze.json');completion=read(OUT/'completion.json')
    for kind in ['input_sha256','source_sha256']:
        for name,h in freeze[kind].items():assert sha(ROOT/name)==h,name
    quality=pd.read_csv(ROOT/'data/e13_quality_v3/cycle_quality.csv')
    quality=quality[quality.processable_candidate].set_index('Experiment_Index')
    assert len(quality)==1196 and quality.any_flag.sum()==65
    outer_assignment=pd.read_csv(ROOT/'data/ress_revision/freeze_v1/outer_assignment.csv')
    inner_assignment=pd.read_csv(ROOT/'data/ress_revision/freeze_v1/inner_assignment.csv')
    decisions=selections=roles=hashes=0;frames=[]
    for item in completion['completed']:
        folder=OUT/f"seed_{item['seed']}_outer_{item['outer']}"
        assert read(folder/'complete.json')==item
        for name,h in item['output_sha256'].items():assert sha(folder/name)==h;hashes+=1
        frame=pd.read_csv(folder/'external_predictions.csv');frames.append(frame)
        internal=pd.read_csv(folder/'inner_metadata.csv');external=pd.read_csv(folder/'outer_metadata.csv')
        assert set(internal.Serial_Number).isdisjoint(external.Serial_Number)
        assignment=outer_assignment[outer_assignment.seed==item['seed']]
        outer_units=set(assignment.loc[assignment.outer_fold==item['outer'],'Serial_Number'])
        outer_ids=set(quality.index[quality.Serial_Number.isin(outer_units)])
        assert set(external.Experiment_Index)==outer_ids
        inner_map=inner_assignment[(inner_assignment.seed==item['seed'])&(inner_assignment.outer_fold==item['outer'])]
        for role in read(folder/'training_roles.json'):
            if role['role']=='outer':held=outer_units
            else:held=set(inner_map.loc[inner_map.inner_fold==int(role['role'][-1]),'Serial_Number'])
            parent=set(quality.index)-outer_ids
            if role['role']=='outer':expected_eval=outer_ids;expected_fit=parent
            else:
                expected_eval={i for i in parent if quality.loc[i,'Serial_Number'] in held}
                expected_fit=parent-expected_eval
            removed={i for i in expected_fit if quality.loc[i,'any_flag']} if role['policy']=='Q1' else set()
            assert set(role['retained_ids'])==expected_fit-removed
            assert set(role['removed_ids'])==removed
            assert set(role['evaluation_ids'])==expected_eval
            assert set(role['training_units'])==set(quality.loc[list(expected_fit-removed),'Serial_Number'])
            assert set(role['training_units'])==set(quality.loc[list(expected_fit),'Serial_Number'])
            assert set(role['training_units']).isdisjoint(role['evaluation_units'])
            assert role['real_cycles']==len(expected_fit-removed)
            first,second=role['boundaries'];assert min(first,second-first,123-second)>=15
            roles+=1
        stored=read(folder/'inner_selections.json')
        for arm in ['Q0_F246','Q0_empirical','Q1_F246','Q1_empirical']:
            inside=np.load(folder/f'{arm}_inner_probability.npy');outside=np.load(folder/f'{arm}_outer_probability.npy')
            assert inside.shape==(len(internal),8) and outside.shape==(len(external),8)
            assert np.isfinite(inside).all() and np.isfinite(outside).all()
            assert np.allclose(inside.sum(axis=1),1) and np.allclose(outside.sum(axis=1),1)
            primary=next(s for s in stored if s['arm']==arm and s['target']==.90 and s['budget']==.40)
            threshold=np.inf if primary['threshold']=='infinity' else primary['threshold']
            assert threshold==independent_selection(internal.Fault_Mode.to_numpy(),inside)
            selections+=1
            for s in [x for x in stored if x['arm']==arm]:
                t=np.inf if s['threshold']=='infinity' else s['threshold']
                alarm=outside[:,2:].sum(axis=1)>=t
                predicted=np.where(alarm,CLASSES[2:][outside[:,2:].argmax(axis=1)],CLASSES[:2][outside[:,:2].argmax(axis=1)])
                g=frame[(frame.arm==arm)&np.isclose(frame.target,s['target'])&np.isclose(frame.budget,s['budget'])]
                assert np.array_equal(g.Experiment_Index,external.Experiment_Index)
                assert np.array_equal(g.prediction,predicted)
                assert np.array_equal(g.any_flag,g.Experiment_Index.map(quality.any_flag))
                decisions+=len(g)
    pred=pd.concat(frames,ignore_index=True);metrics=pd.read_csv(OUT/'seed_metrics.csv');checks=0;maximum=0.
    for row in metrics.itertuples():
        g=pred[(pred.arm==row.arm)&(pred.seed==row.seed)&np.isclose(pred.target,row.target)&np.isclose(pred.budget,row.budget)]
        assert len(g)==1196 and g.Experiment_Index.nunique()==1196
        values=scalar_metrics(g.Fault_Mode.to_numpy(),g.prediction.to_numpy())
        for name,value in values.items():
            actual=getattr(row,name);assert np.isclose(actual,value,atol=1e-12,rtol=0,equal_nan=True)
            if np.isfinite(value):maximum=max(maximum,abs(actual-value))
        assert bool(row.external_feasible)==bool(values['min_recall']>=row.target-1e-12 and values['FA0']<=row.budget+1e-12)
        checks+=1
    summary=pd.read_csv(OUT/'summary.csv')
    for row in summary.itertuples():
        group=metrics[(metrics.arm==row.arm)&np.isclose(metrics.target,row.target)&np.isclose(metrics.budget,row.budget)]
        for column in ['FA0','min_recall','detect_F01','detect_F02','detect_F03','detect_F04','detect_F05','detect_F06']:
            assert np.isclose(getattr(row,column),group[column].mean(),atol=1e-12,rtol=0)
        assert row.external_feasible_seeds==group.external_feasible.sum()
    result=dict(passed=True,input_hashes=len(freeze['input_sha256']),source_hashes=len(freeze['source_sha256']),
        output_hashes=hashes,training_roles_verified=roles,independent_primary_selections=selections,
        decision_rows_verified=decisions,seed_metric_rows_verified=checks,max_scalar_error=maximum,
        new_fits=completion['new_fits'],common_evaluation_cycles=1196,whole_units_excluded=0,
        verifier_sha256=sha(Path(__file__)),independent_helper_sha256=sha(ROOT/'scripts/e14_verify.py'),
        completion_sha256=sha(OUT/'completion.json'))
    path=OUT/'verification.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
