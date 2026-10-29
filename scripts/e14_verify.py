"""Independent scalar/prediction checks for E14.e; no model fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e14_empirical_v2'
CLASSES=np.array(['N01','N02','F01','F02','F03','F04','F05','F06'])


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def scalar_metrics(y,prediction):
    alarm=np.isin(prediction,CLASSES[2:]);result={}
    for cls in CLASSES[2:]:
        actual=y==cls
        result[f'detect_{cls}']=float(alarm[actual].mean()) if actual.any() else np.nan
        result[f'exact_{cls}']=float((prediction[actual]==cls).mean()) if actual.any() else np.nan
    for cls in CLASSES[:2]:
        actual=y==cls;result[f'FA0_{cls}']=float(alarm[actual].mean()) if actual.any() else np.nan
    normal=np.isin(y,CLASSES[:2]);result['FA0']=float(alarm[normal].mean()) if normal.any() else np.nan
    result['min_recall']=min(result[f'detect_{cls}'] for cls in CLASSES[2:7])
    return result


def independent_selection(y,probability):
    q=probability[:,2:].sum(axis=1)
    thresholds=np.r_[np.inf,np.unique(q)[::-1]]
    if thresholds[-1]>0:thresholds=np.r_[thresholds,0.]
    alarms=q[None,:]>=thresholds[:,None]
    recalls=np.column_stack([alarms[:,y==cls].mean(axis=1) for cls in CLASSES[2:7]])
    fa0=alarms[:,np.isin(y,CLASSES[:2])].mean(axis=1)
    good=np.flatnonzero((fa0<=.40+1e-12)&(recalls.min(axis=1)>=.90-1e-12))
    if len(good):
        best=min(good,key=lambda i:(fa0[i],-recalls[i].min(),-recalls[i].mean(),-thresholds[i]))
    else:
        good=np.flatnonzero(fa0<=.40+1e-12)
        best=min(good,key=lambda i:(-recalls[i].min(),fa0[i],-recalls[i].mean(),-thresholds[i]))
    return float(thresholds[best])


def main():
    freeze=read(OUT/'freeze.json');completion=read(OUT/'completion.json')
    for kind in ['input_sha256','source_sha256']:
        for name,value in freeze[kind].items():assert sha(ROOT/name)==value,name
    old=read(ROOT/'data/e1_full_v1/completion_manifest.json')['completed']
    originals={(x['seed'],x['outer']):ROOT/x['output'] for x in old}
    population=pd.read_csv(ROOT/'data/ress_revision/freeze_v1/train_primary.csv')
    metadata=population.set_index('Experiment_Index')
    long=pd.read_csv(ROOT/'data/ress_revision/freeze_v1/long_primary.csv')
    long=long[long.Experiment_Index.isin(metadata.index)]
    signal=long.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement')
    for origin in originals.values():
        manifest=read(origin/'run_manifest.json')
        original_freeze=origin/'protocol_freeze.json'
        assert sha(original_freeze)==manifest['protocol_freeze_sha256']
        source=read(original_freeze)['source_sha256']
        assert source['src/pbit/augmentation.py']==sha(ROOT/'src/pbit/augmentation.py')
    output_count=selection_count=decision_rows=probability_arrays=region_objectives=0
    for item in completion['completed']:
        folder=OUT/f"seed_{item['seed']}_outer_{item['outer']}"
        for name,value in item['output_sha256'].items():
            assert sha(folder/name)==value,name;output_count+=1
        current=read(folder/'complete.json');assert current==item
        inner=pd.read_csv(folder/'inner_metadata.csv');outer=pd.read_csv(folder/'outer_metadata.csv')
        assert set(inner.Serial_Number).isdisjoint(outer.Serial_Number)
        for b in read(folder/'boundaries.json'):
            assert set(b['training_units']).isdisjoint(b['evaluation_units'])
            first,second=b['boundaries'];assert first>=15 and second-first>=15 and 123-second>=15
            train_meta=metadata.loc[b['real_ids']]
            evaluation_meta=metadata.loc[b['evaluation_ids']]
            assert set(train_meta.Serial_Number)==set(b['training_units'])
            assert set(evaluation_meta.Serial_Number)==set(b['evaluation_units'])
            x=signal.loc[b['real_ids']].to_numpy(float)
            centered=x-x.mean(axis=1,keepdims=True);std=x.std(axis=1,keepdims=True)
            z=np.zeros_like(x);valid=std[:,0]>=1e-6;z[valid]=centered[valid]/std[valid]
            counts=train_meta.Serial_Number.value_counts()
            weights=1/train_meta.Serial_Number.map(counts).to_numpy()/len(counts)
            cost=0.
            for start,end in [(0,first),(first,second),(second,123)]:
                region=z[:,start:end];residual=region-region.mean(axis=1,keepdims=True)
                cost+=float(weights @ (residual*residual).sum(axis=1))
            assert np.isclose(cost,b['objective'],atol=1e-9,rtol=1e-10)
            region_objectives+=1
        selections=read(folder/'inner_selections.json')
        predictions=pd.read_csv(folder/'external_predictions.csv')
        for arm in ['M0_F246','M1_uniform','M2_empirical','M3_combined']:
            inside=np.load(folder/f'{arm}_inner_probability.npy');outside=np.load(folder/f'{arm}_outer_probability.npy')
            for probability,expected in [(inside,len(inner)),(outside,len(outer))]:
                assert probability.shape==(expected,8) and np.isfinite(probability).all()
                assert (probability>=0).all() and np.allclose(probability.sum(axis=1),1)
                probability_arrays+=1
            if arm=='M0_F246':
                origin=originals[(item['seed'],item['outer'])]
                assert np.array_equal(inside,np.load(origin/'untuned_inner_probability.npy'))
                assert np.array_equal(outside,np.load(origin/'untuned_external_probability.npy'))
            primary=next(s for s in selections if s['arm']==arm and s['target']==.90 and s['budget']==.40)
            expected=independent_selection(inner.Fault_Mode.to_numpy(),inside)
            stored=np.inf if primary['threshold']=='infinity' else primary['threshold']
            assert expected==stored,(item,arm,expected,stored)
            selection_count+=1
            for selection in [s for s in selections if s['arm']==arm]:
                threshold=np.inf if selection['threshold']=='infinity' else selection['threshold']
                alarm=outside[:,2:].sum(axis=1)>=threshold
                predicted=np.where(alarm,CLASSES[2:][outside[:,2:].argmax(axis=1)],CLASSES[:2][outside[:,:2].argmax(axis=1)])
                g=predictions[(predictions.arm==arm)&np.isclose(predictions.target,selection['target'])&np.isclose(predictions.budget,selection['budget'])]
                assert np.array_equal(g.Experiment_Index,outer.Experiment_Index)
                assert np.array_equal(g.prediction,predicted)
                decision_rows+=len(g)
    allpred=pd.concat([pd.read_csv(OUT/f"seed_{x['seed']}_outer_{x['outer']}"/'external_predictions.csv') for x in completion['completed']],ignore_index=True)
    metrics=pd.read_csv(OUT/'seed_metrics.csv')
    maximum=0.;checks=0
    for row in metrics.itertuples():
        g=allpred[(allpred.arm==row.arm)&(allpred.seed==row.seed)&np.isclose(allpred.target,row.target)&np.isclose(allpred.budget,row.budget)]
        expected=scalar_metrics(g.Fault_Mode.to_numpy(),g.prediction.to_numpy())
        for name,value in expected.items():
            actual=getattr(row,name);assert np.isclose(value,actual,atol=1e-12,rtol=0,equal_nan=True)
            if np.isfinite(value):maximum=max(maximum,abs(value-actual))
        assert bool(row.external_feasible)==bool(expected['min_recall']>=row.target-1e-12 and expected['FA0']<=row.budget+1e-12)
        checks+=1
    e10=pd.read_csv(ROOT/'data/e10_v1/summary.csv')
    control=metrics[metrics.arm=='M0_F246']
    original=e10[e10.arm=='B1_complete']
    # E10 summary may contain per-seed rows; match its actual schema explicitly.
    if 'seed' in original:
        joined=control.merge(original,on=['seed','target','budget'],suffixes=('_new','_old'))
        assert len(joined)==90
        for column in ['FA0','min_recall','detect_F02','detect_F05']:
            assert np.allclose(joined[column+'_new'],joined[column+'_old'],atol=1e-12,rtol=0)
    else:
        aggregate=control.groupby(['target','budget'])[['FA0','min_recall','detect_F02','detect_F05']].mean().reset_index()
        joined=aggregate.merge(original,on=['target','budget'],suffixes=('_new','_old'))
        assert len(joined)==9
        for column in ['FA0','min_recall','detect_F02','detect_F05']:
            assert np.allclose(joined[column+'_new'],joined[column+'_old'],atol=1e-12,rtol=0)
    result=dict(passed=True,input_hashes=len(freeze['input_sha256']),source_hashes=len(freeze['source_sha256']),
        output_hashes=output_count,probability_arrays=probability_arrays,independent_primary_selections=selection_count,
        independent_training_region_objectives=region_objectives,
        independently_checked_decision_rows=decision_rows,independent_seed_metric_rows=checks,max_scalar_error=maximum,
        control_reproduces_E10_B1=True,new_classifier_fits=completion['new_fits'],
        limitation='No physical phase validation or independent cohort; regions are numeric only.',
        verifier_sha256=sha(Path(__file__)),completion_sha256=sha(OUT/'completion.json'))
    target=OUT/'verification.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
