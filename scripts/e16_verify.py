"""Independent verification of region ablations and their reused reference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from e14_verify import sha,read,independent_selection,scalar_metrics,CLASSES

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/e16_empirical_v1';E14=ROOT/'data/e14_empirical_v2'


def main():
    freeze=read(OUT/'freeze.json');completion=read(OUT/'completion.json')
    for kind in ['input_sha256','source_sha256']:
        for name,h in freeze[kind].items():assert sha(ROOT/name)==h,name
    count=decisions=hashes=specs=0;frames=[]
    arms=['M2_reference','only_R1','only_R2','only_R3','without_R1','without_R2','without_R3','limits_minus1','limits_plus1']
    for item in completion['completed']:
        folder=OUT/f"seed_{item['seed']}_outer_{item['outer']}";reference=E14/folder.name
        assert read(folder/'complete.json')==item
        for name,h in item['output_sha256'].items():assert sha(folder/name)==h;hashes+=1
        inner=pd.read_csv(folder/'inner_metadata.csv');outer=pd.read_csv(folder/'outer_metadata.csv')
        assert set(inner.Serial_Number).isdisjoint(outer.Serial_Number)
        pd.testing.assert_frame_equal(inner,pd.read_csv(reference/'inner_metadata.csv'))
        pd.testing.assert_frame_equal(outer,pd.read_csv(reference/'outer_metadata.csv'))
        boundaries={b['role']:b['boundaries'] for b in read(reference/'boundaries.json')}
        for spec in read(folder/'representation_specs.json'):
            expected=14 if spec['arm'].startswith('only') else 30 if spec['arm'].startswith('without') else 48
            assert spec['n_features']==expected
            assert len(spec['features'])==expected and len(set(spec['features']))==expected
            if spec['arm'].startswith(('only','without')):
                target='r'+spec['arm'][-1]
                for feature in spec['features']:
                    refs={token for token in feature.split('_') if token in ['r1','r2','r3']}
                    assert refs and (refs<={target} if spec['arm'].startswith('only') else target not in refs)
            shift=-1 if spec['arm']=='limits_minus1' else 1 if spec['arm']=='limits_plus1' else 0
            assert spec['boundaries']==[int(v)+shift for v in boundaries[spec['role']]]
            specs+=1
        selection=read(folder/'inner_selections.json');pred=pd.read_csv(folder/'external_predictions.csv');frames.append(pred)
        for arm in arms:
            inside=np.load(folder/f'{arm}_inner_probability.npy');outside=np.load(folder/f'{arm}_outer_probability.npy')
            assert inside.shape==(len(inner),8) and outside.shape==(len(outer),8)
            assert np.isfinite(inside).all() and np.isfinite(outside).all()
            assert np.allclose(inside.sum(axis=1),1) and np.allclose(outside.sum(axis=1),1)
            if arm=='M2_reference':
                assert np.array_equal(inside,np.load(reference/'M2_empirical_inner_probability.npy'))
                assert np.array_equal(outside,np.load(reference/'M2_empirical_outer_probability.npy'))
            primary=next(s for s in selection if s['arm']==arm and s['target']==.90 and s['budget']==.40)
            t=np.inf if primary['threshold']=='infinity' else primary['threshold']
            assert t==independent_selection(inner.Fault_Mode.to_numpy(),inside);count+=1
            for s in [s for s in selection if s['arm']==arm]:
                t=np.inf if s['threshold']=='infinity' else s['threshold'];alarm=outside[:,2:].sum(axis=1)>=t
                result=np.where(alarm,CLASSES[2:][outside[:,2:].argmax(axis=1)],CLASSES[:2][outside[:,:2].argmax(axis=1)])
                g=pred[(pred.arm==arm)&np.isclose(pred.target,s['target'])&np.isclose(pred.budget,s['budget'])]
                assert np.array_equal(g.Experiment_Index,outer.Experiment_Index) and np.array_equal(g.prediction,result)
                decisions+=len(g)
    pred=pd.concat(frames,ignore_index=True);metrics=pd.read_csv(OUT/'seed_metrics.csv');maximum=0.;checked=0
    for row in metrics.itertuples():
        g=pred[(pred.arm==row.arm)&(pred.seed==row.seed)&np.isclose(pred.target,row.target)&np.isclose(pred.budget,row.budget)]
        assert len(g)==1195 and g.Experiment_Index.nunique()==1195
        values=scalar_metrics(g.Fault_Mode.to_numpy(),g.prediction.to_numpy())
        for name,value in values.items():
            actual=getattr(row,name);assert np.isclose(actual,value,atol=1e-12,rtol=0,equal_nan=True)
            if np.isfinite(value):maximum=max(maximum,abs(actual-value))
        assert bool(row.external_feasible)==bool(values['min_recall']>=row.target-1e-12 and values['FA0']<=row.budget+1e-12)
        checked+=1
    summary=pd.read_csv(OUT/'summary.csv')
    for row in summary.itertuples():
        group=metrics[(metrics.arm==row.arm)&np.isclose(metrics.target,row.target)&np.isclose(metrics.budget,row.budget)]
        for column in ['FA0','min_recall','detect_F01','detect_F02','detect_F03','detect_F04','detect_F05','detect_F06']:
            assert np.isclose(getattr(row,column),group[column].mean(),atol=1e-12,rtol=0)
        assert row.external_feasible_seeds==group.external_feasible.sum()
    old=pd.read_csv(E14/'seed_metrics.csv');old=old[old.arm=='M2_empirical']
    reference=metrics[metrics.arm=='M2_reference'].merge(old,on=['seed','target','budget'],suffixes=('_new','_old'))
    assert len(reference)==90
    for col in ['FA0','min_recall','detect_F02','detect_F05']:
        assert np.allclose(reference[col+'_new'],reference[col+'_old'],atol=1e-12,rtol=0)
    result=dict(passed=True,input_hashes=len(freeze['input_sha256']),source_hashes=len(freeze['source_sha256']),
        output_hashes=hashes,representation_specs_verified=specs,independent_primary_selections=count,
        decision_rows_verified=decisions,seed_metric_rows_verified=checked,max_scalar_error=maximum,
        reference_reproduces_E14=True,new_fits=completion['new_fits'],
        verifier_sha256=sha(Path(__file__)),independent_helper_sha256=sha(ROOT/'scripts/e14_verify.py'),completion_sha256=sha(OUT/'completion.json'))
    p=OUT/'verification.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
