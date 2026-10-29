"""E14.e fixed empirical-region contrast with verified E1/E10 reuse."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pbit.augmentation import augment_training_set
from pbit.triage_policy import curve,select_curve,predict,TARGETS,BUDGETS
from nested_cv_e1 import base_params,fit_probability,load_cache
from e14_regions import learn_regions,region_features

OUT=ROOT/'data/e14_empirical_v2'
FROZEN=ROOT/'data/ress_revision/freeze_v1'
ARMS=('M0_F246','M1_uniform','M2_empirical','M3_combined')
META=['Experiment_Index','Serial_Number','Serial_Run_Index','Fault_Mode','Severity_Index']
SCENARIOS=[(r,b) for r in TARGETS for b in BUDGETS]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def serial(v):
    if isinstance(v,dict): return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [serial(x) for x in v]
    if isinstance(v,(np.integer,np.bool_)):return v.item()
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return 'infinity' if v==np.inf else None
    return v


def write(p,v): Path(p).write_text(json.dumps(serial(v),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def dependencies(item):
    folder=ROOT/item['output']
    assert sha(folder/'run_manifest.json')==item['run_manifest_sha256']
    previous=read(folder/'run_manifest.json')
    names=['untuned_inner_probability.npy','untuned_external_probability.npy']
    for role in ['inner0','inner1','inner2','inner3','inner4','outer']:
        names += [f'cache/{role}.npz',f'cache/{role}_training_metadata.csv',f'cache/{role}_evaluation_metadata.csv']
    for name in names: assert sha(folder/name)==previous['output_sha256'][name],name
    return [folder/'run_manifest.json']+[folder/name for name in names]


def freeze():
    items=read(ROOT/'data/e1_full_v1/completion_manifest.json')['completed']
    assert len(items)==50
    sources=[Path(__file__),ROOT/'scripts/e14_regions.py',ROOT/'scripts/e14_summarize.py',ROOT/'tests/test_e14_regions.py',
             ROOT/'docs/protocolos/E14_empirico.md',ROOT/'docs/protocolos/E14_empirico_emenda1.md',
             ROOT/'scripts/nested_cv_e1.py',ROOT/'scripts/nested_cv_controlled.py',
             ROOT/'src/pbit/triage_policy.py',ROOT/'src/pbit/augmentation.py',ROOT/'src/pbit/modeling.py']
    inputs=[FROZEN/n for n in ['train_primary.csv','long_primary.csv','inner_assignment.csv','outer_assignment.csv','freeze_manifest.json']]
    inputs += [ROOT/'data/e1_full_v1/completion_manifest.json',ROOT/'data/e10_v1/summary.csv',ROOT/'data/e10_v1/completion.json']
    for item in items: inputs.extend(dependencies(item))
    cfg=dict(partitions=50,new_fits=900,regions=3,min_region_length=15,threads=4,workers=4,arms=ARMS,
             targets=TARGETS,budgets=BUDGETS,physical_phase_names=False)
    record=dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(inputs))},config=serial(cfg))
    if (OUT/'freeze.json').exists():
        old=read(OUT/'freeze.json')
        for k in record:assert old[k]==record[k],k
    else:
        assert not OUT.exists()
        old=read(FROZEN/'freeze_manifest.json')
        for p in inputs[:4]: assert sha(p)==old['output_sha256'][p.relative_to(ROOT).as_posix()]
        OUT.mkdir()
        for p in sources:
            snapshot=OUT/'frozen_sources'/p.relative_to(ROOT);snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes(p.read_bytes())
        write(OUT/'freeze.json',dict(**record,frozen_utc=datetime.now(timezone.utc).isoformat(),
              environment={n:importlib.metadata.version(n) for n in ('numpy','pandas','scikit-learn','lightgbm')}))
    return items


def run_partition(item):
    seed,outer=item['seed'],item['outer']
    folder=OUT/f'seed_{seed}_outer_{outer}'
    if (folder/'complete.json').exists():
        result=read(folder/'complete.json')
        for name,h in result['output_sha256'].items():assert sha(folder/name)==h,name
        return result
    assert not folder.exists(),'Partial partition needs an explicit recorded amendment.'
    folder.mkdir()
    start=time.monotonic()
    dependencies(item)
    origin=ROOT/item['output']
    population=pd.read_csv(FROZEN/'train_primary.csv')
    long=pd.read_csv(FROZEN/'long_primary.csv')
    long=long[long.Experiment_Index.isin(population.Experiment_Index)].copy()
    matrix=long.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement')
    assert matrix.shape==(1195,123) and not matrix.isna().any().any()
    old_inner=np.load(origin/'untuned_inner_probability.npy')
    old_outer=np.load(origin/'untuned_external_probability.npy')
    inner_meta=[];inner_scores={arm:[] for arm in ARMS}
    outer_meta=None;outer_scores={}
    boundary_records=[];fit_seconds=[];feature_names={}
    for index,role in enumerate(['inner0','inner1','inner2','inner3','inner4','outer']):
        cached=load_cache(origin/f'cache/{role}.npz')
        training=pd.read_csv(origin/f'cache/{role}_training_metadata.csv')
        evaluation=pd.read_csv(origin/f'cache/{role}_evaluation_metadata.csv')
        real=training[training.Experiment_Index.isin(population.Experiment_Index)].reset_index(drop=True)
        assert np.array_equal(training.iloc[:len(real)].Experiment_Index,real.Experiment_Index)
        assert set(training.Serial_Number).isdisjoint(evaluation.Serial_Number)
        assert np.array_equal(cached['y'],training.Fault_Mode) and np.array_equal(cached['y_eval'],evaluation.Fault_Mode)
        synthetic=augment_training_set(long,set(real.Experiment_Index),random_state=seed+index if role!='outer' else seed)
        syn_meta=synthetic[META].drop_duplicates('Experiment_Index').reset_index(drop=True)
        pd.testing.assert_frame_equal(syn_meta,training.iloc[len(real):].reset_index(drop=True),check_dtype=False)
        syn_matrix=synthetic.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement')
        real_x=matrix.loc[real.Experiment_Index].to_numpy(float)
        expanded=np.vstack([real_x,syn_matrix.loc[syn_meta.Experiment_Index].to_numpy(float)])
        eval_x=matrix.loc[evaluation.Experiment_Index].to_numpy(float)
        boundaries,objective=learn_regions(real_x,real.Serial_Number.to_numpy())
        boundary_records.append(dict(role=role,boundaries=boundaries,objective=objective,
            training_units=sorted(real.Serial_Number.unique()),evaluation_units=sorted(evaluation.Serial_Number.unique()),
            real_ids=real.Experiment_Index.tolist(),synthetic_cycles=len(syn_meta),evaluation_ids=evaluation.Experiment_Index.tolist()))
        uniform,names=region_features(expanded,(41,82));uniform_eval,_=region_features(eval_x,(41,82))
        empirical,_=region_features(expanded,boundaries);empirical_eval,_=region_features(eval_x,boundaries)
        representations={'M1_uniform':(uniform,uniform_eval),'M2_empirical':(empirical,empirical_eval)}
        for arm in ('M1_uniform','M2_empirical'):
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            train_x,test_x=representations[arm]
            representations[arm]=(imputer.fit_transform(train_x),imputer.transform(test_x))
            feature_names[arm]=names
        representations['M3_combined']=(np.c_[cached['X'],representations['M2_empirical'][0]],
                                          np.c_[cached['X_eval'],representations['M2_empirical'][1]])
        feature_names['M3_combined']=['historical_'+str(i) for i in range(246)]+names
        scores={}
        if role!='outer':
            offset=sum(len(m) for m in inner_meta)
            scores['M0_F246']=old_inner[offset:offset+len(evaluation)]
        else:scores['M0_F246']=old_outer
        for arm,(train_x,test_x) in representations.items():
            t=time.monotonic()
            scores[arm]=fit_probability(dict(X=train_x,y=cached['y'],X_eval=test_x),base_params(dict(seed=seed,threads=4)))
            fit_seconds.append(dict(role=role,arm=arm,seconds=time.monotonic()-t))
        if role!='outer':
            inner_meta.append(evaluation.assign(inner_fold=index))
            for arm in ARMS:inner_scores[arm].append(scores[arm])
        else:
            outer_meta=evaluation
            outer_scores=scores
    inner_meta=pd.concat(inner_meta,ignore_index=True)
    assert sum(len(x) for x in inner_scores['M0_F246'])==len(old_inner)
    assert set(inner_meta.Serial_Number).isdisjoint(outer_meta.Serial_Number)
    inner_meta.to_csv(folder/'inner_metadata.csv',index=False)
    outer_meta.to_csv(folder/'outer_metadata.csv',index=False)
    selected=[];predictions=[]
    for arm in ARMS:
        inside=np.concatenate(inner_scores[arm]);outside=outer_scores[arm]
        assert np.allclose(inside.sum(axis=1),1) and np.allclose(outside.sum(axis=1),1)
        np.save(folder/f'{arm}_inner_probability.npy',inside)
        np.save(folder/f'{arm}_outer_probability.npy',outside)
        values=curve(inner_meta.Fault_Mode.to_numpy(),inside)
        for target,budget in SCENARIOS:
            selection=select_curve(values,target,budget)
            selected.append(dict(arm=arm,**selection))
            frame=outer_meta[['Experiment_Index','Serial_Number','Fault_Mode']].copy()
            frame['prediction']=predict(outside,selection['threshold'])
            frame['arm']=arm;frame['seed']=seed;frame['outer']=outer
            frame['target']=target;frame['budget']=budget
            predictions.append(frame)
    pd.concat(predictions,ignore_index=True).to_csv(folder/'external_predictions.csv',index=False)
    write(folder/'inner_selections.json',selected)
    write(folder/'boundaries.json',boundary_records)
    write(folder/'feature_names.json',feature_names)
    pd.DataFrame(fit_seconds).to_csv(folder/'fit_times.csv',index=False)
    result=dict(seed=seed,outer=outer,new_fits=len(fit_seconds),seconds=time.monotonic()-start,
                external_performance_calculated=False,
                output_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    write(folder/'complete.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full'],required=True)
    args=parser.parse_args();items=freeze()
    if args.stage=='pilot':
        item=next(i for i in items if i['seed']==42 and i['outer']==0)
        result=run_partition(item)
        write(OUT/'pilot.json',dict(**result,projected_full_hours=result['seconds']*50/4/3600))
        print(json.dumps({k:v for k,v in read(OUT/'pilot.json').items() if k!='output_sha256'},indent=2),flush=True)
        return
    assert (OUT/'pilot.json').exists()
    if not (OUT/'batch_start.json').exists():write(OUT/'batch_start.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),new_fit_budget=900))
    done=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run_partition,item) for item in items]
        for future in as_completed(jobs):
            result=future.result();done.append(result)
            write(OUT/'progress.json',dict(completed=len(done),total=50,updated_utc=datetime.now(timezone.utc).isoformat()))
            print(f'E14.e partitions completed: {len(done)}/50',flush=True)
    write(OUT/'completion.json',dict(completed=done,new_fits=sum(x['new_fits'] for x in done),
          finished_utc=datetime.now(timezone.utc).isoformat(),status='complete'))
    from e14_summarize import summarize
    summarize()


if __name__=='__main__':main()
