"""Predefined ablations of numerical regions; no physical phase assignments."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import importlib.metadata
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pbit.augmentation import augment_training_set
from pbit.triage_policy import curve,select_curve,predict,TARGETS,BUDGETS
from nested_cv_e1 import base_params,fit_probability,load_cache
from e14_regions import region_features
from e14_run import read,write,sha,dependencies

OUT=ROOT/'data/e16_empirical_v1'
E14=ROOT/'data/e14_empirical_v2'
FROZEN=ROOT/'data/ress_revision/freeze_v1'
META=['Experiment_Index','Serial_Number','Serial_Run_Index','Fault_Mode','Severity_Index']
NEW_ARMS=['only_R1','only_R2','only_R3','without_R1','without_R2','without_R3','limits_minus1','limits_plus1']
ARMS=['M2_reference']+NEW_ARMS


def feature_mask(names,arm):
    assert arm.startswith(('only_R','without_R'))
    target=int(arm[-1])
    refs=[{int(x) for x in re.findall(r'(?:^|_)r([123])(?=_|$)',name)} for name in names]
    return np.array([r <= {target} if arm.startswith('only') else target not in r for r in refs])


def freeze():
    items=read(ROOT/'data/e1_full_v1/completion_manifest.json')['completed']
    sources=[Path(__file__),ROOT/'scripts/e16_summarize.py',ROOT/'tests/test_e16_regions.py',
             ROOT/'docs/protocolos/E16_empirico.md',ROOT/'scripts/e14_regions.py',ROOT/'scripts/e14_run.py',
             ROOT/'scripts/nested_cv_e1.py',ROOT/'scripts/nested_cv_controlled.py',ROOT/'src/pbit/augmentation.py',
             ROOT/'src/pbit/modeling.py',ROOT/'src/pbit/triage_policy.py']
    inputs=[FROZEN/'train_primary.csv',FROZEN/'long_primary.csv',FROZEN/'freeze_manifest.json',
        ROOT/'data/e1_full_v1/completion_manifest.json',E14/'freeze.json',E14/'completion.json',E14/'verification.json']
    check=read(E14/'verification.json');assert check['passed'] and sha(E14/'completion.json')==check['completion_sha256']
    for item in items:
        inputs.extend(dependencies(item))
        folder=E14/f"seed_{item['seed']}_outer_{item['outer']}"
        complete=read(folder/'complete.json')
        for name in ['M2_empirical_inner_probability.npy','M2_empirical_outer_probability.npy','boundaries.json','inner_metadata.csv','outer_metadata.csv']:
            assert sha(folder/name)==complete['output_sha256'][name]
            inputs.append(folder/name)
        inputs.append(folder/'complete.json')
    record=dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(inputs))},
        config=dict(arms=ARMS,new_fits=2400,threads=4,workers=4,population_cycles=1195,physical_phase_names=False))
    if (OUT/'freeze.json').exists():
        old=read(OUT/'freeze.json')
        for key in record:assert old[key]==record[key],key
    else:
        assert not OUT.exists();OUT.mkdir()
        for p in sources:
            dest=OUT/'frozen_sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        write(OUT/'freeze.json',dict(**record,utc=datetime.now(timezone.utc).isoformat(),
            environment={n:importlib.metadata.version(n) for n in ['numpy','pandas','lightgbm','scikit-learn']}))
    return items


def run_partition(item):
    seed,outer=item['seed'],item['outer'];folder=OUT/f'seed_{seed}_outer_{outer}'
    if (folder/'complete.json').exists():
        result=read(folder/'complete.json')
        for name,h in result['output_sha256'].items():assert sha(folder/name)==h
        return result
    assert not folder.exists(),'Partial partition requires explicit amendment.'
    folder.mkdir();started=time.monotonic();dependencies(item)
    origin=ROOT/item['output'];prior=E14/f'seed_{seed}_outer_{outer}'
    population=pd.read_csv(FROZEN/'train_primary.csv');long=pd.read_csv(FROZEN/'long_primary.csv')
    long=long[long.Experiment_Index.isin(population.Experiment_Index)]
    signal=long.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement')
    boundaries={b['role']:b for b in read(prior/'boundaries.json')}
    inside_ref=np.load(prior/'M2_empirical_inner_probability.npy');outside_ref=np.load(prior/'M2_empirical_outer_probability.npy')
    inner_meta=[];inner_scores={arm:[] for arm in ARMS};outer_scores={};fit_times=[];specs=[]
    for index,role in enumerate(['inner0','inner1','inner2','inner3','inner4','outer']):
        cached=load_cache(origin/f'cache/{role}.npz')
        training=pd.read_csv(origin/f'cache/{role}_training_metadata.csv');evaluation=pd.read_csv(origin/f'cache/{role}_evaluation_metadata.csv')
        real=training[training.Experiment_Index.isin(population.Experiment_Index)].reset_index(drop=True)
        b=boundaries[role];assert real.Experiment_Index.tolist()==b['real_ids']
        assert evaluation.Experiment_Index.tolist()==b['evaluation_ids']
        augmented=augment_training_set(long,set(real.Experiment_Index),random_state=seed+index if role!='outer' else seed)
        syn_meta=augmented[META].drop_duplicates('Experiment_Index').reset_index(drop=True)
        pd.testing.assert_frame_equal(syn_meta,training.iloc[len(real):].reset_index(drop=True),check_dtype=False)
        syn=augmented.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement').loc[syn_meta.Experiment_Index].to_numpy(float)
        x=np.vstack([signal.loc[real.Experiment_Index].to_numpy(float),syn]);xe=signal.loc[evaluation.Experiment_Index].to_numpy(float)
        base,names=region_features(x,b['boundaries']);base_eval,_=region_features(xe,b['boundaries'])
        scores={}
        if role=='outer':scores['M2_reference']=outside_ref
        else:
            offset=sum(len(g) for g in inner_meta);scores['M2_reference']=inside_ref[offset:offset+len(evaluation)]
        for arm in NEW_ARMS:
            if arm.startswith(('only','without')):
                mask=feature_mask(names,arm);train_x=base[:,mask];eval_x=base_eval[:,mask];used_bounds=b['boundaries']
                expected=14 if arm.startswith('only') else 30;assert mask.sum()==expected
                used_names=[name for name,keep in zip(names,mask) if keep]
            else:
                shift=-1 if arm=='limits_minus1' else 1;used_bounds=[int(v)+shift for v in b['boundaries']]
                train_x,_=region_features(x,used_bounds);eval_x,_=region_features(xe,used_bounds)
                used_names=names
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            cache=dict(X=imp.fit_transform(train_x),X_eval=imp.transform(eval_x),y=cached['y'])
            start=time.monotonic();scores[arm]=fit_probability(cache,base_params(dict(seed=seed,threads=4)))
            fit_times.append(dict(role=role,arm=arm,seconds=time.monotonic()-start))
            specs.append(dict(role=role,arm=arm,n_features=train_x.shape[1],boundaries=used_bounds,features=used_names))
        if role=='outer':outer_meta=evaluation;outer_scores=scores
        else:
            inner_meta.append(evaluation.assign(inner_fold=index))
            for arm in ARMS:inner_scores[arm].append(scores[arm])
    inner_meta=pd.concat(inner_meta,ignore_index=True)
    inner_meta.to_csv(folder/'inner_metadata.csv',index=False);outer_meta.to_csv(folder/'outer_metadata.csv',index=False)
    predictions=[];selections=[]
    for arm in ARMS:
        inside=np.concatenate(inner_scores[arm]);outside=outer_scores[arm]
        np.save(folder/f'{arm}_inner_probability.npy',inside);np.save(folder/f'{arm}_outer_probability.npy',outside)
        values=curve(inner_meta.Fault_Mode.to_numpy(),inside)
        for target in TARGETS:
            for budget in BUDGETS:
                selected=select_curve(values,target,budget);selections.append(dict(arm=arm,**selected))
                frame=outer_meta[['Experiment_Index','Serial_Number','Fault_Mode']].copy()
                frame['prediction']=predict(outside,selected['threshold']);frame['arm']=arm
                frame['seed']=seed;frame['outer']=outer;frame['target']=target;frame['budget']=budget;predictions.append(frame)
    pd.concat(predictions,ignore_index=True).to_csv(folder/'external_predictions.csv',index=False)
    write(folder/'inner_selections.json',selections);write(folder/'representation_specs.json',specs)
    pd.DataFrame(fit_times).to_csv(folder/'fit_times.csv',index=False)
    result=dict(seed=seed,outer=outer,new_fits=len(fit_times),seconds=time.monotonic()-started,external_performance_calculated=False,
        output_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    write(folder/'complete.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full'],required=True);args=parser.parse_args();items=freeze()
    if args.stage=='pilot':
        result=run_partition(next(i for i in items if i['seed']==42 and i['outer']==0));write(OUT/'pilot.json',result)
        print({k:v for k,v in result.items() if k!='output_sha256'},flush=True);return
    assert (OUT/'pilot.json').exists()
    if not (OUT/'batch_start.json').exists():write(OUT/'batch_start.json',dict(utc=datetime.now(timezone.utc).isoformat(),new_fit_budget=2400))
    done=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run_partition,item) for item in items]):
            done.append(future.result());write(OUT/'progress.json',dict(completed=len(done),total=50,utc=datetime.now(timezone.utc).isoformat()))
            print(f'E16.e partitions completed: {len(done)}/50',flush=True)
    write(OUT/'completion.json',dict(completed=done,new_fits=sum(x['new_fits'] for x in done),finished_utc=datetime.now(timezone.utc).isoformat()))
    from e16_summarize import summarize
    summarize()


if __name__=='__main__':main()
