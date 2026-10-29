"""Fixed global-quality training contrast; evaluation always retains Q0 cases."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import importlib.metadata
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
from pbit.features import extract_features
from pbit.triage_policy import curve,select_curve,predict,TARGETS,BUDGETS
from nested_cv_e1 import base_params,fit_probability
from e14_regions import learn_regions,region_features
from e14_run import read,write,sha

OUT=ROOT/'data/e15_global_v1'
FROZEN=ROOT/'data/ress_revision/freeze_v1'
META=['Experiment_Index','Serial_Number','Serial_Run_Index','Fault_Mode','Severity_Index']
ARMS=['Q0_F246','Q0_empirical','Q1_F246','Q1_empirical']


def select_training(frame,flags,policy):
    assert policy in ['Q0','Q1']
    return frame.copy() if policy=='Q0' else frame.loc[~frame.Experiment_Index.map(flags)].copy()


def population_data():
    audit=pd.read_csv(ROOT/'data/e13_quality_v3/cycle_quality.csv')
    candidate=audit[audit.processable_candidate].sort_values('Experiment_Index')
    canonical=pd.read_csv(ROOT/'data/PBIT_features_r1.csv').set_index('Experiment_Index')
    population=canonical.loc[candidate.Experiment_Index].reset_index()
    assert len(population)==1196 and population.Serial_Number.nunique()==54
    assert population[META].equals(candidate[META].reset_index(drop=True))
    flags=candidate.set_index('Experiment_Index').any_flag
    assert int(flags.sum())==65
    long=pd.read_csv(ROOT/'data/PBIT_atuador_DESCARAC_r1_long_clean.csv')
    long=long[long.Experiment_Index.isin(population.Experiment_Index)].copy()
    signal=long.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement')
    assert signal.shape==(1196,123) and not signal.isna().any().any()
    return population,flags,long,signal


def freeze():
    sources=[Path(__file__),ROOT/'scripts/e15_summarize.py',ROOT/'tests/test_e15_quality_policy.py',
             ROOT/'docs/protocolos/E15_global.md',ROOT/'scripts/e14_regions.py',ROOT/'scripts/e14_run.py',
             ROOT/'scripts/nested_cv_e1.py',ROOT/'scripts/nested_cv_controlled.py',
             ROOT/'src/pbit/features.py',ROOT/'src/pbit/augmentation.py',ROOT/'src/pbit/modeling.py',ROOT/'src/pbit/triage_policy.py']
    inputs=[ROOT/n for n in ['data/PBIT_features_r1.csv','data/PBIT_atuador_DESCARAC_r1_long_clean.csv',
        'data/e13_quality_v3/cycle_quality.csv','data/e13_quality_v3/completion.json','data/e13_quality_v3/verification.json',
        'data/e13_quality_v3/freeze.json','data/ress_revision/freeze_v1/train_primary.csv',
        'data/ress_revision/freeze_v1/outer_assignment.csv','data/ress_revision/freeze_v1/inner_assignment.csv',
        'data/ress_revision/freeze_v1/freeze_manifest.json']]
    old=read(ROOT/'data/e13_quality_v3/completion.json')
    assert sha(ROOT/'data/e13_quality_v3/cycle_quality.csv')==old['output_sha256']['cycle_quality.csv']
    frozen=read(FROZEN/'freeze_manifest.json')
    for name in ['train_primary.csv','outer_assignment.csv','inner_assignment.csv']:
        p=FROZEN/name;assert sha(p)==frozen['output_sha256'][p.relative_to(ROOT).as_posix()]
    record=dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
                config=dict(arms=ARMS,partitions=50,new_fits=1200,threads=4,workers=4,regions=3,min_length=15,
                            evaluation_cycles=1196,flags='frozen global numeric suspicion, not instrumental confirmation'))
    if (OUT/'freeze.json').exists():
        prior=read(OUT/'freeze.json')
        for key in record:assert record[key]==prior[key],key
    else:
        assert not OUT.exists();OUT.mkdir()
        for p in sources:
            copy=OUT/'frozen_sources'/p.relative_to(ROOT);copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes())
        write(OUT/'freeze.json',dict(**record,frozen_utc=datetime.now(timezone.utc).isoformat(),
            environment={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm']}))


def run_partition(task):
    seed,outer=task;folder=OUT/f'seed_{seed}_outer_{outer}'
    if (folder/'complete.json').exists():
        item=read(folder/'complete.json')
        for name,h in item['output_sha256'].items():assert sha(folder/name)==h,name
        return item
    assert not folder.exists(),'Partial execution requires a recorded amendment.'
    folder.mkdir();started=time.monotonic()
    population,flags,long,signal=population_data()
    feature_cols=[c for c in population if c not in META];assert len(feature_cols)==246
    outer_assignment=pd.read_csv(FROZEN/'outer_assignment.csv')
    outer_assignment=outer_assignment[outer_assignment.seed==seed]
    assert outer_assignment.groupby('Serial_Number').outer_fold.nunique().eq(1).all()
    held_units=set(outer_assignment.loc[outer_assignment.outer_fold==outer,'Serial_Number'])
    training=population[~population.Serial_Number.isin(held_units)].reset_index(drop=True)
    testing=population[population.Serial_Number.isin(held_units)].reset_index(drop=True)
    inner_assignment=pd.read_csv(FROZEN/'inner_assignment.csv')
    inner_assignment=inner_assignment[(inner_assignment.seed==seed)&(inner_assignment.outer_fold==outer)]
    assert inner_assignment.groupby('Serial_Number').inner_fold.nunique().eq(1).all()
    roles=[]
    for k in range(5):
        held=set(inner_assignment.loc[inner_assignment.inner_fold==k,'Serial_Number'])
        roles.append((f'inner{k}',training[~training.Serial_Number.isin(held)],training[training.Serial_Number.isin(held)],seed+k))
    roles.append(('outer',training,testing,seed))
    inner_meta=[];inner_scores={arm:[] for arm in ARMS};outer_scores={};descriptions=[];fit_times=[]
    for role,real_base,evaluation,aug_seed in roles:
        evaluation=evaluation.reset_index(drop=True)
        assert set(real_base.Serial_Number).isdisjoint(evaluation.Serial_Number)
        if role!='outer':inner_meta.append(evaluation[META].assign(inner_fold=int(role[-1])))
        for policy in ['Q0','Q1']:
            real=select_training(real_base,flags,policy).reset_index(drop=True)
            augmented=augment_training_set(long,set(real.Experiment_Index),random_state=aug_seed)
            syn=extract_features(augmented) if len(augmented) else real.iloc[:0].copy()
            expanded=pd.concat([real,syn],ignore_index=True)
            if len(syn):
                syn_signal=augmented.pivot(index='Experiment_Index',columns='Sample_Index',values='Measurement').loc[syn.Experiment_Index].to_numpy(float)
            else:syn_signal=np.empty((0,123))
            real_signal=signal.loc[real.Experiment_Index].to_numpy(float)
            train_signal=np.vstack([real_signal,syn_signal])
            evaluation_signal=signal.loc[evaluation.Experiment_Index].to_numpy(float)
            boundaries,objective=learn_regions(real_signal,real.Serial_Number.to_numpy())
            reg,names=region_features(train_signal,boundaries);reg_eval,_=region_features(evaluation_signal,boundaries)
            representations={'F246':(expanded[feature_cols].to_numpy(float),evaluation[feature_cols].to_numpy(float)),
                             'empirical':(reg,reg_eval)}
            descriptions.append(dict(role=role,policy=policy,boundaries=boundaries,objective=objective,
                retained_ids=real.Experiment_Index.tolist(),removed_ids=sorted(set(real_base.Experiment_Index)-set(real.Experiment_Index)),
                training_units=sorted(real.Serial_Number.unique()),evaluation_units=sorted(evaluation.Serial_Number.unique()),
                evaluation_ids=evaluation.Experiment_Index.tolist(),training_classes=real.Fault_Mode.value_counts().to_dict(),
                real_cycles=len(real),synthetic_cycles=len(syn)))
            for representation,(x,x_eval) in representations.items():
                arm=f'{policy}_{representation}';imputer=SimpleImputer(strategy='median',keep_empty_features=True)
                cache=dict(X=imputer.fit_transform(x),X_eval=imputer.transform(x_eval),y=expanded.Fault_Mode.to_numpy())
                begin=time.monotonic();prob=fit_probability(cache,base_params(dict(seed=seed,threads=4)))
                fit_times.append(dict(role=role,arm=arm,seconds=time.monotonic()-begin))
                if role=='outer':outer_scores[arm]=prob
                else:inner_scores[arm].append(prob)
    inner_meta=pd.concat(inner_meta,ignore_index=True)
    assert len(inner_meta)==len(training) and inner_meta.Experiment_Index.nunique()==len(training)
    inner_meta.to_csv(folder/'inner_metadata.csv',index=False);testing[META].to_csv(folder/'outer_metadata.csv',index=False)
    predictions=[];selections=[]
    for arm in ARMS:
        inside=np.concatenate(inner_scores[arm]);outside=outer_scores[arm]
        np.save(folder/f'{arm}_inner_probability.npy',inside);np.save(folder/f'{arm}_outer_probability.npy',outside)
        values=curve(inner_meta.Fault_Mode.to_numpy(),inside)
        for target in TARGETS:
            for budget in BUDGETS:
                selected=select_curve(values,target,budget);selections.append(dict(arm=arm,**selected))
                frame=testing[['Experiment_Index','Serial_Number','Fault_Mode']].copy()
                frame['prediction']=predict(outside,selected['threshold']);frame['arm']=arm
                frame['seed']=seed;frame['outer']=outer;frame['target']=target;frame['budget']=budget
                frame['any_flag']=frame.Experiment_Index.map(flags);predictions.append(frame)
    pd.concat(predictions,ignore_index=True).to_csv(folder/'external_predictions.csv',index=False)
    write(folder/'inner_selections.json',selections);write(folder/'training_roles.json',descriptions)
    pd.DataFrame(fit_times).to_csv(folder/'fit_times.csv',index=False)
    item=dict(seed=seed,outer=outer,new_fits=len(fit_times),seconds=time.monotonic()-started,
              external_performance_calculated=False,output_sha256={p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    write(folder/'complete.json',item);return item


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','full'],required=True);args=parser.parse_args()
    freeze()
    if args.stage=='pilot':
        item=run_partition((42,0));write(OUT/'pilot.json',item)
        print({k:v for k,v in item.items() if k!='output_sha256'},flush=True);return
    assert (OUT/'pilot.json').exists()
    if not (OUT/'batch_start.json').exists():write(OUT/'batch_start.json',dict(utc=datetime.now(timezone.utc).isoformat(),new_fit_budget=1200))
    seeds=[42,0,7,13,21,1,2,3,4,5];tasks=[(seed,outer) for seed in seeds for outer in range(5)];done=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run_partition,t) for t in tasks]):
            done.append(future.result());write(OUT/'progress.json',dict(completed=len(done),total=50,utc=datetime.now(timezone.utc).isoformat()))
            print(f'E15.g partitions completed: {len(done)}/50',flush=True)
    write(OUT/'completion.json',dict(completed=done,new_fits=sum(x['new_fits'] for x in done),finished_utc=datetime.now(timezone.utc).isoformat()))
    from e15_summarize import summarize
    summarize()


if __name__=='__main__':main()
