"""Independent explicit-decision frontiers for all nine scenario selections."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checked=0;source_hashes={};counts={}
    for name in ['e14_empirical_v2','e15_global_v1','e16_empirical_v1']:
        folder=ROOT/'data'/name
        complete=json.loads((folder/'completion.json').read_text())
        source_hashes[(folder/'completion.json').relative_to(ROOT).as_posix()]=sha(folder/'completion.json')
        stage_feasible=0
        for item in complete['completed']:
            part=folder/f"seed_{item['seed']}_outer_{item['outer']}"
            meta=pd.read_csv(part/'inner_metadata.csv');y=meta.Fault_Mode.to_numpy()
            stored=json.loads((part/'inner_selections.json').read_text())
            for arm in sorted({s['arm'] for s in stored}):
                path=part/f'{arm}_inner_probability.npy';probability=np.load(path)
                source_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
                q=probability[:,2:].sum(axis=1)
                thresholds=np.r_[np.inf,np.unique(q)[::-1]]
                if thresholds[-1]>0:thresholds=np.r_[thresholds,0.]
                alarms=q[None,:]>=thresholds[:,None]
                normal=np.isin(y,['N01','N02'])
                defined=normal.any() and all((y==f'F0{i}').any() for i in range(1,6))
                if defined:
                    recalls=np.column_stack([alarms[:,y==f'F0{i}'].mean(axis=1) for i in range(1,6)])
                    fa0=alarms[:,normal].mean(axis=1);minimum=recalls.min(axis=1);mean=recalls.mean(axis=1)
                for selection in [s for s in stored if s['arm']==arm]:
                    if not defined:
                        expected=np.inf;feasible=False
                    else:
                        allowed=np.flatnonzero(fa0<=selection['budget']+1e-12)
                        eligible=allowed[minimum[allowed]>=selection['target']-1e-12]
                        feasible=bool(len(eligible))
                        if feasible:
                            best=min(eligible,key=lambda k:(fa0[k],-minimum[k],-mean[k],-thresholds[k]))
                        else:
                            best=min(allowed,key=lambda k:(-minimum[k],fa0[k],-mean[k],-thresholds[k]))
                        expected=thresholds[best]
                    actual=np.inf if selection['threshold']=='infinity' else selection['threshold']
                    assert expected==actual,(name,item['seed'],item['outer'],arm,selection['target'],selection['budget'])
                    assert feasible==selection['feasible']
                    checked+=1;stage_feasible+=int(feasible)
            for file in ['inner_metadata.csv','inner_selections.json']:
                p=part/file;source_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
        counts[name]=stage_feasible
    result=dict(passed=True,independent_scenario_selections=checked,internally_feasible_by_stage=counts,
        method='All distinct score thresholds evaluated by explicit boolean decisions; no producer policy function imported.',
        source_sha256=source_hashes,checker_sha256=sha(Path(__file__)))
    path=ROOT/'data/e13_e16_editorial/admissibility_checks.json'
    assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
