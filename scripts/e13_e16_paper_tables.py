"""Generate manuscript tables only after all three stage verifications pass."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cases=[('e14_empirical_v2','tab_e14_regions.tex',[
        ('M0_F246','F246 (M0)'),('M1_uniform','Uniform regions (M1)'),
        ('M2_empirical','Empirical regions (M2)'),('M3_combined','F246 + empirical (M3)')]),
        ('e15_global_v1','tab_e15_filtering.tex',[
        ('Q0_F246','Unfiltered F246 (Q0)'),('Q1_F246','Filtered F246 (Q1)'),
        ('Q0_empirical','Unfiltered empirical (Q0)'),('Q1_empirical','Filtered empirical (Q1)')]),
        ('e16_empirical_v1','tab_e16_ablation.tex',[
        ('M2_reference','All regions (reference)'),('only_R1','R1 only'),('only_R2','R2 only'),('only_R3','R3 only'),
        ('without_R1','Without R1'),('without_R2','Without R2'),('without_R3','Without R3'),
        ('limits_minus1','Both limits $-1$'),('limits_plus1','Both limits $+1$')])]
    records=[]
    for folder,name,arms in cases:
        p=ROOT/'data'/folder
        verified=json.loads((p/'verification.json').read_text());assert verified['passed']
        data=pd.read_csv(p/'summary.csv');data=data[np.isclose(data.target,.90)&np.isclose(data.budget,.40)].set_index('arm')
        lines=[f'% Source: verified data/{folder}/summary.csv; target .90; FA0 budget .40.',
            r'\begin{tabular}{lrrrrrrr}',r'\toprule',
            r'Configuration & FA0 & F01 & F02 & F03 & F04 & F05 & F06 (expl.) \\',r'\midrule']
        for arm,label in arms:
            row=data.loc[arm]
            values=[row['FA0']]+[row[f'detect_F0{i}'] for i in range(1,7)]
            lines.append(label+' & '+' & '.join(f'{value:.3f}' for value in values)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}','']
        target=ROOT/'paper/supplement/tables'/name;target.write_text('\n'.join(lines),encoding='utf-8')
        records.append(dict(summary=(p/'summary.csv').relative_to(ROOT).as_posix(),summary_sha256=sha(p/'summary.csv'),
            verification_sha256=sha(p/'verification.json'),table=target.relative_to(ROOT).as_posix(),table_sha256=sha(target)))
    (ROOT/'data/e13_e16_paper_tables.json').write_text(json.dumps(dict(generator_sha256=sha(Path(__file__)),tables=records),indent=2)+'\n',encoding='utf-8')
    print('Three tables generated from verified primary scenarios.')


if __name__=='__main__':main()
