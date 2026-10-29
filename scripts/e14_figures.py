"""Presentation figures for the completed empirical-region experiment."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/e14_empirical_v2'
ARMS=['M0_F246','M1_uniform','M2_empirical','M3_combined']
LABELS=['F246','Uniform regions','Empirical regions','F246 + empirical']


def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.png',dpi=170)
    fig.savefig(OUT/f'{name}.pdf')
    plt.close(fig)


def main():
    summary=pd.read_csv(OUT/'summary.csv')
    s=summary[np.isclose(summary.target,.90)&np.isclose(summary.budget,.40)].set_index('arm').loc[ARMS]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[2,1]})
    classes=['F01','F02','F03','F04','F05']
    x=np.arange(5);width=.19
    for i,(arm,label) in enumerate(zip(ARMS,LABELS)):
        axes[0].bar(x+(i-1.5)*width,[s.loc[arm,'detect_'+c] for c in classes],width=width,label=label)
    axes[0].axhline(.90,color='black',linestyle='--',linewidth=1,label='Recall target 0.90')
    axes[0].set(xticks=x,xticklabels=classes,ylabel='Fault-detection recall',ylim=(0,1.04))
    axes[0].legend(fontsize=8,ncol=2,loc='lower right')
    axes[1].bar(np.arange(4),s.FA0,color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:4])
    axes[1].axhline(.40,color='black',linestyle='--',linewidth=1)
    axes[1].set(xticks=np.arange(4),xticklabels=['F246','Uniform','Empirical','Combined'],ylabel='False-alarm rate',ylim=(0,1.04))
    axes[1].tick_params(axis='x',labelrotation=25)
    fig.suptitle('Primary scenario — means over the same ten seeds')
    save(fig,'primary_performance')
    b=pd.read_csv(OUT/'region_boundaries.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,col,title,ref in zip(axes,['first','second'],['First numeric boundary','Second numeric boundary'],[41,82]):
        ax.hist(b[col],bins=np.arange(14.5,109.5),alpha=.7,label='Learned boundaries')
        ax.axvline(ref,color='black',linestyle='--',label='Uniform control')
        ax.set(xlabel='Last measurement index of region',ylabel='Training partitions',title=title)
        ax.legend(fontsize=8)
    save(fig,'region_boundaries')
    u=pd.read_csv(OUT/'primary_unit_class_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,cls in zip(axes,['F02','F05']):
        part=u[u.Fault_Mode==cls].pivot(index='Serial_Number',columns='arm',values='alarm_rate').sort_index()
        y=part.M2_empirical-part.M1_uniform
        ax.bar(np.arange(len(y)),y,color=np.where(y>=0,'#28784e','#aa4d3a'))
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(xticks=np.arange(len(y)),xticklabels=[x.replace('SN ','') for x in y.index],
               xlabel='Serial number',ylabel='Detection recall difference',title=cls,ylim=(-1,1))
        ax.tick_params(axis='x',labelrotation=60)
    fig.suptitle('Empirical minus uniform regions — primary scenario')
    save(fig,'unit_differences')


if __name__=='__main__':main()
