"""Label-free region learning and absolute-signal descriptors."""
import numpy as np


DESCRIPTORS=('mean','median','std','q25','q75','min','max','range','rms','slope','delta',
             'mean_abs_diff','max_abs_diff','plateau_fraction')


def learn_regions(signals, units, min_length=15):
    signals=np.asarray(signals,dtype=float)
    units=np.asarray(units)
    assert signals.ndim==2 and signals.shape[1]==123 and np.isfinite(signals).all()
    assert len(signals)==len(units) and len(signals)>0
    _,inverse,counts=np.unique(units,return_inverse=True,return_counts=True)
    weights=1/counts[inverse]/len(counts)
    std=signals.std(axis=1,keepdims=True)
    z=np.divide(signals-signals.mean(axis=1,keepdims=True),std,
                out=np.zeros_like(signals),where=std>=1e-6)
    prefix=np.c_[np.zeros(len(z)),np.cumsum(z,axis=1)]
    square=np.c_[np.zeros(len(z)),np.cumsum(z*z,axis=1)]
    n=signals.shape[1]
    costs=np.full((n+1,n+1),np.inf)
    for start in range(n-min_length+1):
        ends=np.arange(start+min_length,n+1)
        sums=prefix[:,ends]-prefix[:,start,None]
        squares=square[:,ends]-square[:,start,None]
        costs[start,ends]=weights @ (squares-sums*sums/(ends-start))
    best=None
    for first in range(min_length,n-2*min_length+1):
        for second in range(first+min_length,n-min_length+1):
            candidate=(float(costs[0,first]+costs[first,second]+costs[second,n]),first,second)
            if best is None or candidate<best:
                best=candidate
    return (best[1],best[2]),best[0]


def region_features(signals,boundaries):
    signals=np.asarray(signals,dtype=float)
    first,second=boundaries
    assert 0<first<second<signals.shape[1]
    assert np.isfinite(signals).all()
    columns=[];names=[];means=[];stds=[]
    for region,(start,end) in enumerate(zip((0,first,second),(first,second,signals.shape[1])),1):
        x=signals[:,start:end]
        mean=x.mean(axis=1);std=x.std(axis=1);d=np.abs(np.diff(x,axis=1))
        t=np.arange(x.shape[1],dtype=float);t-=t.mean()
        values=[mean,np.median(x,axis=1),std,np.quantile(x,.25,axis=1),np.quantile(x,.75,axis=1),
                x.min(axis=1),x.max(axis=1),np.ptp(x,axis=1),np.sqrt((x*x).mean(axis=1)),
                x@t/(t@t),x[:,-1]-x[:,0],d.mean(axis=1),d.max(axis=1),(d<=1e-9).mean(axis=1)]
        columns.extend(values);names.extend(f'r{region}_{name}' for name in DESCRIPTORS)
        means.append(mean);stds.append(std)
    for i,j in ((0,1),(0,2),(1,2)):
        columns.extend([means[i]-means[j],stds[i]-stds[j]])
        names.extend([f'r{i+1}_r{j+1}_mean_difference',f'r{i+1}_r{j+1}_std_difference'])
    return np.column_stack(columns),names
