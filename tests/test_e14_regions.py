from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e14_regions import learn_regions,region_features


def test_recovers_numeric_changes_without_labels():
    x=np.r_[np.zeros(30),np.full(60,2.),np.full(33,-1.)]
    bounds,cost=learn_regions(np.vstack([x,x+100]),['a','b'])
    assert bounds==(30,90) and abs(cost)<1e-10


def test_unit_weight_invariance_and_flat_cycle():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(3,123));x[2]=7
    bounds,cost=learn_regions(x,['a','b','c'])
    b2,c2=learn_regions(np.vstack([x[0],x[0],x[1],x[2]]),['a','a','b','c'])
    assert bounds==b2 and np.isclose(cost,c2)
    flat,c=learn_regions(np.ones((2,123)),['a','b'])
    assert flat==(15,30) and c==0


def test_descriptors_keep_absolute_level_and_dimensions():
    x=np.tile(np.arange(123,dtype=float),(2,1));x[1]+=10
    features,names=region_features(x,(41,82))
    assert features.shape==(2,48) and len(set(names))==48
    assert np.isclose(features[1,names.index('r1_mean')]-features[0,names.index('r1_mean')],10)
    assert np.isclose(features[0,names.index('r2_slope')],1)
    assert np.isclose(features[0,names.index('r1_delta')],40)
