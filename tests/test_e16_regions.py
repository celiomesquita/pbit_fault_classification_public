from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e14_regions import region_features
from e16_run import feature_mask


def test_ablation_removes_all_cross_region_dependencies():
    _,names=region_features(np.ones((1,123)),(15,59))
    for i in [1,2,3]:
        only=feature_mask(names,f'only_R{i}');without=feature_mask(names,f'without_R{i}')
        assert only.sum()==14 and without.sum()==30
        assert not (only&without).any()
    assert feature_mask(names,'only_R2')[names.index('r2_mean')]
    assert not feature_mask(names,'without_R2')[names.index('r1_r2_mean_difference')]
    assert feature_mask(names,'without_R2')[names.index('r1_r3_mean_difference')]
