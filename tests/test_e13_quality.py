import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e13_quality import signal_quality,parse_source_values,measurement_tokens


def test_original_separator_not_a_measurement():
    rows=[['F01'],['1'],['SN 001'],['']]+[[str(i)] for i in range(1,124)]
    assert measurement_tokens(rows,0)==[str(i) for i in range(1,124)]


def test_spreadsheet_error_is_missing_not_zero():
    values,tokens=parse_source_values(['1.2','','#VALUE!','0'])
    assert values[0]==1.2 and np.isnan(values[1:3]).all() and values[3]==0
    assert tokens==['#VALUE!']


def test_missing_interrupts_plateau():
    x=np.ones(123);x[::5]=np.nan
    result=signal_quality(np.arange(1,124),x)
    assert result['longest_constant_run']==4
    assert result['n_missing_positions']==25
    assert not result['plateau_flag']


def test_flat_signal_not_instrumental_invalid():
    result=signal_quality(np.arange(1,124),np.zeros(123))
    assert result['near_constant'] and result['longest_zero_run']==123
    assert result['quality_category']=='suspect'
    assert not result['spike_rule_defined']


def test_spike_and_no_metadata_dependency():
    rng=np.random.default_rng(51)
    x=np.cumsum(rng.normal(0,.01,123));x[60]+=10
    result=signal_quality(np.arange(1,124),x)
    assert result['isolated_spike_flag'] and result['isolated_spike_count']>=1
    assert result==signal_quality(np.arange(1,124)[::-1],x[::-1])


def test_structure_and_coverage():
    x=np.arange(123,dtype=float);x[:70]=np.nan
    result=signal_quality(np.arange(1,124),x)
    assert result['quality_category']=='insufficient_coverage'
    indices=np.arange(1,124);indices[5]=4
    result=signal_quality(indices,np.arange(123,dtype=float))
    assert result['structural_problem'] and result['duplicate_indices']==1
    assert result['quality_category']=='structural_problem'
