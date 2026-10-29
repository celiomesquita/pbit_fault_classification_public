from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e15_run import select_training


def test_training_filter_independent_of_class_and_preserves_source():
    frame=pd.DataFrame(dict(Experiment_Index=[1,2,3],Serial_Number=['a','a','b'],Fault_Mode=['F06','N02','F02']))
    flags=pd.Series([True,False,False],index=[1,2,3])
    assert select_training(frame,flags,'Q0').Experiment_Index.tolist()==[1,2,3]
    assert select_training(frame,flags,'Q1').Experiment_Index.tolist()==[2,3]
    changed=frame.copy();changed['Fault_Mode']=['N01','F06','N02']
    assert select_training(changed,flags,'Q1').Experiment_Index.tolist()==[2,3]
    assert frame.Experiment_Index.tolist()==[1,2,3]
