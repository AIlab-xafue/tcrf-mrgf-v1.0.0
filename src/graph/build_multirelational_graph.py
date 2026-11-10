"""Validate the finalized quarter-aligned relation assets."""
from pathlib import Path
import pandas as pd

def validate_graph():
    root=Path(__file__).resolve().parents[2]
    edges=pd.read_csv(root/'data/graph/joint_edges.csv',dtype={'firm_a':str,'firm_b':str})
    labels=pd.read_csv(root/'data/metadata/labels.csv',dtype={'firm_id':str})
    keys=set(zip(labels.feature_quarter,labels.firm_id))
    assert len(edges)==1323
    assert edges.relation_type.value_counts().to_dict()=={'IPC':882,'Patent':441}
    assert not edges.duplicated(['feature_quarter','firm_a','firm_b','relation_type']).any()
    for row in edges.itertuples():
        assert (row.feature_quarter,row.firm_a) in keys and (row.feature_quarter,row.firm_b) in keys
    print('GRAPH_VALIDATION = PASS')

if __name__=='__main__':validate_graph()
