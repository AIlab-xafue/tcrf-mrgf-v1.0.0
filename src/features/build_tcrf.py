"""Validate frozen risk factors using the formal training statistics rule."""
from pathlib import Path
import numpy as np
import pandas as pd

def validate_tcrf():
    root = Path(__file__).resolve().parents[2]
    node = pd.read_csv(root/'data/model_input/node_features.csv',dtype={'firm_id':str},float_precision='round_trip')
    breadth = pd.read_csv(root/'data/technology/ipc_breadth.csv',dtype={'firm_id':str},float_precision='round_trip')
    edges = pd.read_csv(root/'data/graph/joint_edges.csv',dtype={'firm_a':str,'firm_b':str})
    active = set(zip(edges.firm_a,edges.feature_quarter))|set(zip(edges.firm_b,edges.feature_quarter))
    state = node[['firm_id','feature_quarter','split','net_profit_margin']].copy()
    state['isolated_flag'] = [int(key not in active) for key in state[['firm_id','feature_quarter']].itertuples(index=False,name=None)]
    state['ipc_breadth'] = breadth.ipc_breadth
    fit = state.split.eq('Train')
    for variable,z in [('net_profit_margin','z_npm'),('isolated_flag','z_iso'),('ipc_breadth','z_breadth')]:
        values = state.loc[fit,variable].astype(float)
        state[z] = (state[variable]-float(values.mean()))/float(values.std(ddof=1))
    pressure = np.maximum(-state.z_npm,0)
    np.testing.assert_allclose(node.isolation_risk,pressure*np.maximum(state.z_iso,0),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(node.breadth_risk,pressure*np.maximum(-state.z_breadth,0),rtol=1e-14,atol=1e-14)
    print('TCRF_VALIDATION = PASS')

if __name__=='__main__':
    validate_tcrf()
