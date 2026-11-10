import random, hashlib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def tensorhash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def metrics(y, p, t):
    pred = p >= t
    truth = y == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, 'accuracy': (tp + tn) / len(y), 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}

def batches_from(node, edges, labels, features):
    batches = {}
    metas = {}
    hashes = {}
    for split in ['Train', 'Valid', 'Test']:
        graphs = []
        frames = []
        for q in sorted(labels.loc[labels.split.eq(split), 'feature_quarter'].unique()):
            li = labels[labels.feature_quarter.eq(q)].sort_values('firm_id')
            xx = node[node.feature_quarter.eq(q)].sort_values('firm_id')
            idx = {f: i for i, f in enumerate(li.firm_id)}
            ee = []
            types = []
            for r in edges[edges.feature_quarter.eq(q)].sort_values(['relation_type', 'firm_a', 'firm_b']).itertuples(index=False):
                ee.extend([(idx[r.firm_a], idx[r.firm_b]), (idx[r.firm_b], idx[r.firm_a])])
                types.extend([0 if r.relation_type == 'Patent' else 1] * 2)
            graphs.append(Data(x=torch.tensor(xx[features].to_numpy(np.float32)), y=torch.tensor(li.distress_label.to_numpy(np.float32)), edge_index=torch.tensor(ee, dtype=torch.long).reshape(-1, 2).t().contiguous(), edge_type=torch.tensor(types, dtype=torch.long)))
            frames.append(li[['firm_id', 'feature_quarter', 'target_quarter', 'distress_label']])
        batches[split] = Batch.from_data_list(graphs)
        metas[split] = pd.concat(frames, ignore_index=True)
        hashes[split] = tensorhash({'x': batches[split].x, 'edge_index': batches[split].edge_index, 'edge_type': batches[split].edge_type})
    return (batches, metas, hashes)
