"""Formal shared-backbone training from finalized experimental assets."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.models.tcrf_mrgf import Backbone, Model
from src.training.training_utils import seed_all, metrics, batches_from

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return pd.read_csv(path, dtype={'firm_id':str,'firm_a':str,'firm_b':str}, float_precision='round_trip')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-ablation', action='store_true')
    parser.add_argument('--design', choices=['direct_technology','text_patent','patent_only','jaccard_ipc'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    schema = json.loads((ROOT/'data/metadata/feature_schema.json').read_text())
    labels = read(ROOT/'data/metadata/labels.csv')
    full = read(ROOT/'data/model_input/node_features.csv')
    edges = read(ROOT/'data/graph/joint_edges.csv')
    if args.design:
        full = read(ROOT/f'data/model_input/{args.design}.csv')
        edges = read(ROOT/f'data/graph/{args.design}_edges.csv')
    configurations = ['M0_BASE','M1_TCRF_ONLY','M2_GRAPH_ONLY','M3_FULL'] if args.module_ablation else ['M3_FULL']
    out = ROOT/'outputs'/(args.design or ('ablation' if args.module_ablation else 'main'))
    inputs = []
    for configuration in configurations:
        node = full.copy()
        if configuration in ['M0_BASE','M2_GRAPH_ONLY']:
            node[schema['tcrf_features']] = 0.
        selected_edges = edges if configuration in ['M2_GRAPH_ONLY','M3_FULL'] else edges.iloc[:0]
        inputs.append((configuration, node, selected_edges))
    run_configurations(inputs, labels, schema, out)


def run_configurations(inputs, labels, schema, out):
    features = schema['finance_features'] + schema['tcrf_features']
    out.mkdir(parents=True, exist_ok=False)
    train_labels = labels.loc[labels.split.eq('Train'),'distress_label']
    weight = float((train_labels == 0).sum()/train_labels.sum())
    rows = []
    for configuration, node, selected_edges in inputs:
        batches, metas, _ = batches_from(node, selected_edges, labels, features)
        for seed in schema['formal_seeds']:
            seed_all(seed)
            shared = copy.deepcopy(Backbone().state_dict())
            seed_all(seed)
            model = Model('GAT_HEAD_MLP64')
            model.backbone.load_state_dict(shared)
            seed_all(seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
            pos_weight = torch.tensor(weight, dtype=torch.float32)
            best = float('inf')
            stale = 0
            for epoch in range(1,301):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(batches['Train']),batches['Train'].y,pos_weight=pos_weight)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    valid_loss = float(F.binary_cross_entropy_with_logits(model(batches['Valid']),batches['Valid'].y,pos_weight=pos_weight))
                if valid_loss < best:
                    best = valid_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= 30:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                valid_logits = model(batches['Valid']).double().numpy()
            valid_prob = 1/(1+np.exp(-valid_logits))
            grid = [{'threshold':i/100,**metrics(metas['Valid'].distress_label.to_numpy(),valid_prob,i/100)} for i in range(5,96)]
            threshold = max(grid,key=lambda r:(r['f1'],r['accuracy'],-r['threshold']))['threshold']
            torch.save({'state_dict':best_state,'seed':seed,'best_epoch':best_epoch,'threshold':threshold,'valid_loss':best},out/f'{configuration}_seed_{seed}.pt')
            with torch.no_grad():
                logits = model(batches['Test']).double().numpy()
            probabilities = 1/(1+np.exp(-logits))
            values = metrics(metas['Test'].distress_label.to_numpy(),probabilities,threshold)
            rows.append({'configuration':configuration,'seed':seed,**{k:values[k] for k in ['precision','recall','f1','accuracy']}})
            pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby('configuration')[['precision','recall','f1','accuracy']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')

if __name__ == '__main__':
    main()
