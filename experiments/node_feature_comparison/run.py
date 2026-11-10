"""Verify archived node-feature results or explicitly run future G1 training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from src.training.train_model import read, run_configurations
from src.training.training_utils import batches_from

METRICS = ['Precision', 'Recall', 'F1', 'ACC']


def load_inputs():
    config = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    schema = json.loads((ROOT / 'data/metadata/feature_schema.json').read_text())
    full = read(ROOT / 'data/model_input/node_features.csv')
    direct = read(ROOT / 'data/model_input/direct_technology.csv')
    pressure = read(HERE / 'pressure.csv')
    labels = read(ROOT / 'data/metadata/labels.csv')
    edges = read(ROOT / 'data/graph/joint_edges.csv')
    keys = ['firm_id', 'feature_quarter']
    assert full[keys].equals(pressure[keys])
    metadata = keys + ['target_quarter', 'split']
    assert full[metadata].equals(labels[metadata])
    assert full[metadata].equals(direct[metadata])
    assert full[schema['finance_features']].equals(direct[schema['finance_features']])
    assert labels.groupby('split').size().to_dict() == {'Train': 2336, 'Valid': 584, 'Test': 584}
    assert len(edges) == 1323
    assert edges.relation_type.value_counts().to_dict() == {'Patent': 441, 'IPC': 882}
    assert config['seeds'] == schema['formal_seeds']
    slots = schema['tcrf_features']
    features = schema['finance_features'] + slots
    assert len(features) == config['input_dim'] == 27
    finance = full.assign(**dict.fromkeys(slots, 0.0))
    groups = {
        'G0_FINANCE_ONLY': finance,
        'G1_PRESSURE_ONLY': finance.assign(**{slots[0]: pressure['pressure']}),
        'G2_DIRECT_TECH': direct,
        'G3_TCRF': full,
    }
    for name, node in groups.items():
        _, _, hashes = batches_from(node, edges, labels, features)
        assert hashes == config['input_tensor_hashes'][name], name
    settings = yaml.safe_load((ROOT / 'configs/formal_training.yaml').read_text())
    assert settings == {
        'optimizer': config['optimizer'].lower(), 'learning_rate': config['lr'],
        'loss': 'weighted_bce', 'weight_decay': config['weight_decay'],
        'max_epochs': config['max_epochs'], 'patience': config['patience'],
        'formal_seeds': config['seeds'], 'checkpoint_selection': 'minimum_validation_weighted_bce',
        'threshold_grid': [0.05, 0.95, 0.01],
        'threshold_selection': ['f1', 'accuracy', 'smaller_threshold'],
    }
    train_y = labels.loc[labels.split.eq('Train'), 'distress_label']
    assert (train_y == 0).sum() / train_y.sum() == config['pos_weight']
    return config, schema, labels, edges, groups


def verify_results(config):
    results = read(HERE / 'per_seed_results.csv')
    reference = read(HERE / 'summary_results.csv').set_index(['group', 'split'])
    expected = config['input_tensor_hashes'].keys()
    assert len(results) == 80 and len(reference) == 8
    assert not results.duplicated(['group', 'split', 'seed']).any()
    assert set(reference.index) == {(g, s) for g in expected for s in ['Valid', 'Test']}
    for (group, split), frame in results.groupby(['group', 'split']):
        assert sorted(frame.seed) == config['seeds']
        for metric in METRICS:
            np.testing.assert_allclose(
                [frame[metric].mean(), frame[metric].std(ddof=1)],
                reference.loc[(group, split), [f'{metric}_mean', f'{metric}_std']].to_numpy(float),
                atol=1e-14, rtol=0,
            )
    seed_pairs = read(HERE / 'paired_seed_differences.csv')
    pairs = read(HERE / 'paired_comparisons.csv')
    lookup = results.set_index(['group', 'split', 'seed'])
    group_ids = {name[:2]: name for name in expected}
    assert len(seed_pairs) == 120 and len(pairs) == 48
    for (contrast, split), frame in seed_pairs.groupby(['comparison', 'split']):
        left, right = (group_ids[key] for key in contrast.split('-'))
        assert sorted(frame.seed) == config['seeds']
        for metric in METRICS:
            delta = lookup.loc[(left, split), metric] - lookup.loc[(right, split), metric]
            np.testing.assert_allclose(frame.set_index('seed')[metric + '_delta'].sort_index(),
                                       delta.sort_index(), atol=1e-14, rtol=0)
            saved = pairs.loc[(pairs.comparison == contrast) & (pairs.split == split) &
                              (pairs.metric == metric)].iloc[0]
            np.testing.assert_allclose([delta.mean(), delta.std(ddof=1)],
                                       [saved.mean_delta, saved.sample_std], atol=1e-14, rtol=0)
            assert [int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum())] == [
                saved.positive, saved.negative, saved.tie]
    print(reference.xs('Test', level='split').to_string())
    print('NODE_FEATURE_COMPARISON = PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pressure-only', type=Path, metavar='NEW_OUTPUT_DIR',
                        help='Explicitly launch the historical G1 training protocol in a new directory')
    args = parser.parse_args()
    torch.set_num_threads(2)
    config, schema, labels, edges, groups = load_inputs()
    verify_results(config)
    if args.train_pressure_only:
        torch.use_deterministic_algorithms(True)
        name = 'G1_PRESSURE_ONLY'
        run_configurations([(name, groups[name], edges)], labels, schema, args.train_pressure_only)


if __name__ == '__main__':
    main()
