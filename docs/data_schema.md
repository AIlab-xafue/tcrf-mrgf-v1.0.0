# Data schema

`labels.csv`: firm_id, feature_quarter, target_quarter, split, distress_label. Train contains 2336 rows and 236 positives; Valid contains 584 rows and 64 positives; Test contains 584 rows and 69 positives.

`node_features.csv`: the same ordered metadata keys plus Finance25 and isolation_risk, breadth_risk. Finance variables are already preprocessed. Do not fit a new transform before consuming these values. The financial schema lists feature order.

`joint_edges.csv`: feature_quarter, firm_a, firm_b, pair_key, relation_type, edge_weight. Patent has 441 undirected pairs and IPC has 882. Batching expands each pair in both directions and maps Patent to 0 and IPC to 1. Edge weights remain metadata; the formal GAT consumes relation embeddings.

`financial_sequences.npz`: x of shape (3504,4,25), distress_label, firm_id, feature_quarter, target_quarter, split. The sequence contains finalized financial inputs.

`structural_baseline/node_features.npy`: (27,146,796) derived feature tensor. `quarter_graphs.npz`: patent and ipc adjacency matrices, quarters and firms. Four-quarter sequences terminate at the labeled feature quarter.

Design matrices use the same feature slots and metadata as the main input. In direct_technology.csv the final two slots hold standardized technology states, as specified by that comparison.
