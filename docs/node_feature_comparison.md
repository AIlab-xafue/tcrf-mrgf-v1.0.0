# Comparison of Node Feature Construction Methods under the Full Multi-Relational Technology Graph

## Added Experiment

This increment records the formal comparison of four node inputs on the frozen Joint graph. The graph has 441 Patent and 882 IPC relation pairs. The experiment uses 146 firms and 3,504 observations, with 2,336 Train, 584 Valid, and 584 Test rows.

## Experimental Groups

| Group | Identifier | Actual input after Finance25 | Historical run |
| --- | --- | --- | --- |
| G0 | FINANCE_ONLY | Two zero slots | M2_GRAPH_ONLY |
| G1 | PRESSURE_ONLY | Frozen profitability pressure and one zero slot | G1_PRESSURE_ONLY |
| G2 | DIRECT_TECH | Standardized isolation and breadth, in that order | G2_RAW_TECH, reused RawTech run |
| G3 | TCRF | Isolation risk and breadth risk, in that order | M3_FULL |

All groups use 27 inputs and the full graph. Finance-only G0 uses the graph-only ablation M2. G1 pressure is the original frozen value of `max(-z_npm, 0)`. The original TCRF slots are `p * max(u_iso, 0)` and `p * max(-u_breadth, 0)`.

## Source Code and Data Lineage

The formal full-graph driver reused the module-ablation training engine for ten pressure-only runs. It reused the thirty existing M2, RawTech, and M3 runs. The portable entry reuses the release training loop, model, and batching functions. Its protocol contains the original per-split input tensor hashes for all four groups.

The original runs use the same GAT backbone, 32-to-64-to-1 classifier, weighted BCE with positive weight 2100/236, Adam at 0.001 with weight decay 0.0001, and the formal ten seeds. Checkpoint selection uses strictly lower Valid weighted BCE with 30 epochs of patience and a 300-epoch limit. The Valid threshold grid spans 0.05 to 0.95 in increments of 0.01, with F1, accuracy, and the smaller threshold as the ordered selection criteria.

## Added Files

All experiment files are under `experiments/node_feature_comparison/`.

| File | Purpose |
| --- | --- |
| `run.py` | Construct the four input variants and verify the frozen archive |
| `protocol.json` | Formal settings, historical aliases, and input hashes |
| `pressure.csv` | One frozen G1 feature keyed by firm and quarter |
| `per_seed_results.csv` | 80 original records across Valid and Test |
| `summary_results.csv` | Original means and sample standard deviations by split |
| `paired_seed_differences.csv` | Original differences for six contrasts by seed and split |
| `paired_comparisons.csv` | Original paired summaries and direction counts |

The public manifest lists the new files and this note. The existing training module exposes its shared run loop to the new entry.

## Reused Existing Assets

The entry uses `data/graph/joint_edges.csv`, the existing labels and feature schema, `node_features.csv` for Finance25 and G3, and `direct_technology.csv` for G2. The same Joint graph is passed to every group.

## Execution Instructions

From the extracted release root, with the environment in `requirements.txt` installed, run the offline checks.

```sh
python -B scripts/verify_release.py
python -B experiments/node_feature_comparison/run.py
```

For a future authorized training reproduction of the ten G1 runs, the explicit command is shown below. Choose a new output directory. The archived tables provide G0, G2, and G3 through the formal reuse described above.

```sh
python -B experiments/node_feature_comparison/run.py --train-pressure-only outputs/node_feature_g1
```

## Result Verification

| Group | Test Precision | Test Recall | Test F1 | Test ACC |
| --- | ---: | ---: | ---: | ---: |
| G0 | 0.580478 | 0.778261 | 0.662981 | 0.906507 |
| G1 | 0.598033 | 0.757971 | 0.666938 | 0.910616 |
| G2 | 0.567339 | 0.817391 | 0.668652 | 0.904110 |
| G3 | 0.589144 | 0.775362 | 0.668667 | 0.909247 |

All four input batches match the original tensor hashes across the three splits. Means, sample standard deviations, and paired differences were recalculated from the original seed records. Read-only inference with all 40 archived checkpoints reproduced the saved Valid and Test probabilities exactly in the audit environment. The saved thresholds were used throughout. The release uses the display identifier G2_DIRECT_TECH in the two group-labeled tables and preserves all original numeric cells.

## Missing Assets or Limitations

The original checkpoints and individual predictions were located and verified in the source archive. The supplied ZIP distributes aggregate records and prepared inputs, so the increment follows that distribution scope. The packaged offline command verifies inputs and archived results. A standalone checkpoint replay requires access to the private source archive. The future training command was inspected but was outside the execution scope of this audit.
