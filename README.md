# Technology-Conditioned Risk Factors and Multi-Relational Graph Fusion for Financial Distress Prediction

This package starts from finalized quarter-aligned experimental assets. Finance25 and two technology-conditioned risk factors form 27-dimensional node inputs. Patent and IPC relations are encoded by a shared relation-aware GAT with a 32-to-64-to-1 classifier.

## Directory structure

- `configs/`: verified model, training, and experiment settings.
- `src/`: model definitions, batching, training, feature validation, and graph validation.
- `scripts/`: executable validation and experiment commands.
- `data/metadata/`: labels, split assignments, feature order, and seeds.
- `data/model_input/`: finalized node matrices and baseline tensors.
- `data/technology/`: quarter-level IPC breadth.
- `data/graph/`: frozen relation graphs.
- `results/`: existing formal aggregate results.
- `docs/`: schemas, protocol, and file manifest.

## Environment setup

Use Python 3.13 and install dependencies with `python -m pip install -r requirements.txt`. The local verification environment used CPU PyTorch 2.12.1 and PyG 2.8.0. Numerical agreement across software versions and hardware requires independent verification.

## Input schema

The dataset contains 146 firms and 3504 firm-quarter observations from 2018Q4 through 2024Q3. Keys are `firm_id` and `feature_quarter`; `target_quarter` identifies the following quarter. `split` is Train, Valid, or Test. Labels are stored as `distress_label`. The feature schema gives the ordered 25 financial variables and two TCRF variables. See `docs/data_schema.md`.

## Validation without training

```sh
python scripts/verify_release.py
python src/features/build_tcrf.py
python src/graph/build_multirelational_graph.py
```

## Formal experiment commands

Run the following commands only when training is intended:

```sh
python scripts/run_financial_baselines.py
python scripts/run_neural_baselines.py
python scripts/run_module_ablation.py
python scripts/run_main_experiment.py
python scripts/run_design_comparisons.py
```

`python scripts/run_full_pipeline.py` executes this sequence after validation. The released node and graph assets are used directly. Labels and splits are fixed. Each experiment refuses to reuse an existing output directory. Move an earlier output directory deliberately before repeating an experiment.

## Output locations

Existing aggregate results are in `results/main`, `results/ablation`, and `results/design_comparisons`. Execution writes new checkpoints and per-seed aggregate metrics under `outputs/`. Individual evaluation probabilities remain in memory.

## Main aggregate results

| Configuration | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Finance baseline M0 | 0.551159 | 0.746377 | 0.633356 | 0.897945 |
| TCRF only M1 | 0.576196 | 0.765217 | 0.656102 | 0.905137 |
| Graph only M2 | 0.580478 | 0.778261 | 0.662981 | 0.906507 |
| Full M3 | 0.589144 | 0.775362 | 0.668667 | 0.909247 |

The full nine-model comparison is in `results/main/model_comparison.csv`. Standard deviations use ddof=1.

## Reproducibility notes

M0 and M1 retain native GAT self-loops. M0 and M2 zero the two TCRF slots while retaining the 27-dimensional backbone. M1 retains the frozen risk factors and removes inter-firm propagation. The main model uses Adam, learning rate 0.001, Weighted BCE with training negative-to-positive ratio, and the ten listed formal seeds. Checkpoint selection minimizes validation weighted BCE. Threshold selection maximizes validation F1, then accuracy, then favors the smaller threshold on the 0.05-to-0.95 grid with step 0.01.

The financial MLP and sequence baselines retain their own verified validation selection rules. Sequence checkpoint ranking uses validation F1, correlation, and average precision as in its formal implementation; final reported metrics are precision, recall, F1, and accuracy. The structural baseline consumes frozen derived feature and adjacency tensors. Design comparisons consume their own frozen node and graph pairs, including their corresponding risk factors. No training was performed during package organization.

The financial baseline command also evaluates LR using a validation-recall-matched threshold against the packaged full-model validation true-positive counts. These counts are per-seed aggregate references.


The full-graph node feature comparison and its offline verification command are described in [the experiment update](docs/node_feature_comparison.md).
