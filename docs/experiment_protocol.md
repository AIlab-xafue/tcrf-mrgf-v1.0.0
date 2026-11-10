# Experiment protocol

The released main architecture is 27 to 64 to 64, a single four-head GAT with 32 output dimensions, a two-type eight-dimensional relation embedding, and classifier 32 to 64 to 1. GAT concatenation is disabled. Self-loops use mean edge attributes. Dropout is 0.3; optimizer weight decay is 0.0001. Maximum training duration is 300 epochs with patience 30.

Seeds are 1, 7, 11, 17, 21, 33, 42, 77, 1001, 2024. The training positive weight is 2100/236. The main checkpoint is selected by strict minimum validation weighted BCE. The threshold grid is integer 5 through 95 divided by 100, with F1, accuracy, smaller-threshold tie breaking. Aggregate results use sample standard deviations.

The financial MLP selects checkpoints using validation F1 then accuracy. Sequence baselines preserve their original validation ranking. The structural adaptation uses minimum validation weighted BCE. See the implementations for the retained baseline selection details.

Design comparisons retain finalized matrices for direct technology-state concatenation, text Patent plus TF-IDF IPC, Patent without IPC, and Patent plus Jaccard IPC. Full uses Patent plus TF-IDF IPC. Each graph-dependent risk matrix stays paired with its own graph. No alternative graph or risk state is generated during organization.
