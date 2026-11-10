import numpy as np

def score(y, p, t):
    y = np.asarray(y, int)
    z = np.asarray(p) >= t
    tp = int(((y == 1) & z).sum())
    fp = int(((y == 0) & z).sum())
    tn = int(((y == 0) & ~z).sum())
    fn = int(((y == 1) & ~z).sum())
    return dict(precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, accuracy=(tp + tn) / len(y), tp=tp, fp=fp, tn=tn, fn=fn)

def choose_valid_threshold(y_valid, p_valid, target_tp):
    """Only Valid arguments. >= boundaries: exact tp, else least tp overshoot,
    then higher precision then higher threshold. Includes empty-positive boundary."""
    p = np.asarray(p_valid)
    candidates = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
    feasible = []
    for t in candidates:
        m = score(y_valid, p, float(t))
        if m['tp'] >= target_tp:
            feasible.append((m['tp'] - target_tp, -m['precision'], -float(t), float(t), m))
    b = min(feasible, key=lambda x: x[:3])
    return (b[3], b[4], len(candidates))
