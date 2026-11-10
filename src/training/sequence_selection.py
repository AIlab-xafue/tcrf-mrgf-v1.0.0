import numpy as np
from sklearn.metrics import average_precision_score

def selector(y, p, historical=False):
    grid = np.asarray([i / 100 for i in range(5, 96)])
    pred = p[None, :] >= grid[:, None]
    truth = y[None, :] == 1
    tp = (pred & truth).sum(1).astype(float)
    fp = (pred & ~truth).sum(1).astype(float)
    fn = (~pred & truth).sum(1).astype(float)
    tn = (~pred & ~truth).sum(1).astype(float)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(91), where=2 * tp + fp + fn > 0)
    if historical:
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros(91), where=denom > 0)
        ap = float(average_precision_score(y, p))
        idx = max(range(91), key=lambda k: (f1[k], mcc[k], ap, -grid[k]))
        return (float(grid[idx]), (float(f1[idx]), float(mcc[idx]), ap, -float(grid[idx])))
    acc = (tp + tn) / len(y)
    idx = max(range(91), key=lambda k: (f1[k], acc[k], -grid[k]))
    return (float(grid[idx]), None)
