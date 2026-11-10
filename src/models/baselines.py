import copy, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
THRESHOLDS = np.arange(5,96,dtype=np.int64)/100
METRICS = ["precision","recall","f1","accuracy"]

def threshold_grid(y, probability):
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    z = probability[:, None] >= THRESHOLDS[None, :]
    pos = y[:, None] == 1
    tp = (z & pos).sum(axis=0)
    fp = (z & ~pos).sum(axis=0)
    fn = (~z & pos).sum(axis=0)
    tn = (~z & ~pos).sum(axis=0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(THRESHOLDS)), where=denominator != 0)
    acc = (tp + tn) / len(y)
    precision = np.divide(tp, tp + fp, out=np.zeros(len(THRESHOLDS)), where=tp + fp != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(len(THRESHOLDS)), where=tp + fn != 0)
    index = max(range(len(THRESHOLDS)), key=lambda i: (f1[i], acc[i], -THRESHOLDS[i]))
    grid = pd.DataFrame({'threshold': THRESHOLDS, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc})
    return (float(THRESHOLDS[index]), {m: float(grid.iloc[index][m]) for m in METRICS}, grid)

def four_metrics(y, probability, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(probability) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, 'accuracy': (tp + tn) / len(y)}

class MLP(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dims, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def mlp_fit(xt, yt, xv, yv, seed, pos_weight):
    seed_all(seed)
    model = MLP(xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x = torch.tensor(xt, dtype=torch.float32)
    y = torch.tensor(yt, dtype=torch.float32)
    vx = torch.tensor(xv, dtype=torch.float32)
    weight = torch.tensor(pos_weight, dtype=torch.float32)
    best = None
    stale = 0
    history = []
    for epoch in range(1, 301):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight)
        if not torch.isfinite(loss):
            raise RuntimeError('MLP nonfinite Train loss')
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(vx)).numpy().astype(np.float64)
        t, vm, _ = threshold_grid(yv, pv)
        key = (vm['f1'], vm['accuracy'])
        improved = best is None or key > best[0]
        history.append({'epoch': epoch, 'train_weighted_bce': float(loss.detach()), 'Valid_F1': vm['f1'], 'Valid_ACC': vm['accuracy'], 'threshold': t, 'checkpoint_improved': improved})
        if improved:
            best = (key, copy.deepcopy(model.state_dict()), epoch)
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(vx)).numpy().astype(np.float64)
    return (model, pv, {'best_epoch': best[2], 'epochs_run': epoch, 'early_stopped': epoch < 300}, pd.DataFrame(history))

def classic_fit(name, xt, yt, xv, seed, ratio, cw):
    if name == 'LR':
        model = LogisticRegression(C=1.0, solver='liblinear', max_iter=5000, random_state=seed, class_weight='balanced')
        model.fit(xt, yt)
    elif name == 'SVM':
        model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=seed, class_weight='balanced')
        model.fit(xt, yt)
    elif name == 'XGBoost':
        model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=ratio, eval_metric='logloss', random_state=seed, n_jobs=4)
        model.fit(xt, yt)
    else:
        raise ValueError(name)
    return (model, model.predict_proba(xv)[:, 1], {'best_epoch': None, 'epochs_run': None, 'early_stopped': False}, None)

def probability(model, x, neural):
    if neural:
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
    else:
        p = model.predict_proba(x)[:, 1]
    p = np.asarray(p, dtype=np.float64)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p
