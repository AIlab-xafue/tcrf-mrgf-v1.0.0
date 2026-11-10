import torch
from torch import nn
from torch.nn import functional as F

class GCN(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, 64)
        self.b = nn.Linear(64, 64)

    def forward(self, x, adj):
        a = adj + torch.eye(len(adj), device=x.device)
        d = a.sum(1).clamp_min(1).pow(-0.5)
        a = d[:, None] * a * d[None, :]
        return F.relu(a @ self.b(F.dropout(F.relu(a @ self.a(x)), p=0.3, training=self.training)))

class Net(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.p = GCN(d)
        self.i = GCN(d)
        self.att = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.lstm = nn.LSTM(64, 32, 2, batch_first=True, dropout=0.3)
        self.out = nn.Linear(32, 1)

    def embed(self, x, pa, ia):
        h = torch.stack([self.p(x, pa), self.i(x, ia)], 1)
        return (torch.softmax(self.att(h).squeeze(-1), 1).unsqueeze(-1) * h).sum(1)

    def logits(self, x, pa, ia, ends):
        needed = sorted({q for t in ends for q in range(t - 3, t + 1)})
        emb = {q: self.embed(x[q], pa[q], ia[q]) for q in needed}
        sequence = torch.stack([torch.stack([emb[q] for q in range(t - 3, t + 1)], 1) for t in ends]).reshape(-1, 4, 64)
        return self.out(self.lstm(sequence)[0][:, -1]).squeeze(-1)
