from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

class Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(27, 64)
        self.fc2 = nn.Linear(64, 64)
        self.relation_embedding = nn.Embedding(2, 8)
        self.gat = GATConv(64, 32, heads=4, concat=False, dropout=0.3, edge_dim=8, add_self_loops=True, fill_value='mean')

    def forward(self, d):
        h = F.relu(self.fc1(d.x))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.fc2(h))
        return F.elu(self.gat(h, d.edge_index, edge_attr=self.relation_embedding(d.edge_type)))

class Model(nn.Module):

    def __init__(self, architecture):
        super().__init__()
        self.backbone = Backbone()
        self.architecture = architecture
        assert architecture == 'GAT_HEAD_MLP64'
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, d):
        return self.head(self.backbone(d)).squeeze(-1)
