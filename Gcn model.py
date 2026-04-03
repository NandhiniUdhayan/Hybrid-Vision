# models/gcn_model.py

import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        return torch.relu(self.linear(x))


class GCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(3, 64)
        self.gcn2 = GCNLayer(64, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x = x.permute(0, 2, 1)
        return self.pool(x).squeeze(-1)


