# === FILE: code/Dapper/mtl_model.py ===
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[128,64], dropout=0.3):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DapperMTL(nn.Module):
    def __init__(self, input_dim, shared_hidden=[256,128], dropout=0.3):
        super().__init__()
        self.encoder = MLP(input_dim, hidden=shared_hidden, dropout=dropout)
        shared_dim = shared_hidden[-1]
        # Task heads
        self.head_valence = nn.Linear(shared_dim, 1)
        self.head_arousal = nn.Linear(shared_dim, 1)
        self.head_panas_pos = nn.Linear(shared_dim, 1)  # Positive affect

    def forward(self, x):
        h = self.encoder(x)
        v = self.head_valence(h).squeeze(-1)
        a = self.head_arousal(h).squeeze(-1)
        p = self.head_panas_pos(h).squeeze(-1)
        return {'valence': v, 'arousal': a, 'panas_pos': p}
