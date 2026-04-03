# models/fusion_model.py

import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, v, i):
        combined = torch.cat([v, i], dim=1)
        weights = self.fc(combined)

        v_w = weights[:, 0].unsqueeze(1)
        i_w = weights[:, 1].unsqueeze(1)

        return v_w * v + i_w * i

