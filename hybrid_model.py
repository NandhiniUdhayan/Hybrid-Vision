# models/hybrid_model.py

import torch.nn as nn
from models.gcn_model import GCNModel
from models.lstm_model import IMULSTM
from models.fusion_model import AttentionFusion

class HybridModel(nn.Module):
    def __init__(self, num_classes=82):
        super().__init__()

        self.gcn = GCNModel()
        self.lstm = IMULSTM()
        self.fusion = AttentionFusion()

        self.fc = nn.Linear(64, num_classes)

    def forward(self, vision, adj, imu):
        v_feat = self.gcn(vision, adj)
        i_feat = self.lstm(imu)

        fused = self.fusion(v_feat, i_feat)

        return self.fc(fused)