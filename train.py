# train.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from data_loader import YogaDataset
from models.hybrid_model import HybridModel

def train():
    dataset = YogaDataset()
    loader = DataLoader(dataset, batch_size=8)

    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    adj = torch.eye(17)

    for epoch in range(5):
        total_loss = 0

        for vision, imu, labels in loader:
            outputs = model(vision, adj, imu)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()