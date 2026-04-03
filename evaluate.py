# evaluate.py

import torch
from torch.utils.data import DataLoader

from data_loader import YogaDataset
from models.hybrid_model import HybridModel
from utils import compute_accuracy

def evaluate():
    dataset = YogaDataset()
    loader = DataLoader(dataset, batch_size=8)

    model = HybridModel()
    model.eval()

    adj = torch.eye(17)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for vision, imu, labels in loader:
            outputs = model(vision, adj, imu)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    acc = compute_accuracy(y_true, y_pred)
    print("Accuracy:", acc)

if __name__ == "__main__":
    evaluate()