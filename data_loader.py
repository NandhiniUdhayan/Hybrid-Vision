# data_loader.py

import torch
from torch.utils.data import Dataset
import numpy as np

class YogaDataset(Dataset):
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vision = np.random.rand(17, 3)
        imu = np.random.rand(50, 6)
        label = np.random.randint(0, 82)

        return (
            torch.tensor(vision, dtype=torch.float32),
            torch.tensor(imu, dtype=torch.float32),
            torch.tensor(label)
        )