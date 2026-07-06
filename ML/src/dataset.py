import torch
import numpy as np

class BodyFatDataset(torch.utils.data.Dataset):

    def __init__(self, data_frame):
        self.data = torch.tensor(data_frame.values, dtype=torch.float32)
        self.X = np.delete(self.data, 1, axis=1)  # Features (all columns except Bodyfat)
        self.y = self.data[:, 1]   # Target (Bodyfat column)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]