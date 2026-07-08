import torch
import numpy as np

class BodyFatDataset(torch.utils.data.Dataset):

    def __init__(self, data_frame, target_column='BodyFat'):
        feature_cols = [col for col in data_frame.columns if col != target_column]
        X = data_frame[feature_cols].values.astype(np.float32)
        y = data_frame[target_column].values.astype(np.float32)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]