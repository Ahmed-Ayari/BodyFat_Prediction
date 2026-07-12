import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms

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


class BodyMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, target_scaler=None):
        data_dir = Path(data_dir)
        self.mask_dir = data_dir / "mask"
        self.mask_left_dir = data_dir / "mask_left"

        hwg_metadata = pd.read_csv(data_dir / "hwg_metadata.csv").set_index("subject_id")
        measurements = pd.read_csv(data_dir / "measurements.csv").set_index("subject_id")
        photo_map = pd.read_csv(data_dir / "subject_to_photo_map.csv")

        photo_map = photo_map[photo_map["subject_id"].isin(measurements.index) & photo_map["subject_id"].isin(hwg_metadata.index)]

        self.samples = []
        for _, row in photo_map.iterrows():
            photo_id = row["photo_id"]
            front_path = self.mask_dir / f"{photo_id}.png"
            side_path = self.mask_left_dir / f"{photo_id}.png"
            if front_path.exists() and side_path.exists():
                self.samples.append((row["subject_id"], photo_id))

        self.measurements = measurements
        self.measurement_cols = list(measurements.columns)

        hwg_metadata["gender"] = hwg_metadata["gender"].apply(lambda x: 1 if x == "male" else 0)
        self.metadata = hwg_metadata
        self.metadata_cols = list(hwg_metadata.columns)

        if target_scaler is None:
            subject_ids = [s[0] for s in self.samples]
            vals = self.measurements.loc[subject_ids].values.astype(np.float32)
            self.target_mean = vals.mean(axis=0)
            self.target_std = vals.std(axis=0)
            self.target_std[self.target_std == 0] = 1.0
        else:
            self.target_mean = target_scaler["mean"]
            self.target_std = target_scaler["std"]

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_target_scaler(self):
        return {"mean": self.target_mean, "std": self.target_std}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, photo_id = self.samples[index]

        front = Image.open(self.mask_dir / f"{photo_id}.png").convert("L")
        side = Image.open(self.mask_left_dir / f"{photo_id}.png").convert("L")

        front_t = self.transform(front)
        side_t = self.transform(side)

        image = torch.cat([front_t, side_t], dim=0)  # shape: (2, 224, 224)

        target = self.measurements.loc[subject_id].values.astype(np.float32)
        target = (target - self.target_mean) / self.target_std

        subject_gender = self.metadata.loc[subject_id, "gender"]

        return image, torch.from_numpy(target), torch.tensor(subject_gender, dtype=torch.float32)