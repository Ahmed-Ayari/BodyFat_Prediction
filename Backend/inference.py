import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "ML"))
sys.path.append(str(PROJECT_ROOT / "ML/src"))
sys.path.append(str(PROJECT_ROOT / "ML/checkpoints"))


from model import BodyFatMLP


def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = BodyFatMLP(
        input_size=checkpoint["input_size"],
        hidden_sizes=checkpoint["hidden_sizes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["feature_columns"], device


def load_scaler(scaler_path):
    return joblib.load(scaler_path)


def predict(model, scaler, feature_columns, raw_input: dict, device):
    """
    raw_input: dict of {feature_name: value}, must contain all feature_columns.
    """
    row = pd.DataFrame([raw_input])[feature_columns]
    scaled = scaler.transform(row.values)

    X = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(X).squeeze(-1)

    return pred.item()


if __name__ == "__main__":
    model, feature_columns, device = load_model("checkpoints/bodyfat_mlp.pt")
    scaler = load_scaler("checkpoints/bodyfat_scaler.pkl")

    example_input = {
        "Age": 30,
        "Weight": 180,
        "Height": 70,
        "Neck": 38,
        "Chest": 100,
        "Abdomen": 90,
        "Hip": 98,
        "Thigh": 58,
        "Knee": 38,
        "Ankle": 22,
        "Biceps": 32,
        "Forearm": 28,
        "Wrist": 18,
    }

    prediction = predict(model, scaler, feature_columns, example_input, device)
    print(f"Predicted body fat: {prediction:.2f}%")