from PIL import Image
from torchvision import transforms
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


from model import BodyFatMLP, BodyM_EfficientNet


def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = ""
    
    if "input_size" in checkpoint:
        model_name = "BodyFatMLP"
        model = BodyFatMLP(
            input_size=checkpoint["input_size"],
            hidden_sizes=checkpoint["hidden_sizes"],
        )
    else:
        model_name = "BodyM_EfficientNet"
        model = BodyM_EfficientNet(
            in_channels=checkpoint["in_channels"],
            dropout_rate=checkpoint["dropout_rate"],
            output_size=checkpoint["output_size"],
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    extra_info = checkpoint.get("feature_columns", None) if model_name == "BodyFatMLP" else checkpoint["measurement_cols"]
    target_dict = dict(mean=checkpoint["target_mean"], std=checkpoint["target_std"]) if model_name == "BodyM_EfficientNet" else None

    if model_name == "BodyFatMLP":
        return model, extra_info, device
    else:
        return model, extra_info, target_dict, device


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

def predict_vision_model(model, scaler, front_image_path, side_image_path, device, measurement_cols):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    front_image = transform(Image.open(front_image_path).convert("L"))
    side_image = transform(Image.open(side_image_path).convert("L"))

    X = torch.cat([front_image, side_image], dim=0).unsqueeze(0).to(device)  # shape: (1, 2, 224, 224)

    with torch.no_grad():
        pred = model(X)

    scaled_pred = pred.cpu().numpy()
    real_pred = (scaled_pred * scaler["std"]) + scaler["mean"]

    real_pred = real_pred.squeeze(0)  # shape: (num_measurements,)

    labled_pred = {measurement_cols[i]: real_pred[i] for i in range(len(real_pred))}

    return labled_pred


if __name__ == "__main__":
    model, feature_columns, device = load_model("checkpoints/bodyfat_mlp.pt")
    scaler = load_scaler("checkpoints/bodyfat_scaler.pkl")

    vision_model, measurement_cols, target_dict, device = load_model("checkpoints/bodym_efficientnet.pt")

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