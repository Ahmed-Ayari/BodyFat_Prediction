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


CHECKPOINT_DIR = PROJECT_ROOT / "ML" / "checkpoints"
DATA_DIR = PROJECT_ROOT / "ML" / "data" / "BodyM"


def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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
    scaled = scaler.transform(row)

    X = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(X).squeeze(-1)

    return pred.item()

def predict_vision_model(model, scaler, front_image_mask, side_image_mask, device, measurement_cols, gender=1):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    front_image = transform(front_image_mask)
    side_image = transform(side_image_mask)

    gender_tensor = torch.tensor([float(gender)], dtype=torch.float32).to(device)  # shape: (1,)

    X = torch.cat([front_image, side_image], dim=0).unsqueeze(0).to(device)  # shape: (1, 2, 224, 224)

    with torch.no_grad():
        pred = model(X, gender_tensor)  # shape: (1, num_measurements)

    scaled_pred = pred.cpu().numpy()
    real_pred = (scaled_pred * scaler["std"]) + scaler["mean"]

    real_pred = real_pred.squeeze(0)  # shape: (num_measurements,)

    labled_pred = {measurement_cols[i]: real_pred[i] for i in range(len(real_pred))}

    return labled_pred

def combine_predictions(vision_measurements: dict, age: float, weight: float, height: float):
    combined_input = {
        "Age": age,
        "Weight": weight * 2.20462,  # Convert kg to lbs
        "Height": height / 2.54,  # Convert cm to inches
    }

    normalized_measurements = {
        key.strip().lower().replace("_", "-"): value
        for key, value in vision_measurements.items()
    }

    measurement_aliases = {
        "Chest": ["chest"],
        "Abdomen": ["waist"],
        "Hip": ["hip"],
        "Thigh": ["thigh"],
        "Ankle": ["ankle"],
        "Biceps": ["bicep", "biceps"],
        "Forearm": ["forearm"],
        "Wrist": ["wrist"],
    }

    for feature_name, aliases in measurement_aliases.items():
        value = next(
            (normalized_measurements[alias] for alias in aliases if alias in normalized_measurements),
            None,
        )
        if value is None:
            raise ValueError(f"Missing measurement for {feature_name} in vision model predictions.")
        combined_input[feature_name] = value

    return combined_input


model, feature_columns, device = load_model(CHECKPOINT_DIR / "bodyfat_mlp.pt")
scaler = load_scaler(CHECKPOINT_DIR / "bodyfat_scaler.pkl")
vision_model, measurement_cols, target_dict, device = load_model(CHECKPOINT_DIR / "bodym_efficientnet.pt")


if __name__ == "__main__":
    example_input = {
        "Age": 23,
        "Weight": 154.25,
        "Height": 67.75,
        "Chest": 93.1,
        "Abdomen": 85.2,
        "Hip": 94.5,
        "Thigh": 59.0,
        "Ankle": 21.9,
        "Biceps": 32.0,
        "Forearm": 27.4,
        "Wrist": 17.1,
    }

    vision_input = {
        "front_image_path": DATA_DIR / "testA" / "mask" / "0a61f2df7743db47dcbe5c9d0579e7ab.png",
        "side_image_path": DATA_DIR / "testA" / "mask_left" / "0a61f2df7743db47dcbe5c9d0579e7ab.png",
        "gender": 1,
    }

    prediction = predict(model, scaler, feature_columns, example_input, device)
    vision_prediction = predict_vision_model(
        vision_model, 
        target_dict, 
        vision_input["front_image_path"], 
        vision_input["side_image_path"], 
        device, 
        measurement_cols, 
        vision_input["gender"]
        )

    print(f"Predicted body fat: {prediction:.2f}%")
    print("Predicted body measurements:")
    for measurement, value in vision_prediction.items():
        print(f"{measurement}: {value:.2f}")