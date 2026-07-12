import numpy as np
import torch


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_model(model, data_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            batch_preds = model(X_batch).squeeze(-1)
            preds.append(batch_preds.cpu())
            targets.append(y_batch)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    metrics = compute_regression_metrics(targets, preds)
    return metrics, preds, targets


def naive_baseline_metrics(y_train, y_test):
    naive_pred = np.full_like(np.asarray(y_test, dtype=np.float32), y_train.mean())
    return compute_regression_metrics(y_test, naive_pred)


def print_comparison(model_metrics, baseline_metrics, model_name="Model"):
    print(f"{'Metric':<10} {model_name:<15} {'Baseline':<15}")
    print("-" * 40)
    for key in ["mae", "mse", "rmse", "r2"]:
        print(f"{key.upper():<10} {model_metrics[key]:<15.4f} {baseline_metrics[key]:<15.4f}")

def evaluate_vision_model(model, data_loader, device=None,target_mean=None, target_std=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch, gender_batch in data_loader:
            X_batch = X_batch.to(device)
            gender_batch = gender_batch.to(device)
            batch_preds = model(X_batch, gender_batch)
            preds.append(batch_preds.cpu())
            targets.append(y_batch)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    real_preds = (preds * target_std) + target_mean
    real_targets = (targets * target_std) + target_mean

    metrics = compute_regression_metrics(real_targets, real_preds)
    labeled_metrics = label_metrics(metrics, data_loader.dataset.measurement_cols)
    return labeled_metrics, real_preds, real_targets

def label_metrics(metrics, measurement_cols):
    labeled_metrics = dict(zip(measurement_cols, [{k: v[i] for k, v in metrics.items()} for i in range(len(measurement_cols))]))
    return labeled_metrics