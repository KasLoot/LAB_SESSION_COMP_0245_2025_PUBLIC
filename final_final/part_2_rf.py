"""
Random Forest regression training script.
"""
from pathlib import Path
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import matplotlib.pyplot as plt
from part_2 import P2_dataset

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "data_dir": "./data/train/", 
    "test_dir": "./data/test/",
    "batch_size": 64,
    "rf": {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": 42,
    },
    "save_path": "part2_rf_model.pkl",
    "seed": 42,
    "plot": {
        "enabled": False,
        "save_path": "rf_val_scatter.png",
    },
}

# Set random seeds
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

DTYPE = torch.float64

# -------------------------
# Helpers
# -------------------------
def stack_loader_to_arrays(loader: DataLoader):
    """
    Extract X, y numpy arrays from a DataLoader of P2_dataset.
    X = [q_mes, desired_cart_pos], y = [q_des, qd_des]
    """
    X_list = []
    y_list = []
    for batch in loader:
        q_mes, desired_cart_pos, q_des, qd_des = batch
        inputs = torch.cat((q_mes, desired_cart_pos), dim=1).cpu().numpy()
        targets = torch.cat((q_des, qd_des), dim=1).cpu().numpy()
        X_list.append(inputs)
        y_list.append(targets)
    if not X_list:
        return np.empty((0,)), np.empty((0,))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute MSE and R^2 (overall and per-dimension)."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # per-dimension values
    mse_per_dim = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2_per_dim = r2_score(y_true, y_pred, multioutput="raw_values")
    return {"mse": mse, "r2": r2, "mse_per_dim": mse_per_dim, "r2_per_dim": r2_per_dim}


# -------------------------
# Training / evaluation
# -------------------------
def train():
    """Train RandomForest on concatenated .pt dataset and evaluate on val/test sets."""
    batch_size = CONFIG["batch_size"]
    data_dir = CONFIG["data_dir"]
    test_dir = CONFIG["test_dir"]

    dataset = P2_dataset(data_dir=data_dir)
    if len(dataset) == 0:
        print(f"No data found in {data_dir}")
        return

    # split train/val
    n = len(dataset)
    n_train = int(0.8 * n)
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n - n_train])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataset = P2_dataset(data_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) else None

    # prepare X/y
    X_train, y_train = stack_loader_to_arrays(train_loader)
    X_val, y_val = stack_loader_to_arrays(val_loader)

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # configure and fit RF
    rf_cfg = CONFIG["rf"]
    rf = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        n_jobs=rf_cfg["n_jobs"],
        random_state=rf_cfg["random_state"],
    )
    rf.fit(X_train, y_train)

    # Validation metrics
    y_val_pred = rf.predict(X_val)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    print(f"Validation MSE: {val_metrics['mse']:.6f}, R²: {val_metrics['r2']:.6f}")

    # Optional scatter plot for first output dim (if enabled)
    if CONFIG["plot"]["enabled"] and X_val.size and y_val.size:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_val[:, 0], y_val_pred[:, 0], s=2)
        plt.xlabel("True (dim0)"); plt.ylabel("Pred (dim0)")
        plt.title("Validation: true vs pred (dim0)")
        plt.grid(True)
        plt.savefig(CONFIG["plot"]["save_path"], dpi=200)
        plt.close()

    # Test evaluation if test dataset present
    if test_loader is not None:
        X_test, y_test = stack_loader_to_arrays(test_loader)
        y_test_pred = rf.predict(X_test)
        test_metrics = evaluate_predictions(y_test, y_test_pred)
        print(f"Test MSE: {test_metrics['mse']:.6f}, R²: {test_metrics['r2']:.6f}")

    # Save model
    joblib.dump(rf, CONFIG["save_path"])
    print(f"Saved RandomForest model to {CONFIG['save_path']}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    train()



