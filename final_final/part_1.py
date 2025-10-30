"""
Training/evaluation script for Part 1.
"""
import os
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# Central configuration
# -------------------------
CONFIG = {
    "seed": 42,
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data": {
        "train_dir": "./data/train/", 
        "test_dir": "./data/test/",
        "batch_size": 64,
        "eval_batch_size": 64,
    },
    "model": {
        "input_size": 7,
        "output_size": 7,
        "hidden": [128, 256, 128],
        "activation": "leaky_relu",
    },
    "train": {
        "lr": 1e-3,
        "num_epochs": 500,
        "best_model_path": "part1_best_model.pth",
    },
    "plot": {
        "loss_curve_path": "part1_loss_curves.png",
    },
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DTYPE = CONFIG["dtype"]


# -------------------------
# Dataset
# -------------------------
class P1_dataset(Dataset):
    """
    Load .pth trajectory files and provide (q_mes, q_des, tau_cmd) samples as torch tensors.
    Expected .pth keys: 'q_mes_all', 'q_d_all', 'tau_cmd_all'.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.q_mes_all = []
        self.q_des_all = []
        self.tau_cmd_all = []
        self._load_files()

    def _load_files(self):
        files = sorted(os.listdir(self.data_dir))
        for f in files:
            if not f.endswith(".pt"):
                continue
            data = torch.load(os.path.join(self.data_dir, f), weights_only=False)
            self.q_mes_all.append(np.asarray(data.get("q_mes_all", [])))
            self.q_des_all.append(np.asarray(data.get("q_d_all", [])))
            self.tau_cmd_all.append(np.asarray(data.get("tau_cmd_all", [])))

        if len(self.q_mes_all) == 0:
            # Empty dataset guard
            self.q_mes_all = torch.tensor([], dtype=DTYPE)
            self.q_des_all = torch.tensor([], dtype=DTYPE)
            self.tau_cmd_all = torch.tensor([], dtype=DTYPE)
            return

        # Concatenate and convert to tensors
        self.q_mes_all = torch.tensor(np.concatenate(self.q_mes_all, axis=0), dtype=DTYPE)
        self.q_des_all = torch.tensor(np.concatenate(self.q_des_all, axis=0), dtype=DTYPE)
        self.tau_cmd_all = torch.tensor(np.concatenate(self.tau_cmd_all, axis=0), dtype=DTYPE)

        assert (
            self.q_mes_all.shape[0]
            == self.q_des_all.shape[0]
            == self.tau_cmd_all.shape[0]
        ), "Dataset arrays must have same first dimension."

    def __len__(self):
        return int(self.q_mes_all.shape[0])

    def __getitem__(self, idx):
        return self.q_mes_all[idx], self.q_des_all[idx], self.tau_cmd_all[idx]


# -------------------------
# Model
# -------------------------
class P1_MLP(nn.Module):
    """Configurable MLP for torque prediction."""

    def __init__(self, input_size=None, output_size=None, hidden=None, activation="leaky_relu"):
        super().__init__()
        input_size = input_size or CONFIG["model"]["input_size"]
        output_size = output_size or CONFIG["model"]["output_size"]
        hidden = hidden or CONFIG["model"]["hidden"]

        layers = []
        in_dim = input_size
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Utilities
# -------------------------
def plot_loss_curves(train_losses, val_losses, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# -------------------------
# Training loop
# -------------------------
def train():
    device = CONFIG["device"]
    data_cfg = CONFIG["data"]
    train_dir = data_cfg["train_dir"]
    test_dir = data_cfg["test_dir"]
    batch_size = data_cfg["batch_size"]
    eval_batch_size = data_cfg["eval_batch_size"]

    lr = CONFIG["train"]["lr"]
    num_epochs = CONFIG["train"]["num_epochs"]
    best_model_path = CONFIG["train"]["best_model_path"]

    # Prepare datasets/loaders
    train_dataset = P1_dataset(train_dir)
    if len(train_dataset) == 0:
        print("No training data found in", train_dir)
        return

    train_data, val_data = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=eval_batch_size, shuffle=False)
    test_dataset = P1_dataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    
    # Model / optimizer / loss
    model = P1_MLP(
        input_size=CONFIG["model"]["input_size"],
        output_size=CONFIG["model"]["output_size"],
        hidden=CONFIG["model"]["hidden"],
        activation=CONFIG["model"]["activation"],
    )
    model.to(dtype=DTYPE, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Trackers
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            q_mes, q_des, tau_cmd = batch
            inputs = (q_des - q_mes).to(device=device, dtype=DTYPE)
            targets = tau_cmd.to(device=device, dtype=DTYPE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        avg_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_batch_losses = []
            for batch in val_loader:
                q_mes, q_des, tau_cmd = batch
                inputs = (q_des - q_mes).to(device=device, dtype=DTYPE)
                targets = tau_cmd.to(device=device, dtype=DTYPE)
                outputs = model(inputs)
                val_batch_losses.append(float(loss_fn(outputs, targets).item()))
            avg_val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else float("nan")
            val_losses.append(avg_val_loss)
            
            # Test
            test_batch_losses = []
            for batch in test_loader:
                q_mes, q_des, tau_cmd = batch
                inputs = (q_des - q_mes).to(device=device, dtype=DTYPE)
                targets = tau_cmd.to(device=device, dtype=DTYPE)
                outputs = model(inputs)
                test_batch_losses.append(float(loss_fn(outputs, targets).item()))
            avg_test_loss = float(np.mean(test_batch_losses)) if len(test_batch_losses) else float("nan")
            test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}  Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  Test: {avg_test_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} (val loss {best_val_loss:.6f})")

    plot_loss_curves(train_losses, val_losses, save_path=CONFIG["plot"]["loss_curve_path"])
    print("Training finished. Best val loss:", best_val_loss)


# -------------------------
# Evaluation (MSE & R^2)
# -------------------------
def evaluate_best_model(model_path: str = None, data_dir: str = None, batch_size: int = None, device: torch.device = None):
    """
    Evaluate saved model and report per-dimension MSE and R^2 plus overall metrics.
    Prefer sklearn.metrics if available, otherwise fallback to numpy.
    """
    if model_path is None:
        model_path = CONFIG["train"]["best_model_path"]
    if data_dir is None:
        data_dir = CONFIG["data"]["test_dir"]
    if batch_size is None:
        batch_size = CONFIG["data"]["eval_batch_size"]
    if device is None:
        device = CONFIG["device"]

    dataset = P1_dataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = P1_MLP(
        input_size=CONFIG["model"]["input_size"],
        output_size=CONFIG["model"]["output_size"],
        hidden=CONFIG["model"]["hidden"],
        activation=CONFIG["model"]["activation"],
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(dtype=DTYPE, device=device)
    model.eval()

    preds_list, targets_list = [], []
    with torch.no_grad():
        for batch in loader:
            q_mes, q_des, tau_cmd = batch
            inputs = (q_des - q_mes).to(device=device, dtype=DTYPE)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(tau_cmd.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0) if preds_list else np.empty((0, CONFIG["model"]["output_size"]))
    targets = np.concatenate(targets_list, axis=0) if targets_list else np.empty((0, CONFIG["model"]["output_size"]))

    if preds.size == 0 or targets.size == 0:
        print("No data present for evaluation.")
        return {}

    # Per-dimension MSE and R^2
    mse_per_dim = mean_squared_error(targets, preds, multioutput="raw_values")
    r2_per_dim = r2_score(targets, preds, multioutput="raw_values")

    # Overall MSE (scalar) and overall R^2 computed on flattened vectors
    mse_overall = float(mean_squared_error(targets, preds, multioutput="uniform_average"))
    overall_r2 = float(r2_score(targets.flatten(), preds.flatten()))
   
    print(f"Overall MSE: {mse_overall:.6e}")
    print(f"Overall R^2: {overall_r2:.6f}")

    return {
        "mse_per_dim": mse_per_dim,
        "mse_overall": mse_overall,
        "r2_per_dim": r2_per_dim,
        "r2_overall": overall_r2,
        "preds": preds,
        "targets": targets,
    }


# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    # To train: uncomment train()
    train()

    # Evaluate saved model (default paths come from CONFIG)
    evaluate_best_model()
