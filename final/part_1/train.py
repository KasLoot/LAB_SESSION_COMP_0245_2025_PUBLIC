import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import random_split


from process_data import Part_1_Dataset, process_data
from dataclasses import dataclass
from model import Part_1_Model

from tqdm import tqdm

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 400
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_model(config: TrainingConfig = TrainingConfig()):



    dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/raw_data")

    q_diff_std, q_diff_mean, qd_diff_std, qd_diff_mean, tau_cmd_std, tau_cmd_mean = dataset.get_std_mean()

    # Split train dataset into train and validation (e.g., 80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    model = Part_1_Model(input_size=7, hidden_size=128, output_size=7).to(torch.float32)
    
    model.to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')
    model_save_dir = "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_1/checkpoints"
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    epochs_list = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_train_losses = []
        for i, batch in enumerate(train_dataloader):
            q_diff, qd_diff, tau_mes = batch
            q_diff = q_diff.to(torch.float32).to(config.device)
            # qd_diff = qd_diff.to(torch.float32).to(config.device)
            tau_mes = tau_mes.to(torch.float32).to(config.device)
            optimizer.zero_grad()
            logits = model(q_diff)
            loss = criterion(logits, tau_mes)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        
        model.eval()
        epoch_val_losses = []
        for i, batch in enumerate(val_dataloader):
            q_diff, qd_diff, tau_mes = batch
            q_diff = q_diff.to(torch.float32).to(config.device)
            tau_mes = tau_mes.to(torch.float32).to(config.device)
            with torch.no_grad():
                logits = model(q_diff)
                val_loss = criterion(logits, tau_mes)
                epoch_val_losses.append(val_loss.item())
        
        # Calculate average validation loss for the epoch
        avg_val_loss = np.mean(epoch_val_losses)
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs_list.append(epoch + 1)
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'q_diff_std': q_diff_std,
                'q_diff_mean': q_diff_mean,
                'qd_diff_std': qd_diff_std,
                'qd_diff_mean': qd_diff_mean,
                'tau_cmd_std': tau_cmd_std,
                'tau_cmd_mean': tau_cmd_mean
            }
            # Create checkpoints directory if it doesn't exist
            Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model_state, f"{model_save_dir}/best_part_1_model.pth")
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

    # Plot training and validation losses
    plot_training_curves(epochs_list, train_losses, val_losses, model_save_dir)
    
    return train_losses, val_losses


def plot_training_curves(epochs, train_losses, val_losses, save_dir):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plot_path = f"{save_dir}/training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    train_model()