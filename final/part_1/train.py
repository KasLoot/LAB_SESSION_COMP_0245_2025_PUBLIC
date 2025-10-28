import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import random_split


from process_data import Part_1_Dataset
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
    learning_rate: float = 1e-2
    num_epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_model(config: TrainingConfig = TrainingConfig()):



    train_dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/train")

    # Split dataset into training and validation sets
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # val_dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val")


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = Part_1_Model(input_size=7, hidden_size=64, output_size=7).to(torch.float32)
    
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
            q_diff, tau_cmd = batch
            # print(f"q_diff: \n{q_diff}")
            # print(f"tau_cmd: \n{tau_cmd}")
            q_diff = q_diff.to(torch.float32).to(config.device)
            # qd_diff = qd_diff.to(torch.float32).to(config.device)
            tau_cmd = tau_cmd.to(torch.float32).to(config.device)
            optimizer.zero_grad()
            logits = model(q_diff)
            loss = criterion(logits, tau_cmd)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        
        model.eval()
        epoch_val_losses = []
        for i, batch in enumerate(val_dataloader):
            q_diff, tau_cmd = batch
            q_diff = q_diff.to(torch.float32).to(config.device)
            tau_cmd = tau_cmd.to(torch.float32).to(config.device)
            with torch.no_grad():
                logits = model(q_diff)
                val_loss = criterion(logits, tau_cmd)
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
            }
            # Create checkpoints directory if it doesn't exist
            Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model_state, f"{model_save_dir}/best_part_1_model.pth")
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

    # Plot training and validation losses
    plot_training_curves(epochs_list, train_losses, val_losses, model_save_dir)
    



def plot_predictions_vs_true():
    model = Part_1_Model(input_size=7, hidden_size=64, output_size=7).to(torch.float32)
    model.load_state_dict(torch.load(f"/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_1/checkpoints/best_part_1_model.pth", weights_only=False)['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device).eval()
    val_q_diff = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/q_diff.pt").to(torch.float32).to(device)
    val_tau_cmd = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/tau_cmd.pt").to(torch.float32).to(device)

    with torch.no_grad():
        predictions = model(val_q_diff)

    predictions_np = predictions.cpu().numpy()
    true_values_np = val_tau_cmd.cpu().numpy()

    # Plot for each dimension
    num_dimensions = predictions_np.shape[1]
    for i in range(num_dimensions):
        plt.figure(figsize=(10, 6))
        plt.plot(true_values_np[:200, i], label='True Value', linewidth=2)
        plt.plot(predictions_np[:200, i], label='Prediction', linestyle='--', linewidth=2)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'Prediction vs. True Value for Dimension {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        save_dir = "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_1/checkpoints"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plot_path = f"{save_dir}/prediction_vs_true_dim_{i+1}.png"
        # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # print(f"Plot for dimension {i+1} saved to: {plot_path}")
        plt.show()



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


def calculate_r_squared_metrics():
    """
    Calculate and print R-squared and Adjusted R-squared for all seven joints
    using the best model on the validation dataset.
    """
    # Load the best model
    model = Part_1_Model(input_size=7, hidden_size=64, output_size=7).to(torch.float32)
    checkpoint = torch.load(
        "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_1/checkpoints/best_part_1_model.pth",
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device).eval()
    
    # Load validation data
    val_q_diff = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/q_diff.pt").to(torch.float32).to(device)
    val_tau_cmd = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/tau_cmd.pt").to(torch.float32).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(val_q_diff)
    
    # Convert to numpy for calculations
    predictions_np = predictions.cpu().numpy()
    true_values_np = val_tau_cmd.cpu().numpy()
    
    # Number of samples and features (joints)
    n_samples = predictions_np.shape[0]
    n_features = 7  # input_size
    
    print("\n" + "="*80)
    print("R-SQUARED AND ADJUSTED R-SQUARED METRICS FOR ALL SEVEN JOINTS")
    print("="*80)
    print(f"Number of samples: {n_samples}")
    print(f"Number of input features: {n_features}")
    print("-"*80)
    
    # Calculate metrics for each joint
    for joint_idx in range(7):
        y_true = true_values_np[:, joint_idx]
        y_pred = predictions_np[:, joint_idx]
        
        # Calculate R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate Adjusted R-squared
        # Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
        # where n = number of samples, p = number of features
        adjusted_r_squared = 1 - ((1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1))
        
        print(f"Joint {joint_idx + 1}:")
        print(f"  R-squared:          {r_squared:.6f}")
        print(f"  Adjusted R-squared: {adjusted_r_squared:.6f}")
        print("-"*80)
    
    # Calculate overall metrics (average across all joints)
    all_r_squared = []
    all_adjusted_r_squared = []
    
    for joint_idx in range(7):
        y_true = true_values_np[:, joint_idx]
        y_pred = predictions_np[:, joint_idx]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        adjusted_r_squared = 1 - ((1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1))
        
        all_r_squared.append(r_squared)
        all_adjusted_r_squared.append(adjusted_r_squared)
    
    avg_r_squared = np.mean(all_r_squared)
    avg_adjusted_r_squared = np.mean(all_adjusted_r_squared)
    
    print(f"AVERAGE ACROSS ALL JOINTS:")
    print(f"  Average R-squared:          {avg_r_squared:.6f}")
    print(f"  Average Adjusted R-squared: {avg_adjusted_r_squared:.6f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_model()
    calculate_r_squared_metrics()
    plot_predictions_vs_true()
