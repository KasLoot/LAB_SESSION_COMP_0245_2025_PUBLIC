import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import random_split


from get_dataset import Part_2_Dataset
from dataclasses import dataclass
from model import Part_2_Model

from tqdm import tqdm

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_model(config: TrainingConfig = TrainingConfig()):



    train_dataset = Part_2_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/train")

    # Split dataset into training and validation sets
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # val_dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val")


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    model = Part_2_Model(input_size=6, hidden_size=64, output_size=14).to(torch.float32)
    
    model.to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')
    model_save_dir = "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints"
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    epochs_list = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_train_losses = []
        for i, batch in enumerate(train_dataloader):
            cart_pos, cart_des_pos, q_des, qd_des_clip = batch
            input_tensor = torch.cat((cart_pos, cart_des_pos), dim=1)
            target_tensor = torch.cat((q_des, qd_des_clip), dim=1)
            input_tensor = input_tensor.to(torch.float32).to(config.device)
            target_tensor = target_tensor.to(torch.float32).to(config.device)
            optimizer.zero_grad()
            logits = model(input_tensor)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        
        model.eval()
        epoch_val_losses = []
        for i, batch in enumerate(val_dataloader):
            cart_pos, cart_des_pos, q_des, qd_des_clip = batch
            input_tensor = torch.cat((cart_pos, cart_des_pos), dim=1)
            target_tensor = torch.cat((q_des, qd_des_clip), dim=1)
            input_tensor = input_tensor.to(torch.float32).to(config.device)
            target_tensor = target_tensor.to(torch.float32).to(config.device)
            with torch.no_grad():
                logits = model(input_tensor)
                loss = criterion(logits, target_tensor)
                epoch_val_losses.append(loss.item())
        
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
            torch.save(model_state, f"{model_save_dir}/best_part_2_model.pth")
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")

    # Plot training and validation losses
    plot_training_curves(epochs_list, train_losses, val_losses, model_save_dir)
    



def plot_predictions_vs_true():
    model = Part_2_Model(input_size=6, hidden_size=64, output_size=14).to(torch.float32)
    model.load_state_dict(torch.load(f"/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints/best_part_2_model.pth", weights_only=False)['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device).eval()
    val_car_pos = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/cart_pos.pt").to(torch.float32).to(device)
    val_car_des_pos = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/cart_des_pos.pt").to(torch.float32).to(device)
    val_input = torch.cat((val_car_pos, val_car_des_pos), dim=1).to(device)
    val_qd_des_clip = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/qd_des_clip.pt").to(torch.float32).to(device)
    val_q_des = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val/q_des.pt").to(torch.float32).to(device)
    val_target = torch.cat((val_q_des, val_qd_des_clip), dim=1).to(device)
    

    with torch.no_grad():
        predictions = model(val_input)

    predictions_np = predictions.cpu().numpy()
    true_values_np = val_target.cpu().numpy()

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


if __name__ == "__main__":
    train_model()
    # plot_predictions_vs_true()