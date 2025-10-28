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
    batch_size: int = 64
    learning_rate: float = 1e-2
    num_epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_model(config: TrainingConfig = TrainingConfig()):



    train_dataset = Part_2_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/train")

    # # Split dataset into training and validation sets
    # val_size = int(0.2 * len(train_dataset))
    # train_size = len(train_dataset) - val_size
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    test_dataset = Part_2_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/test")    

    # val_dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val")


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = Part_2_Model(input_size=10, output_size=14).to(torch.float64)
    
    model.to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_test_loss = float('inf')
    model_save_dir = "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints"
    
    # Lists to store losses for plotting
    train_losses = []
    test_losses = []
    epochs_list = []

    for epoch in range(config.num_epochs):
        model.train()
        epoch_train_losses = []
        for i, batch in enumerate(train_dataloader):
            q_mes, cart_des_pos, q_des, qd_des_clip = batch
            input_tensor = torch.cat((q_mes, cart_des_pos), dim=1)
            target_tensor = torch.cat((q_des, qd_des_clip), dim=1)
            # print(f"Input tensor shape: {input_tensor.shape}, \nTarget tensor shape: {target_tensor.shape}")
            # print(f"Input tensor sample: {input_tensor[0]}, \nTarget tensor sample: {target_tensor[0]}")
            input_tensor = input_tensor.to(config.device)
            target_tensor = target_tensor.to(config.device)
            optimizer.zero_grad()
            logits = model(input_tensor)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        
        model.eval()
        epoch_test_losses = []
        for i, batch in enumerate(test_dataloader):
            q_mes, cart_des_pos, q_des, qd_des_clip = batch
            input_tensor = torch.cat((q_mes, cart_des_pos), dim=1)
            target_tensor = torch.cat((q_des, qd_des_clip), dim=1)
            input_tensor = input_tensor.to(config.device)
            target_tensor = target_tensor.to(config.device)
            with torch.no_grad():
                logits = model(input_tensor)
                loss = criterion(logits, target_tensor)
                epoch_test_losses.append(loss.item())

        # Calculate average test loss for the epoch
        avg_test_loss = np.mean(epoch_test_losses)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        epochs_list.append(epoch + 1)
        
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }
            # Create checkpoints directory if it doesn't exist
            Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model_state, f"{model_save_dir}/best_part_2_model.pth")
            print(f"Saved Best Model with Test Loss: {best_test_loss:.4f}")

    # Plot training and test losses
    plot_training_curves(epochs_list, train_losses, test_losses, model_save_dir)




def plot_predictions_vs_true():
    model = Part_2_Model(input_size=10, hidden_size=64, output_size=14).to(torch.float32)
    model.load_state_dict(torch.load(f"/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints/best_part_2_model.pth", weights_only=False)['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device).eval()
    val_q_mes = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/test/q_mes.pt").to(torch.float32).to(device)
    val_cart_des_pos = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/test/cart_des_pos.pt").to(torch.float32).to(device)
    val_input = torch.cat((val_q_mes, val_cart_des_pos), dim=1).to(device)
    val_qd_des_clip = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/test/qd_des_clip.pt").to(torch.float32).to(device)
    val_q_des = torch.load("/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/test/q_des.pt").to(torch.float32).to(device)
    val_target = torch.cat((val_q_des, val_qd_des_clip), dim=1).to(device)
    
    print(f"Total validation samples: {val_input.shape[0]}")

    with torch.no_grad():
        predictions = model(val_input)

    predictions_np = predictions.cpu().numpy()
    true_values_np = val_target.cpu().numpy()

    # Plot for each dimension
    num_dimensions = predictions_np.shape[1]
    
    fig, axes = plt.subplots(7, 2, figsize=(15, 25))
    fig.suptitle('Prediction vs. True Value for All Dimensions', fontsize=16)
    axes = axes.flatten()

    for i in range(num_dimensions):
        ax = axes[i]
        ax.plot(true_values_np[:, i], label='True Value', linewidth=2)
        ax.plot(predictions_np[:, i], label='Prediction', linestyle='--', linewidth=2)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Dimension {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(num_dimensions, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the plot
    save_dir = "/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/part_2/checkpoints"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plot_path = f"{save_dir}/prediction_vs_true_all_dims.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
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
    plot_predictions_vs_true()