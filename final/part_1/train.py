import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from process_data import Part_1_Dataset, process_data
from dataclasses import dataclass
from model import Part_1_Model

from tqdm import tqdm


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train_model(config: TrainingConfig = TrainingConfig()):

    train_dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/train")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = Part_1_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/val")
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = Part_1_Model(input_size=7, hidden_size=64, output_size=7).to(torch.float32)
    
    model.to(config.device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            q_diff, qd_diff, tau_mes = batch
            q_diff = q_diff.to(config.device)
            tau_mes = tau_mes.to(config.device)
            optimizer.zero_grad()
            logits = model(q_diff)
            loss = criterion(logits, tau_mes)
            loss.backward()
            optimizer.step()
        model.eval()
        for i, batch in enumerate(val_dataloader):
            q_diff, qd_diff, tau_mes = batch
            q_diff = q_diff.to(config.device)
            tau_mes = tau_mes.to(config.device)
            with torch.no_grad():
                logits = model(q_diff)
                val_loss = criterion(logits, tau_mes)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        


if __name__ == "__main__":
    train_model()