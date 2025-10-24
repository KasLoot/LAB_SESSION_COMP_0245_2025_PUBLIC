import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from process_data import Part_1_Transformer_Dataset, process_data
from dataclasses import dataclass
from model import Part_1_Transformer_Model, ModelConfig
from tqdm import tqdm
import os
from pathlib import Path

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"





def train_model(train_config: TrainingConfig = TrainingConfig(), model_config: ModelConfig = ModelConfig()):

    train_dataset = Part_1_Transformer_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/train")




train_config = TrainingConfig()
model_config = ModelConfig()

train_model(train_config, model_config)