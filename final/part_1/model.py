import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int = 7
    hidden_size: int = 64
    output_size: int = 7
    num_heads: int = 4
    num_layers: int = 2



class Part_1_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Part_1_Model, self).__init__()
        self.rms1 = nn.RMSNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.rms1(x)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = F.silu(self.fc4(x))
        x = self.fc5(x)
        return x


class Part_1_Transformer_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, num_layers=2):
        super(Part_1_Transformer_Model, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        

    def forward(self, x):
        
        return x
    

