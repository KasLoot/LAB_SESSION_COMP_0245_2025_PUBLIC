import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int = 7
    hidden_size: int = 128
    output_size: int = 7
    num_heads: int = 4
    num_layers: int = 2



class Part_1_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(Part_1_Model, self).__init__()
        self.rms = nn.RMSNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rms(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Part_1_Transformer_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, num_layers=2):
        super(Part_1_Transformer_Model, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        

    def forward(self, x):
        
        return x
    

