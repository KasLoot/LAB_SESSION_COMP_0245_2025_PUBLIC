import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass




class Part_2_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(Part_2_Model, self).__init__()
        self.rms = nn.RMSNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.rms2 = nn.RMSNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.rms3 = nn.RMSNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rms(x)
        # print(f"After RMSNorm: {x}")
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.rms2(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.rms3(x)
        x = self.fc3(x)
        return x