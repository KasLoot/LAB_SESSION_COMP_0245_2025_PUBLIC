import torch
import torch.nn as nn

class P2_MLP_None_Encoder(nn.Module):
    def __init__(self, input_size=10, output_size=14):
        super(P2_MLP_None_Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class P2_MLP(nn.Module):
    def __init__(self, input_size=10, output_size=14):
        super(P2_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.mlp(x) 

model = P2_MLP_None_Encoder(input_size=10, output_size=14)

model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized with {model_param_count} trainable parameters.")

model_2 = P2_MLP(input_size=10, output_size=14)
model_2_param_count = sum(p.numel() for p in model_2.parameters() if p.requires_grad)
print(f"Model 2 initialized with {model_2_param_count} trainable parameters.")