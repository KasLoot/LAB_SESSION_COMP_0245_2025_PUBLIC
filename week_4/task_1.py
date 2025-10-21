import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(32)


# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
#     x_glu, x_linear = x[..., ::2], x[..., 1::2]
#     # Clamp the input values
#     x_glu = x_glu.clamp(min=None, max=limit)
#     x_linear = x_linear.clamp(min=-limit, max=limit)
#     out_glu = x_glu * torch.sigmoid(alpha * x_glu)
#     # Note we add an extra bias of 1 to the linear layer
#     return out_glu * (x_linear + 1)

# 2. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 16),
            nn.RMSNorm(16, eps=1e-6),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.RMSNorm(16, eps=1e-6),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.RMSNorm(16, eps=1e-6),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.RMSNorm(16, eps=1e-6),
            nn.SiLU(),
            nn.Linear(16, 10),
            nn.RMSNorm(10, eps=1e-6),
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability
        )

    def forward(self, x):
        x = self.model(x)
        return x

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = MLP().to(device)
torch.compile(model)

model_params = sum(p.numel() for p in model.model.parameters())
print(f'Total model parameters: {model_params}')

# 3. Model Compilation
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss (used with LogSoftmax)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Model Training
epochs = 20
train_losses = []
valid_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # target is not one-hot encoded in PyTorch
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_losses.append(epoch_loss / len(train_loader))
    train_accuracy = 100. * correct / len(train_loader.dataset)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.2f}%')

# 5. Model Evaluation
model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100. * correct / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

model_save_path = './checkpoints/task_1.pth'
model_dict = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses[-1],
    'train_accuracy': train_accuracy,
    'test_loss': test_loss,
    'test_accuracy': test_accuracy
}
torch.save(model_dict, model_save_path)



