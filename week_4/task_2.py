import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import random_split
from dataclasses import dataclass
from ignite.engine import Engine
from ignite.metrics import ConfusionMatrix
import numpy as np
import seaborn as sns


torch.manual_seed(32)

@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 0.01
    epochs: int = 20


def evaluation(model, data_loader, data_split='Validation'):
    # 5. Model Evaluation
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)

    print(f'{data_split} Loss: {loss:.4f}, {data_split} Accuracy: {accuracy:.2f}%')

    return loss, accuracy

config = Config()


# 1. Data Preparation
# Define transformations for the training and test sets
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
])

# Download and load the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
# Split train dataset into train and validation (e.g., 80-20 split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_subset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_subset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)


# 2. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 28),
            nn.RMSNorm(28, eps=1e-6),
            nn.SiLU(),
            nn.Linear(28, 16),
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
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

# 4. Model Training
epochs = config.epochs
train_losses = []
val_losses = []
test_losses = []
best_val_loss = float('inf')
best_val_accuracy = 0.0
best_test_loss = float('inf')
best_test_accuracy = 0.0

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

    val_loss, val_accuracy = evaluation(model, val_loader, data_split='Validation')
    val_losses.append(val_loss)

    test_loss, test_accuracy = evaluation(model, test_loader, data_split='Test')
    test_losses.append(test_loss)

    # Save model if it has the best validation loss so far
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_test_accuracy = test_accuracy
        model_save_path = './checkpoints/task_1.pth'
        model_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        torch.save(model_dict, model_save_path)
        print(f'✓ Best model saved with \nvalidation loss: {val_loss:.4f}, validation accuracy: {val_accuracy:.2f}%\ntest loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2f}%')

# 6. Print Best Model Statistics
print(f'\n{"="*60}')
print(f'Best Model - \nValidation Loss: {best_val_loss:.4f}, Validation Accuracy: {best_val_accuracy:.2f}%\nTest Loss: {best_test_loss:.4f}, Test Accuracy: {best_test_accuracy:.2f}%')
print(f'{"="*60}\n')

# 7. Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='s')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss', marker='^')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./checkpoints/bs{config.batch_size}_lr{config.learning_rate}_loss_plot.png', dpi=300)
# plt.show()
print(f"Loss plot saved to './checkpoints/bs{config.batch_size}_lr{config.learning_rate}_loss_plot.png'")


# 8. Confusion Matrix using PyTorch Ignite
print('\n' + '='*60)
print('Computing Confusion Matrix on Test Dataset')
print('='*60 + '\n')

def inference_step(engine, batch):
    model.eval()
    with torch.no_grad():
        data, target = batch
        data, target = data.to(device), target.to(device)
        output = model(data)
        # Return raw output (log probabilities) - Ignite will handle argmax internally
        return output, target

# Create inference engine
inference_engine = Engine(inference_step)

# Attach confusion matrix metric
cm_metric = ConfusionMatrix(num_classes=10)
cm_metric.attach(inference_engine, 'confusion_matrix')

# Run inference on test dataset
inference_engine.run(test_loader)

# Get the confusion matrix
confusion_matrix = cm_metric.compute().cpu().numpy()

print("Confusion Matrix:")
print(confusion_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Dataset')
plt.tight_layout()
plt.savefig(f'./checkpoints/bs{config.batch_size}_lr{config.learning_rate}_confusion_matrix.png', dpi=300)
plt.show()
print(f"\nConfusion matrix plot saved to './checkpoints/bs{config.batch_size}_lr{config.learning_rate}_confusion_matrix.png'")

# Calculate per-class metrics from confusion matrix
print('\n' + '='*60)
print('Per-Class Metrics from Confusion Matrix')
print('='*60)

for i in range(10):
    # True Positives: diagonal elements
    tp = confusion_matrix[i, i]
    # False Positives: sum of column i minus TP
    fp = confusion_matrix[:, i].sum() - tp
    # False Negatives: sum of row i minus TP
    fn = confusion_matrix[i, :].sum() - tp
    # True Negatives: total - (TP + FP + FN)
    tn = confusion_matrix.sum() - (tp + fp + fn)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f'Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}')

print('\n' + '='*60)



