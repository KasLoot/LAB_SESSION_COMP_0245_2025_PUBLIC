import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(10)


class P2_dataset(Dataset):
    def __init__(self, data_dir):
        print("Initializing P2_dataset...")
        self.q_mes_all = []
        self.desired_cartesian_pos_all = []
        self.q_des_all = []
        self.qd_des_all = []
        for files in os.listdir(data_dir):
            if files.endswith('.pt'):
                file_path = os.path.join(data_dir, files)
                data = torch.load(file_path, weights_only=False)
                self.q_mes_all.append(data['q_mes_all'])
                self.desired_cartesian_pos_all.append(data['final_cartesian_pos'])
                self.q_des_all.append(data['q_d_all'])
                self.qd_des_all.append(data['qd_d_all'])

        self.q_mes_all = torch.tensor(np.concatenate(self.q_mes_all, axis=0), dtype=torch.float64)
        self.desired_cartesian_pos_all = torch.tensor(np.concatenate(self.desired_cartesian_pos_all, axis=0), dtype=torch.float64)
        self.q_des_all = torch.tensor(np.concatenate(self.q_des_all, axis=0), dtype=torch.float64)
        self.qd_des_all = torch.tensor(np.concatenate(self.qd_des_all, axis=0), dtype=torch.float64)

        print(f"Dataset initialized with {self.q_mes_all.shape[0]} samples.")
        print(f"q_mes_all shape: {self.q_mes_all.shape}")
        print(f"desired_cartesian_pos_all shape: {self.desired_cartesian_pos_all.shape}")
        print(f"q_des_all shape: {self.q_des_all.shape}")
        print(f"qd_des_all shape: {self.qd_des_all.shape}")

        assert self.q_mes_all.shape[0] == self.desired_cartesian_pos_all.shape[0] == self.q_des_all.shape[0] == self.qd_des_all.shape[0], "Mismatch in number of samples among data arrays."

    def read_pkl_file(self, file_path):
        data = None
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data


    def __len__(self):
        return self.q_mes_all.shape[0]


    def __getitem__(self, idx):
        return self.q_mes_all[idx], self.desired_cartesian_pos_all[idx], self.q_des_all[idx], self.qd_des_all[idx]


class P2_MLP(nn.Module):
    def __init__(self, input_size=10, output_size=14):
        super(P2_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.mlp(x)


def plot_loss_curves(train_losses, val_losses, test_losses, save_path='part2_loss_curves.png'):
    """
    Plot and save training, validation, and test loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_losses: List of test losses per epoch
        save_path: Path to save the plot image
    """
    num_epochs = len(train_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', markersize=3)
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', markersize=3)
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to '{save_path}'")
    plt.show()


def train():

    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = P2_dataset(data_dir='./data/train/')

    train_data, val_data = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    test_dataset = P2_dataset(data_dir='./data/test/')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.MSELoss()
    model = P2_MLP(input_size=10, output_size=14)
    model.to(torch.float64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_model_path = 'part2_best_model.pth'

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            q_mes, desired_cartesian_pos, q_des, qd_des = batch
            # Training code here
            optimizer.zero_grad()
            inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1)
            targets = torch.cat((q_des, qd_des), dim=1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        avg_train_loss = sum(epoch_loss) / len(epoch_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = []
            for batch in val_loader:
                q_mes, desired_cartesian_pos, q_des, qd_des = batch
                inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1)
                targets = torch.cat((q_des, qd_des), dim=1)
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss.append(loss.item())
            avg_val_loss = sum(val_loss) / len(val_loss)
            val_losses.append(avg_val_loss)

            test_loss = []
            for batch in test_loader:
                q_mes, desired_cartesian_pos, q_des, qd_des = batch
                inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1)
                targets = torch.cat((q_des, qd_des), dim=1)
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                test_loss.append(loss.item())
            avg_test_loss = sum(test_loss) / len(test_loss)
            test_losses.append(avg_test_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
            
            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with test loss: {best_test_loss:.4f}")


    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to '{best_model_path}'")

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, test_losses)


    # Validation code here

def evaluate_best_model(data_dir='./data/test/', model_path='part2_best_model.pth', batch_size=64, device=None):
    """
    评估已保存的 best model，返回每个输出维度的 MSE 和 R^2，以及整体 MSE / R^2。

    不依赖 sklearn，全部用 numpy/torch 计算，避免额外包依赖问题。
    """
    import math
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集（全部数据用于评估）
    dataset = P2_dataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 构建模型并加载权重
    model = P2_MLP(input_size=10, output_size=14)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(torch.float64).to(device)
    model.eval()

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch in loader:
            q_mes, desired_cartesian_pos, q_des, qd_des = batch
            inputs = torch.cat((q_mes, desired_cartesian_pos), dim=1).to(device)
            targets = torch.cat((q_des, qd_des), dim=1).to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    errors = preds - targets

    # 每维 MSE
    mse_per_dim = np.mean(errors**2, axis=0)
    # 整体 MSE（所有元素）
    mse_overall = np.mean(errors**2)

    # 每维 R^2
    y_mean = np.mean(targets, axis=0)
    ss_res = np.sum((targets - preds)**2, axis=0)
    ss_tot = np.sum((targets - y_mean)**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2_per_dim = 1.0 - ss_res / ss_tot
    r2_per_dim = np.where(ss_tot <= np.finfo(float).eps, 1.0, r2_per_dim)

    # 整体 R^2（所有元素合并）
    overall_ss_res = np.sum((targets - preds)**2)
    overall_ss_tot = np.sum((targets - np.mean(targets))**2)
    overall_r2 = 1.0 - overall_ss_res / overall_ss_tot if overall_ss_tot > np.finfo(float).eps else 1.0

    # 打印简要结果
    np.set_printoptions(precision=6, suppress=True)
    print("Per-dimension MSE (len={}):".format(mse_per_dim.size))
    print(mse_per_dim)
    print("Per-dimension R^2:")
    print(r2_per_dim)
    print(f"Overall MSE: {mse_overall:.6e}")
    print(f"Overall R^2: {overall_r2:.6f}")

    metrics = {
        "mse_per_dim": mse_per_dim,
        "mse_overall": mse_overall,
        "r2_per_dim": r2_per_dim,
        "r2_overall": overall_r2,
        "preds": preds,
        "targets": targets,
    }
    return metrics

if __name__ == "__main__":
    train()
    evaluate_best_model()
