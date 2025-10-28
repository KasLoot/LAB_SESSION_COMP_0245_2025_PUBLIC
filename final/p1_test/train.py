import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
import pickle



class P1_Test_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(P1_Test_Model, self).__init__()
        self.rms = nn.RMSNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.dropout1 = nn.Dropout(dropout_rate)
        self.rms2 = nn.RMSNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.rms3 = nn.RMSNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.rms(x)
        # print(f"After RMSNorm: {x}")
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = self.rms2(x)
        x = F.relu(self.fc2(x))
        x = self.rms3(x)
        x = self.fc3(x)
        return x



class P1_Test_Dataset(Dataset):
    def __init__(self, data_path):
        self.q_mes = []
        self.q_des = []
        self.tau_mes = []
        self.tau_cmd = []
        for file_name in os.listdir(data_path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(data_path, file_name)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.q_mes.append(data["q_mes_all"])
                    self.q_des.append(data["q_d_all"])
                    self.tau_mes.append(data["tau_mes_all"])
                    self.tau_cmd.append(data["tau_cmd_all"])
        self.q_mes = torch.tensor(self.q_mes, dtype=torch.float32).view(-1, 7)
        self.q_des = torch.tensor(self.q_des, dtype=torch.float32).view(-1, 7)
        self.tau_mes = torch.tensor(self.tau_mes, dtype=torch.float32).view(-1, 7)
        self.tau_cmd = torch.tensor(self.tau_cmd, dtype=torch.float32).view(-1, 7)
        print(f"q_mes shape: {self.q_mes.shape}")
        print(f"q_des shape: {self.q_des.shape}")
        print(f"tau_mes shape: {self.tau_mes.shape}")
        print(f"tau_cmd shape: {self.tau_cmd.shape}")

    def __len__(self):
        return len(self.q_mes)

    def __getitem__(self, idx):
        return self.q_mes[idx], self.q_des[idx], self.tau_mes[idx], self.tau_cmd[idx]
    

def plot_loss(train_loss_history, val_loss_history):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()    


def train():
    # Training loop
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    dataset = P1_Test_Dataset(data_path="/home/yuxin/LAB_SESSION_COMP_0245_2025_PUBLIC/final/data/raw_data")
    
    train_data_size = int(0.8 * len(dataset))
    val_data_size = len(dataset) - train_data_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_data_size, val_data_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = P1_Test_Model(input_size=7, hidden_size=128, output_size=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_loss_history = []
    best_train_loss = float('inf')
    val_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        for q_mes_batch, q_des_batch, tau_mes_batch, tau_cmd_batch in train_loader:
            optimizer.zero_grad()

            joint_errors = q_des_batch - q_mes_batch
            joint_errors = joint_errors.to(device)
            target_torques = tau_mes_batch.to(device)

            outputs = model(joint_errors)
            loss = criterion(outputs, target_torques)
            # print(f"Training batch loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for q_mes_batch, q_des_batch, tau_mes_batch, tau_cmd_batch in val_loader:
                joint_errors = q_des_batch - q_mes_batch
                joint_errors = joint_errors.to(device)
                target_torques = tau_mes_batch.to(device)

                outputs = model(joint_errors)
                loss = criterion(outputs, target_torques)
                # print(f"Validation batch loss: {loss.item()}")
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "p1_test_model_best.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "p1_test_model_last.pth")

    plot_loss(train_loss_history, val_loss_history)




train()