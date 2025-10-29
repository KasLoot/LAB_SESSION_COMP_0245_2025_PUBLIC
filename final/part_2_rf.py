from sklearn.ensemble import RandomForestRegressor
import numpy as np

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt


class P2_dataset(Dataset):
    def __init__(self, data_dir):
        print("Initializing P2_dataset...")
        self.q_mes_all = []
        self.desired_cartesian_pos_all = []
        self.q_des_all = []
        self.qd_des_all = []
        for files in os.listdir(data_dir):
            if files.endswith('.pkl') and files != 'data_0.pkl':
                file_path = os.path.join(data_dir, files)
                data = self.read_pkl_file(file_path)
                self.q_mes_all.append(data['q_mes_all'])
                self.desired_cartesian_pos_all.append(data['desired_cartesian_pos_all'])
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


def train():

    batch_size = 64

    dataset = P2_dataset(data_dir='./data/')

    train_data, val_data = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    X_list, y_list = [], []
    for batch in train_loader:
        cart_pos, cart_des_pos, q_des, qd_des_clip = batch
        input_tensor = torch.cat((cart_pos, cart_des_pos), dim=1)
        target_tensor = torch.cat((q_des, qd_des_clip), dim=1)
        X_list.append(input_tensor.numpy())
        y_list.append(target_tensor.numpy())

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    rf_model = RandomForestRegressor(
        n_estimators=300,        # 树的数量
        max_depth=30,           # 树的最大深度
        min_samples_leaf=3,     # 防止过拟合
        n_jobs=-1,              # 并行训练
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    X_test, y_test = [], []
    for batch in val_loader:
        cart_pos, cart_des_pos, q_des, qd_des_clip = batch
        input_tensor = torch.cat((cart_pos, cart_des_pos), dim=1)
        target_tensor = torch.cat((q_des, qd_des_clip), dim=1)
        X_test.append(input_tensor.numpy())
        y_test.append(target_tensor.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    y_pred = rf_model.predict(X_test)

    # 计算 MSE / R² 等指标
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}, R²: {r2:.4f}")


if __name__ == "__main__":
    train()



