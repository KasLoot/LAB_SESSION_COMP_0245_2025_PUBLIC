import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import os
import pickle


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def process_data(data_path):
    q_mes_all = []
    qd_mes_all = []
    q_d_all = []
    qd_d_all = []
    tau_mes_all = []
    cart_pos_all = []
    cart_ori_all = []

    # for file end with .pkl in data_path
    for file_name in os.listdir(data_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(data_path, file_name)
            data = load_data(file_path)
            print(f"Loaded data from {file_name}:")
            q_mes_all.append(data["q_mes_all"])
            qd_mes_all.append(data["qd_mes_all"])
            q_d_all.append(data["q_d_all"])
            qd_d_all.append(data["qd_d_all"])
            tau_mes_all.append(data["tau_mes_all"])
            cart_pos_all.append(data["cart_pos_all"])
            cart_ori_all.append(data["cart_ori_all"])

    q_mes_all = torch.tensor(np.array(q_mes_all))
    qd_mes_all = torch.tensor(np.array(qd_mes_all))
    q_d_all = torch.tensor(np.array(q_d_all))
    qd_d_all = torch.tensor(np.array(qd_d_all))
    tau_mes_all = torch.tensor(np.array(tau_mes_all))
    cart_pos_all = torch.tensor(np.array(cart_pos_all))
    cart_ori_all = torch.tensor(np.array(cart_ori_all))

    print(f"q_mes_all shape: {q_mes_all.shape}")
    print(f"qd_mes_all shape: {qd_mes_all.shape}")
    print(f"q_d_all shape: {q_d_all.shape}")
    print(f"qd_d_all shape: {qd_d_all.shape}")
    print(f"tau_mes_all shape: {tau_mes_all.shape}")
    print(f"cart_pos_all shape: {cart_pos_all.shape}")
    print(f"cart_ori_all shape: {cart_ori_all.shape}")

    return q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all


class Part_1_Dataset(Dataset):
    def __init__(self, data_path):
        self.q_mes, self.qd_mes, self.q_d, self.qd_d, self.tau_mes, self.cart_pos, self.cart_ori = process_data(data_path=data_path)

        self.q_diff = self.q_d - self.q_mes
        self.qd_diff = self.qd_d - self.qd_mes

        self.q_diff = self.q_diff.reshape(-1, self.q_diff.shape[-1]).to(torch.float32)
        self.qd_diff = self.qd_diff.reshape(-1, self.qd_diff.shape[-1]).to(torch.float32)
        self.tau_mes = self.tau_mes.reshape(-1, self.tau_mes.shape[-1]).to(torch.float32)

        print(f"q_diff shape: {self.q_diff.shape}")
        print(f"qd_diff shape: {self.qd_diff.shape}")
        print(f"tau_mes shape: {self.tau_mes.shape}")

    def __len__(self):
        return self.q_diff.shape[0]

    def __getitem__(self, idx):
        return self.q_diff[idx], self.qd_diff[idx], self.tau_mes[idx]
