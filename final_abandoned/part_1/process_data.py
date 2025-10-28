import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import os
import pickle


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data



class Part_1_Dataset(Dataset):
    def __init__(self, data_path):
        self.q_diff = torch.load(os.path.join(data_path, "q_diff.pt")).to(torch.float32)
        self.tau_cmd = torch.load(os.path.join(data_path, "tau_cmd.pt")).to(torch.float32)

        print(f"q_diff shape: {self.q_diff.shape}")
        print(f"tau_cmd shape: {self.tau_cmd.shape}")

    # def get_std_mean(self):
    #     q_diff_std = torch.std(self.q_diff, dim=0)
    #     q_diff_mean = torch.mean(self.q_diff, dim=0)
    #     return q_diff_std, q_diff_mean

    def __len__(self):
        return self.q_diff.shape[0]

    def __getitem__(self, idx):
        return self.q_diff[idx], self.tau_cmd[idx]






# def process_data(data_path):
#     q_mes_all = []
#     qd_mes_all = []
#     q_d_all = []
#     qd_d_all = []
#     tau_mes_all = []
#     cart_pos_all = []
#     cart_ori_all = []
#     tau_cmd_all = []

#     # for file end with .pkl in data_path
#     for file_name in os.listdir(data_path):
#         if file_name.endswith('.pkl'):
#             file_path = os.path.join(data_path, file_name)
#             data = load_data(file_path)
#             print(f"Loaded data from {file_name}:")
#             q_mes_all.extend(data["q_mes_all"])
#             qd_mes_all.extend(data["qd_mes_all"])
#             q_d_all.extend(data["q_d_all"])
#             qd_d_all.extend(data["qd_d_all"])
#             tau_mes_all.extend(data["tau_mes_all"])
#             cart_pos_all.extend(data["cart_pos_all"])
#             cart_ori_all.extend(data["cart_ori_all"])
#             tau_cmd_all.extend(data["tau_cmd_all"]) 

#     q_mes_all = torch.tensor(np.array(q_mes_all))
#     qd_mes_all = torch.tensor(np.array(qd_mes_all))
#     q_d_all = torch.tensor(np.array(q_d_all))
#     qd_d_all = torch.tensor(np.array(qd_d_all))
#     tau_mes_all = torch.tensor(np.array(tau_mes_all))
#     cart_pos_all = torch.tensor(np.array(cart_pos_all))
#     cart_ori_all = torch.tensor(np.array(cart_ori_all))
#     tau_cmd_all = torch.tensor(np.array(tau_cmd_all))

#     print(f"q_mes_all shape: {q_mes_all.shape}")
#     print(f"qd_mes_all shape: {qd_mes_all.shape}")
#     print(f"q_d_all shape: {q_d_all.shape}")
#     print(f"qd_d_all shape: {qd_d_all.shape}")
#     print(f"tau_mes_all shape: {tau_mes_all.shape}")
#     print(f"cart_pos_all shape: {cart_pos_all.shape}")
#     print(f"cart_ori_all shape: {cart_ori_all.shape}")
#     print(f"tau_cmd_all shape: {tau_cmd_all.shape}")

#     return q_mes_all, qd_mes_all, q_d_all, qd_d_all, tau_mes_all, cart_pos_all, cart_ori_all, tau_cmd_all


# class Part_1_Dataset(Dataset):
#     def __init__(self, data_path):
#         self.q_mes, self.qd_mes, self.q_d, self.qd_d, self.tau_mes, self.cart_pos, self.cart_ori, self.tau_cmd = process_data(data_path=data_path)

#         self.q_diff = self.q_d - self.q_mes
#         self.qd_diff = self.qd_d - self.qd_mes

#         self.q_diff = self.q_diff.reshape(-1, self.q_diff.shape[-1]).to(torch.float32)
#         self.qd_diff = self.qd_diff.reshape(-1, self.qd_diff.shape[-1]).to(torch.float32)
#         self.tau_cmd = self.tau_cmd.reshape(-1, self.tau_cmd.shape[-1]).to(torch.float32)

#         print(f"q_diff shape: {self.q_diff.shape}")
#         print(f"qd_diff shape: {self.qd_diff.shape}")
#         print(f"tau_cmd shape: {self.tau_cmd.shape}")

#         print(f"q_diff[-1]: {self.q_diff[-1]}")
#         print(f"qd_diff[-1]: {self.qd_diff[-1]}")
#         print(f"tau_mes[100:110]: {self.tau_mes[100:110]}")
#         print(f"tau_cmd[100:110]: {self.tau_cmd[100:110]}")

#         # Normalization Process
#         print("\n--- Normalization Process ---")
#         self.tau_cmd_std = torch.std(self.tau_cmd, dim=0)
#         self.tau_cmd_mean = torch.mean(self.tau_cmd, dim=0)
#         self.q_diff_std = torch.std(self.q_d - self.q_mes, dim=0)
#         self.q_diff_mean = torch.mean(self.q_d - self.q_mes, dim=0)
#         self.qd_diff_std = torch.std(self.qd_d - self.qd_mes, dim=0)
#         self.qd_diff_mean = torch.mean(self.qd_d - self.qd_mes, dim=0)

#         # Add this check right before your training loop
#         print(f"y_train_tensor mean: {self.tau_cmd.mean():.4f}")
#         print(f"y_train_tensor std: {self.tau_cmd.std():.4f}")

#         # 3. Apply the normalization.
#         # We add a small epsilon value to the standard deviation to prevent division by zero
#         # in case a feature happens to have zero variance.
#         epsilon = 1e-8
#         self.tau_cmd = (self.tau_cmd - self.tau_cmd_mean) / (self.tau_cmd_std + epsilon)
#         self.q_diff = (self.q_diff - self.q_diff_mean) / (self.q_diff_std + epsilon)
#         self.qd_diff = (self.qd_diff - self.qd_diff_mean) / (self.qd_diff_std + epsilon)


#         # 4. Verify the result.
#         # The mean of the normalized data should be very close to 0, and the std dev close to 1.
#         print("--- After Normalization ---\n")
#         print(f"tau_cmd New Mean:\n{torch.mean(self.tau_cmd, dim=0)}\n")
#         print(f"tau_cmd New Std Dev:\n{torch.std(self.tau_cmd, dim=0)}\n")
#         print(f"q_diff New Mean:\n{torch.mean(self.q_diff, dim=0)}\n")
#         print(f"q_diff New Std Dev:\n{torch.std(self.q_diff, dim=0)}\n")
#         print(f"qd_diff New Mean:\n{torch.mean(self.qd_diff, dim=0)}\n")
#         print(f"qd_diff New Std Dev:\n{torch.std(self.qd_diff, dim=0)}\n")

#         print(f"q_diff[-1]: {self.q_diff[-1]}")
#         print(f"qd_diff[-1]: {self.qd_diff[-1]}")
#         print(f"tau_cmd[-1]: {self.tau_cmd[-1]}")

#     def get_std_mean(self):
#         return self.q_diff_std, self.q_diff_mean, self.qd_diff_std, self.qd_diff_mean, self.tau_cmd_std, self.tau_cmd_mean

#     def __len__(self):
#         return self.q_diff.shape[0]

#     def __getitem__(self, idx):
#         return self.q_diff[idx], self.tau_cmd[idx]



