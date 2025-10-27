from torch.utils.data import Dataset
import torch
import os

class Part_2_Dataset(Dataset):
    def __init__(self, data_path):
        self.cart_pos = torch.load(os.path.join(data_path, "cart_pos.pt"))
        self.cart_des_pos = torch.load(os.path.join(data_path, "cart_des_pos.pt"))
        self.q_des = torch.load(os.path.join(data_path, "q_des.pt"))
        self.qd_des_clip = torch.load(os.path.join(data_path, "qd_des_clip.pt"))


    def __len__(self):
        return self.qd_des_clip.shape[0]

    def __getitem__(self, idx):
        return self.cart_pos[idx], self.cart_des_pos[idx], self.q_des[idx], self.qd_des_clip[idx]