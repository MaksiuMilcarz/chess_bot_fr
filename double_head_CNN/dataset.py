import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
from utils import bitboard_to_matrix

class HDF5ChessDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file):
        self.h5f = h5py.File(hdf5_file, 'r')
        self.boards = self.h5f['boards']
        self.policies = self.h5f['policies']
        self.values = self.h5f['values']
        self.length = self.boards.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        board = torch.from_numpy(self.boards[idx])
        policy = torch.from_numpy(self.policies[idx])
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        return board, policy, value

    def close(self):
        self.h5f.close()