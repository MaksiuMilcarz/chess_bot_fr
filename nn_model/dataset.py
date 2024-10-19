from torch.utils.data import Dataset
import torch
import os

class ChunkedChessDataset(Dataset):
    def __init__(self, data_dir='data_chunks'):
        """
        Initialize the dataset by listing all chunk files and calculating cumulative sample sizes.

        Args:
        - data_dir (str): Directory where data chunks are stored.
        """
        self.data_dir = data_dir
        self.chunk_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('data_chunk_')])
        self.cumulative_sizes = []
        self.total_samples = 0

        # Calculate cumulative sizes for indexing
        for chunk_file in self.chunk_files:
            data = torch.load(chunk_file)
            chunk_size = len(data['y'])
            self.total_samples += chunk_size
            self.cumulative_sizes.append(self.total_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.

        Args:
        - idx (int): Index of the sample.

        Returns:
        - X (torch.Tensor): Input tensor of shape (13, 8, 8).
        - y (torch.Tensor): Target tensor (integer label).
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} is out of bounds for dataset with {self.total_samples} samples.")

        # Binary search to find the correct chunk
        left, right = 0, len(self.cumulative_sizes) - 1
        while left <= right:
            mid = (left + right) // 2
            if idx < self.cumulative_sizes[mid]:
                right = mid - 1
            else:
                left = mid + 1

        chunk_idx = left
        chunk_file = self.chunk_files[chunk_idx]
        data = torch.load(chunk_file)
        
        # Calculate the sample index within the chunk
        if chunk_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[chunk_idx - 1]

        X = data['X'][sample_idx]
        y = data['y'][sample_idx]

        return X, y