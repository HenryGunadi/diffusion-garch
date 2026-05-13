from torch.utils.data import Dataset
import numpy as np
import torch

class Returns(Dataset):
    def __init__(self, returns: torch.Tensor, window_size):
        super().__init__()
        assert isinstance(returns, torch.Tensor), "returns must be in torch.Tensor"
        returns = returns.to(torch.float32)

        self.window_size = window_size

        self.data = returns
        self.length = len(self.data) - window_size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]   # (L,)
        window = window.unsqueeze(0)                       # (1, L)
        return window