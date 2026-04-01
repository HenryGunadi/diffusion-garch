from torch.utils.data import Dataset
import numpy as np
import torch

class Returns(Dataset):
    def __init__(self, raw_returns: torch.Tensor, window_size, transform=None, standard=True):
        """
            transform: Insert the transformation function\n
            standard: Apply standardization   
        """
        super().__init__()
        assert isinstance(raw_returns, torch.Tensor), "raw returns must be in torch.Tensor"
        raw_returns = raw_returns.to(torch.float32)

        self.window_size = window_size

        if transform:
            raw_returns = transform(raw_returns)
        if standard is True:
            mean = raw_returns.mean()
            std = raw_returns.std()
            raw_returns = (raw_returns - mean) / std 
        
        self.data = raw_returns
        self.length = len(self.data) - window_size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]   # (L,)
        window = window.unsqueeze(0)                       # (1, L)
        return window