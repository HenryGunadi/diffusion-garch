from torch.utils.data import DataLoader, Dataset

class Returns(Dataset):
  def __init__(self, data, window_size, horizon, transform=None):
    super().__init__()

  def __len__(self):
    pass

  def __getitem__(self, index):
    pass