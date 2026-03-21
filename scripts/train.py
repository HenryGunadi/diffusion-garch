import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_step(
  epsilon: torch.Tensor,
  epochs: int,
  model: nn.Module,
  loss_fn: nn.Module,
  optimizer: torch.optim.optimizer,
  device: str = "cuda"
):
  # train_acc, train_loss = 0, 0
  # model.train()

  # for batch, ()
  pass