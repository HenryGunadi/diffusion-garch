import torch
import torch.nn as nn

class RMSELoss(nn.Module):
  def __init__(self, eps=1e-6):
      super().__init__()
      self.mse = nn.MSELoss()
      self.eps = eps
      
  def forward(self, pred, actual):
      loss = torch.sqrt(self.mse(pred, actual) + self.eps)
      return loss