import numpy as np
import torch
from typing import List, Tuple

def forward(x0: torch.Tensor, alpha_hats: torch.Tensor, t: int):
  mean = torch.sqrt(alpha_hats[t]) * x0
  sigma = torch.sqrt(1 - alpha_hats[t])

  return sample_xt(mean, sigma)

def sample_xt(mu: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  eps = torch.randn_like(mu)
  return mu + sigma * eps, eps