from typing import List
import torch.nn as nn
import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

def crop_image(original, expected):
  """
    (N, C, L) -> dimensions
    
    Since we're dealing with 1-feature time series data -> 1D
  """

  original_dim = original.size()[-1]
  expected_dim = expected.size()[-1]

  difference = original_dim - expected_dim
  padding = difference // 2

  cropped = original[:, :, padding:original_dim-padding]

  return cropped

def normalize(x: torch.Tensor):
  assert len(x.size()) == 3, "Incorrect dimension size input. Expect (N, C, L) dimension"
  
  channels = x.size()[1]
  num_groups = min(32, channels)
  while channels % num_groups != 0:
    num_groups -= 1

  return nn.GroupNorm(num_groups, num_channels=channels)(x)

def log_transform(data):
  return np.log(data[1:] / data[:-1])

def attn_block(out_channels: int, x: torch.Tensor, num_heads: int = 4) -> torch.Tensor:
  assert out_channels % num_heads == 0, f"Out channels must be divisible by number of heads (Multiattention-block), ({out_channels}, {num_heads})"

  attn = nn.MultiheadAttention(
    embed_dim=out_channels,
    num_heads=num_heads,
    batch_first=True # expect (N, L, C) dim size 
  )

  x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
  x, _ = attn(x, x, x)
  x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)

  return x

def is_pth(path: str) -> bool:
    return Path(path).suffix.lower() == ".pth"

def posterior_beta(alpha_hats: torch.Tensor, betas, t: int):
  return ((1 - alpha_hats[t-1]) / (1 - alpha_hats[t])) * betas[t]