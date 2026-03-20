from typing import List
import torch.nn as nn
import torch

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
  assert len(x.size()) != 3, "Incorrect dimension size input. Expect (N, C, L) dimension"
  
  channels = x.size()[1]
  num_groups = min(32, channels)
  while channels % num_groups != 0:
    num_groups -= 1

  return nn.GroupNorm(num_groups, num_channels=channels)(x)