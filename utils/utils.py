from typing import List
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

def kl_divergence():
  pass