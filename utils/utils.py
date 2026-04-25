from typing import List
import torch.nn as nn
import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from arch import arch_model
import scipy.stats as stats

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
  num_groups = min(64, channels)
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

def create_dir(path: Path):
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)

def inverse_standard(standard_data, data):
  mean = data.mean()
  std = data.std()

  return standard_data * std + mean

def one_step_rolling_forecast(train_data, test_data):
  history = list(train_data)
  preds = []

  for t in range(len(test_data)):
      model = arch_model(history, vol='Garch', p=1, q=1)
      res = model.fit(disp="off")

      forecast = res.forecast(horizon=1)
      sigma = np.sqrt(forecast.variance.values[-1, 0])
      preds.append(sigma)

      history.append(test_data[t])
      # history = history[1:]

  return preds

def compute_aic_log_likelihood(windows):
  t_wins = 0
  normal_wins = 0
  delta_aic_list = []

  for w in windows:
      mu, sigma = stats.norm.fit(w)
      logL_normal = np.sum(stats.norm.logpdf(w, mu, sigma))
      AIC_normal = 2*2 - 2*logL_normal

      df, loc, scale = stats.t.fit(w)
      logL_t = np.sum(stats.t.logpdf(w, df, loc, scale))
      AIC_t = 2*3 - 2*logL_t

      delta = AIC_normal - AIC_t
      delta_aic_list.append(delta)

      if AIC_t < AIC_normal:
          t_wins += 1
      else:
          normal_wins += 1

  return t_wins, normal_wins, delta_aic_list