import torch
from scipy.stats import multivariate_normal
import torch.nn as nn

def reverse(xT: torch.Tensor, T: int, betas: torch.Tensor, posterior_betas: torch.Tensor, alpha_bars: torch.Tensor, model: nn.Module):
  xt_prev = xT

  for t in range(T, 0, -1):
    if t > 1:
      z = torch.randn_like(xt_prev)
    else:
      z = 0

    alpha_t = 1 - betas[t]
    alpha_t_bar = alpha_bars[t]
    epsilon_t = model(xt_prev, t)
    std_t = torch.sqrt(posterior_betas[t])

    xt_prev = (1 / torch.sqrt(alpha_t)) * (xt_prev - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)) * epsilon_t) + (std_t * z)
  
  x0 = xt_prev
  return x0