import torch
from scipy.stats import multivariate_normal

def sample_prev_xt(x0: torch.Tensor, xt: torch.Tensor, alpha_hats: torch.Tensor, betas: torch.Tensor, t: int):
  """
    Since q(x_{t-1} | x_t, x_0) is defined as a closed form for training
    we could directly sample x_{t-1}
  """

  posterior_mean = (torch.sqrt(alpha_hats[t - 1] - 1) * betas[t]) / (1 - alpha_hats[t]) * x0 + (torch.sqrt(alpha_hats[t]) * (1 - alpha_hats[t - 1])) / (1 - alpha_hats[t]) * xt
  posterior_var = (1 - alpha_hats[t - 1]) / (1 - alpha_hats[t]) * betas[t]
  epsilon = torch.randn_like(xt) # this would sample noise from standard normal multivar gaussian

  prev_xt = posterior_mean + posterior_var * epsilon
  return prev_xt