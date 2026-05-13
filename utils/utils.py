from typing import List
import torch.nn as nn
import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from arch import arch_model
import scipy.stats as stats
from statsmodels.tsa.stattools import acf, pacf, adfuller
import matplotlib.pyplot as plt
import yfinance as yf
import math


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

# def normalize(x: torch.Tensor):
#   assert len(x.size()) == 3, "Incorrect dimension size input. Expect (N, C, L) dimension"
  
#   channels = x.size()[1]
#   num_groups = min(64, channels)
#   while channels % num_groups != 0:
#     num_groups -= 1

#   return nn.GroupNorm(num_groups, num_channels=channels)(x)

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

def one_step_rolling_forecast(train_data, test_data, dist="t"):
  """
    default-dist: t
  """
  history = list(train_data)
  preds = []
  losses = []

  for t in range(len(test_data)):
    model = arch_model(history, vol='Garch', p=1, q=1, dist=dist)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=1)
    sigma = np.sqrt(forecast.variance.values[-1, 0])

    preds.append(sigma)

    var = sigma ** 2

    # modified qlike loss
    loss = (test_data[t] ** 2 / var) + np.log(var)
    losses.append(loss)

    history.append(test_data[t])
    history = history[1:]

  return preds, losses

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

def compute_aic_log_likelihood_stdresid(models):
    """
    Compare Normal vs Student-t fit on standardized residuals
    from GARCH models.

    Returns:
    - t_wins: how often Student-t wins
    - normal_wins: how often Normal wins
    - delta_aic_list: AIC(normal) - AIC(t)
    """

    t_wins = 0
    normal_wins = 0
    delta_aic_list = []

    for res in models:

        z = res.std_resid

        mu, sigma = stats.norm.fit(z)
        logL_normal = np.sum(stats.norm.logpdf(z, mu, sigma))
        AIC_normal = 2 * 2 - 2 * logL_normal 

        df, loc, scale = stats.t.fit(z)
        logL_t = np.sum(stats.t.logpdf(z, df, loc, scale))
        AIC_t = 2 * 3 - 2 * logL_t 

        delta = AIC_normal - AIC_t
        delta_aic_list.append(delta)

        if AIC_t < AIC_normal:
            t_wins += 1
        else:
            normal_wins += 1

    return t_wins, normal_wins, delta_aic_list

def split_into_windows(data, window_size):
  """
  Non-overlapping windows.

  Returns:
    np.ndarray of shape (n_windows, window_size)
  """
  n = len(data)
  n_windows = n // window_size 

  trimmed = data[:n_windows * window_size]
  windows = trimmed.reshape(n_windows, window_size)

  return windows

def test_stationarity(windows, maxlag=10, regression="ct"):
  result = []

  for window in windows:
    res = adfuller(window, maxlag=maxlag, regression=regression)
    result.append(res)

  return result

def plot_distribution(synthetic_data, empirical_window):
  std = synthetic_data.reshape(-1).std()
  mu = synthetic_data.reshape(-1).mean()

  data_min = synthetic_data.reshape(-1).min()
  data_max = synthetic_data.reshape(-1).max()

  data_min_emp = empirical_window.reshape(-1).min()
  data_max_emp = empirical_window.reshape(-1).max()

  xmin = min(data_min, data_min_emp)
  xmax = max(data_max, data_max_emp)

  x = np.linspace(xmin, xmax, 1000)

  kde = stats.gaussian_kde(synthetic_data.reshape(-1))
  kde_emp = stats.gaussian_kde(empirical_window.reshape(-1))

  norm = stats.norm(
      loc=mu,
      scale=std,
  )

  fig, ax = plt.subplots(figsize=(8, 10), nrows=2)
  fig.suptitle("Comparison of Marginal Return Distributions KDE (Scott) - 128 Window", fontsize=14)

  ax[0].plot(x, kde(x), label="KDE Scott Synthetic")
  ax[0].plot(x, norm.pdf(x), label="Normal")
  ax[0].set_title("Normal Distribution")
  ax[0].legend()

  ax[1].plot(x, kde(x), label="KDE Scott Synthetic")
  ax[1].plot(x, kde_emp(x), label="KDE Scott Empirical")
  ax[1].set_title("Emprical vs Synthetic")
  ax[1].legend()

  plt.tight_layout()
  plt.show()

def load_and_split_snp500(
  window = None,
  start_interval="2010-01-01",
  end_interval="2026-01-01",
  interval="1d",
  ticker="^GSPC",
  transform_fn=None,
  cut=0.2,
):
  """
  Loads S&P 500 data, splits into train/val/test, and applies optional transform.

  Returns:
      train, val, test (and raw split if needed)
  """
  raw = yf.Ticker(ticker).history(
    start=start_interval,
    end=end_interval,
    interval=interval
  )["Close"].to_numpy()

  split = math.ceil(len(raw) * cut)
  val_split = len(raw) - split * 2
  test_split = len(raw) - split

  train_raw = raw[:val_split]
  val_raw = raw[val_split:test_split]
  test_raw = raw[test_split:]

  if transform_fn is not None:
    train = transform_fn(train_raw)
    val = transform_fn(val_raw)
    test = transform_fn(test_raw)
  else:
    train, val, test = train_raw, val_raw, test_raw

  return {
    "train": train,
    "val": val,
    "test": test,
    "train_raw": train_raw,
    "val_raw": val_raw,
    "test_raw": test_raw,
    "window": window
  }

def compute_var(sigma_paths, capital = 1000):
  rt_paths = []

  for sigma_path in sigma_paths:
    rt = np.zeros(len(sigma_path))

    for sigma in sigma_path:
      sigma2 = sigma ** 2
      z = stats.t.rvs(loc=0, scale=1)
      

