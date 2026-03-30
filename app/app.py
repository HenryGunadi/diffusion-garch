import joblib
from pathlib import Path
from datetime import date
import yfinance as yf
from utils import log_transform, posterior_beta, inverse_standard, one_step_forecast
import matplotlib.pyplot as plt
from nn import Unet1D
import torch
from diffusion import reverse
import math


# data
ticker = "^GSPC"
start_interval = "2016-12-01"
end_interval = "2026-01-01"
interval = "1d"

raw_snp500 = torch.tensor(yf.Ticker(ticker).history(start=start_interval, end=end_interval, interval=interval)["Close"].to_numpy())

split = math.ceil(len(raw_snp500) * 0.15)
val_split = len(raw_snp500) - math.ceil(len(raw_snp500) * 0.15) * 2
test_split = len(raw_snp500) - math.ceil(len(raw_snp500) * 0.15)
train_raw_snp500, val_raw_snp500, test_raw_snp500 = raw_snp500[:val_split], raw_snp500[val_split:test_split], raw_snp500[test_split:]

# training data
train_raw_snp500 = log_transform(train_raw_snp500)

encoder_in_channels = [1, 4, 8, 16]
encoder_out_channels = [4, 8, 16, 32]
decoder_in_channels = [32, 16, 8, 4]
decoder_out_channels = [16, 8, 4, 1]
attn_res = 16
n_res_block = 2
T = 1000
num_heads = 4
betas = torch.linspace(1e-4, 2e-2, T)
alpha_hats = torch.cumprod(
  input=1-betas,
  dim=0,
  dtype=torch.float32
)
model_diff = Unet1D(
  attn_res=attn_res,
  n_res_block=n_res_block,
  encoder_in_channels=encoder_in_channels,
  encoder_out_channels=encoder_out_channels,
  decoder_in_channels=decoder_in_channels,
  decoder_out_channels=decoder_out_channels,
  T=T,
  num_heads=num_heads
)
SAVE_PATH = dir / "models" / "model_diff_v0.pth"
checkpoint = torch.load(SAVE_PATH, weights_only=True)
model_diff.load_state_dict(checkpoint["model_state_dict"])

xT = torch.randn(size=(32, 1, 32))
posterior_betas = torch.tensor([posterior_beta(alpha_hats=alpha_hats, betas=betas, t=t) for t in range(T)])

# synthetic data
synthetic_data = reverse(
  xT=xT,
  T=T,
  betas=betas,
  posterior_betas=posterior_betas,
  alpha_bars=alpha_hats,
  model=model_diff
).squeeze(1).flatten(0, 1).detach().numpy()

synthetic_data = inverse_standard(
  standard_data=synthetic_data,
  data=train_raw_snp500
)

dir = Path().resolve().parent / "models" / "garch_diff_v0.pkl"
ticker = "^GSPC"
start = "2026-01-02"
end = str(date.today())
window = 32

raw_test_data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")["Close"]
rolling_var = raw_test_data.rolling(window=window).var()
test_data = log_transform(raw_test_data.to_numpy())

print(test_data)

# one-step forecast
# for i in range(test_data.shape[0]):
  

# print(garch_model.summary())