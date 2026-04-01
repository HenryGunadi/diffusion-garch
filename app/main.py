from pathlib import Path
from datetime import date
import math
import numpy as np
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from nn import Unet1D
from diffusion import reverse
from utils import log_transform, posterior_beta, inverse_standard, one_step_forecast

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="DiffVol")

# DIR = app/ so index.html is right here, models/ is one level up
DIR        = Path(__file__).parent
MODELS_DIR = DIR.parent / "models"

# ── Model config (fixed) ─────────────────────────────────────────────────────
T           = 1000
BATCH_SHAPE = (32, 1, 32)

encoder_in_channels  = [1,  4,  8, 16]
encoder_out_channels = [4,  8, 16, 32]
decoder_in_channels  = [32, 16,  8,  4]
decoder_out_channels = [16,  8,  4,  1]
attn_res    = 16
n_res_block = 2
num_heads   = 4

betas      = torch.linspace(1e-4, 2e-2, T)
alpha_hats = torch.cumprod(1 - betas, dim=0, dtype=torch.float32)

model_diff: Unet1D | None = None


@app.on_event("startup")
def load_model():
    global model_diff
    save_path = MODELS_DIR / "model_v0.pth"
    if not save_path.exists():
        print(f"[warn] model not found at {save_path}")
        return
    m = Unet1D(
        attn_res=attn_res,
        n_res_block=n_res_block,
        encoder_in_channels=encoder_in_channels,
        encoder_out_channels=encoder_out_channels,
        decoder_in_channels=decoder_in_channels,
        decoder_out_channels=decoder_out_channels,
        T=T,
        num_heads=num_heads,
    )
    ckpt = torch.load(save_path, weights_only=True)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    model_diff = m
    print("[ok] model loaded")


# ── Training data ────────────────────────────────────────────────────────────
def fetch_training_data() -> np.ndarray:
    raw = yf.Ticker("^GSPC").history(
        start="2016-12-01", end="2026-01-01", interval="1d"
    )["Close"].to_numpy()
    val_split = len(raw) - math.ceil(len(raw) * 0.15) * 2
    return log_transform(raw[:val_split])


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(DIR / "index.html")


class RunResponse(BaseModel):
    dates:      list[str]
    diff_preds: list[float]
    emp_preds:  list[float]
    proxy:      list[float]
    n_synthetic: int
    n_test:     int
    status:     str


@app.get("/api/run", response_model=RunResponse)
async def run():
    if model_diff is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check models/model_v0.pth")

    # Live test data
    raw_test = yf.Ticker("^GSPC").history(
        start="2026-01-02", end=str(date.today()), interval="1d"
    )["Close"]

    if len(raw_test) < 2:
        raise HTTPException(status_code=422, detail="Not enough live data (need >= 2 bars)")

    test_data            = log_transform(raw_test.to_numpy()) * 100
    proxy_squared_return = (test_data ** 2).tolist()
    dates                = [str(d) for d in raw_test.index[1:].date]

    # Training data for scaling
    train_log = fetch_training_data()

    # Synthetic generation
    xT = torch.randn(size=BATCH_SHAPE)
    posterior_betas = torch.tensor(
        [posterior_beta(alpha_hats=alpha_hats, betas=betas, t=t) for t in range(T)]
    )

    with torch.no_grad():
        synthetic = reverse(
            xT=xT,
            T=T,
            betas=betas,
            posterior_betas=posterior_betas,
            alpha_bars=alpha_hats,
            model=model_diff,
        ).squeeze(1).flatten(0, 1).detach().numpy()

    synthetic = inverse_standard(standard_data=synthetic, data=train_log) * 100

    # Forecasts
    diff_preds = one_step_forecast(synthetic, test_data)
    emp_preds  = one_step_forecast(train_log * 100, test_data)

    # Align proxy length
    proxy = proxy_squared_return[1:] if len(diff_preds) != len(proxy_squared_return) else proxy_squared_return
    if len(dates) > len(diff_preds):
        dates = dates[:len(diff_preds)]

    return RunResponse(
        dates=dates,
        diff_preds=[round(float(v), 6) for v in diff_preds],
        emp_preds=[round(float(v), 6) for v in emp_preds],
        proxy=[round(float(v), 6) for v in proxy],
        n_synthetic=int(np.prod(BATCH_SHAPE[::2])),
        n_test=len(test_data),
        status="ok",
    )


@app.get("/api/health")
async def health():
    return {
        "model_loaded": model_diff is not None,
        "device": "cpu",
        "T": T,
        "batch_shape": list(BATCH_SHAPE),
    }