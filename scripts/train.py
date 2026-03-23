import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusion import forward
from nn import Unet1D, EarlyStopping
from tqdm import tqdm

def train_step(
  data: DataLoader,
  model: Unet1D,
  loss_fn: nn.Module,
  optimizer: torch.optim.Optimizer,
  alpha_hats: torch.Tensor,
  T: int,
  device: str = "cuda"
):
  train_loss = 0
  model.train()

  for X in data:
    X = X.to(device, dtype=torch.float64)
    batch_size = X.size()[0]

    t = torch.randint(1, T + 1, size=(batch_size,), device=device)
    xt, epsilon = forward(X, alpha_hats, t)
    epsilon_theta = model(xt, t)

    print("epsilon theta : ", epsilon_theta)
    print("epsilon size : ", epsilon_theta.size())
    
    loss = loss_fn(epsilon_theta, epsilon)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

  train_loss = train_loss / len(data)

  return train_loss

def test_step(
  data: DataLoader,
  loss_fn: nn.Module,
  model: Unet1D,
  alpha_hats: torch.Tensor,
  T: int,
  device: str = "cuda"
):
  test_loss = 0
  model.eval()

  with torch.inference_mode():
    for X in data:
      X = X.to(device)
      batch_size = X.size()[0]
      
      t = torch.randint(1, T + 1, size=(batch_size,), device=device)
      xt, epsilon = forward(X, alpha_hats, t)

      epsilon_theta = model(xt, t)
      loss = loss_fn(epsilon_theta, epsilon)
      
      test_loss += loss.item()

  test_loss = test_loss / len(data)

  return test_loss

def train(
  train_data: DataLoader,
  test_data: DataLoader,
  optimizer: torch.optim.Optimizer,
  loss_fn: nn.Module,
  epochs: int,
  alpha_hats: torch.Tensor,
  model: Unet1D,
  T: int,
  scheduler: torch.optim.lr_scheduler._LRScheduler=None,
  early_stopping: EarlyStopping=None,
  device: str = "cuda",
):
  results = {
    "train_loss": [],
    "test_loss": [],
  }
  model = model.to(device)

  for epoch in tqdm(range(epochs)):
    train_loss = train_step(
      data=train_data,
      model=model,
      loss_fn=loss_fn,
      optimizer=optimizer,
      alpha_hats=alpha_hats,
      T=T,
      device=device
    )

    print("train passed")

    test_loss = test_step(
      data=test_data,
      loss_fn=loss_fn,
      model=model,
      alpha_hats=alpha_hats,
      T=T,
      device=device
    )

    print(f"Epoch : {epoch} | train_loss : {train_loss} | test_loss : {test_loss}")

    results["train_loss"].append(train_loss)
    results["test_loss"].append(test_loss)

    if early_stopping is not None:
      early_stopping.check_early_stop(test_loss)

      if early_stopping.stop_training:
          print(f"Early stopping at epoch : {epoch}")
          break

    if scheduler is not None:
        scheduler.step(test_loss)

  if early_stopping is not None and not early_stopping.stop_training:
      print("Training completed. Saving best model...")
      torch.save({
          "model_state_dict": early_stopping.best_model_state,
          "best_loss": early_stopping.best_loss
      }, early_stopping.save_path)
  
  return results