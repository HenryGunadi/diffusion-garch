import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusion import forward
from nn import Unet1D, EarlyStopping
from tqdm import tqdm
import time

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

    t = torch.randint(0, T, size=(batch_size,), device=device)
    xt, epsilon = forward(X, alpha_hats, t)

    epsilon_theta = model(xt, t)

    loss = loss_fn(epsilon_theta, epsilon)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
  
  train_loss = train_loss / len(data)

  return train_loss

def evaluate(
  data: DataLoader,
  loss_fn: nn.Module,
  model: Unet1D,
  alpha_hats: torch.Tensor,
  T: int,
  device: str = "cuda"
):
  val_loss = 0
  model.eval()

  with torch.inference_mode():
    for X in data:
      X = X.to(device)
      batch_size = X.size()[0]
      
      t = torch.randint(0, T, size=(batch_size,), device=device)
      xt, epsilon = forward(X, alpha_hats, t)

      epsilon_theta = model(xt, t)
      loss = loss_fn(epsilon_theta, epsilon)
      
      val_loss += loss.item()

  val_loss = val_loss / len(data)

  return val_loss

def train(
  train_data: DataLoader,
  val_data: DataLoader,
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
  """
    If an early stopping module is not provided, save manually after training finishes.
  """
  results = {
    "train_loss": [],
    "val_loss": [],
    "training_time": 0
  }
  model = model.to(device)
  start_time = time.time()

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

    val_loss = evaluate(
      data=val_data,
      loss_fn=loss_fn,
      model=model,
      alpha_hats=alpha_hats,
      T=T,
      device=device
    )

    print(f"Epoch : {epoch} | train_loss : {train_loss:.2f} | val_loss : {val_loss:.2f}")

    results["train_loss"].append(train_loss)
    results["val_loss"].append(val_loss)

    # early stop
    if early_stopping is not None:
      early_stopping.check_early_stop(val_loss=results["val_loss"], train_loss=results["train_loss"])

      if early_stopping.stop_training:
          print(f"Early stopping at epoch : {epoch}")
          break
    
    # scheduler
    if scheduler is not None:
      scheduler.step(val_loss)
      print("Current LR: ", scheduler.get_last_lr())

  if early_stopping is not None and not early_stopping.stop_training:
    print("Training completed. Saving best model...")

    if early_stopping.save_path.exists():
      print("Overwriting an existing model...")

    torch.save({
        "model_state_dict": early_stopping.best_model_state,
        "best_loss": early_stopping.best_loss,
        "train_loss": train_loss,
        "val_loss": val_loss 
    }, early_stopping.save_path)

  end_time = time.time()
  total_time = end_time - start_time
  print(f"Total training time: {total_time} seconds")
  results["training_time"] = total_time
  
  return results