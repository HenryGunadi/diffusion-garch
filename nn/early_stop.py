import torch.nn as nn
from pathlib import Path
import copy
import torch
from utils import is_pth

class EarlyStopping():
  def __init__(self, model: nn.Module, save_path: Path,  file_name: str, patience=5, delta=0, verbose=False) -> None:
    self.save_path = save_path
    self.save_path.mkdir(parents=True, exist_ok=True)
    self.save_path = self.save_path / file_name

    assert is_pth(self.save_path), "Model save path must be in .pth format"

    self.patience = patience
    self.model = model
    self.patience = patience
    self.delta = delta
    self.verbose = verbose
    self.best_loss = None
    self.no_improvement_count = 0
    self.stop_training = False
    self.best_model_state = None
    self.file_name = file_name

  def check_early_stop(self, val_loss, train_loss):
    if self.best_loss is None or val_loss[-1] < (self.best_loss - self.delta):
      self.no_improvement_count = 0
      self.best_model_state = copy.deepcopy(self.model.state_dict())
      self.best_loss = val_loss[-1]
      

      if self.verbose:
         print("New best model found. Current loss:", val_loss[-1])
    else:
      self.no_improvement_count += 1
      if self.no_improvement_count == self.patience:
        self.stop_training = True
        torch.save({
          "model_state_dict": self.best_model_state,
          "best_loss": self.best_loss,
          "train_loss": train_loss,
          "val_loss": val_loss
        }, self.save_path)

        if self.verbose:
          print("Stopping early as no improvement has been observed")
          print("Best Loss", self.best_loss)

          if self.save_path.exists():
            print("Overwriting an existing model...")

          print("The best model has been saved")