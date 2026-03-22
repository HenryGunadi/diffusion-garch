import torch.nn as nn
from pathlib import Path
import copy
import torch

class EarlyStopping():
  def __init__(self, model: nn.Module, save_path: Path, patience=5, delta=0, verbose=False) -> None:
    self.patience = patience
    self.model = model
    self.save_path = save_path
    self.patience = patience
    self.delta = delta
    self.verbose = verbose
    self.best_loss = None
    self.no_improvement_count = 0
    self.stop_training = False
    self.best_model_state = None

  def check_early_stop(self, test_loss):
    if self.best_loss is None or test_loss < (self.best_loss - self.delta):
      self.no_improvement_count = 0
      self.best_model_state = copy.deepcopy(self.model.state_dict())
      self.best_loss = test_loss

      if self.verbose:
         print("New best model found. Test loss:", test_loss)
    else:
      self.no_improvement_count += 1
      if self.no_improvement_count == self.patience:
        self.stop_training = True
        torch.save({
          "model_state_dict": self.best_model_state,
          "best_loss": self.best_loss
        }, self.save_path)

        if self.verbose:
          print("Stopping early as no improvement has been observed")
          print("Best Loss", self.best_loss)
          print("The best model has been saved")