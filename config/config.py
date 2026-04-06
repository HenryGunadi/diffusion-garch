class Config():
  def __init__(self):
    # model config
    self.model_config = {}

    # training config
    self.train_config = {}

  def set_model_config(
    self,
    **kwargs
  ):
    for key, value in kwargs.items():
      self.model_config[key] = value

    return self