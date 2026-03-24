import math
import torch
import torch.nn as nn

class SinusoidalEmbeddings(nn.Module): 
  def __init__(self, time_steps: int, embed_dim: int):
    super().__init__()
    position = torch.arange(time_steps).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * - (math.log(10000.0) / embed_dim))
    embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)

    embeddings[:, 0::2] = torch.sin(position * div) # for even positions
    embeddings[:, 1::2] = torch.cos(position * div) # for odd positions

    self.embeddings = embeddings

  def forward(self, x: torch.Tensor, t) -> torch.Tensor:
    embeddings = self.embeddings[t].to(x.device)
    return embeddings[:, :, None] # (N, C, L)