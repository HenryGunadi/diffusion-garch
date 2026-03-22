import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal
from utils import crop_image, normalize
from .pe import SinusoidalEmbeddings

class Res1DBlock(nn.Module):
  """
    p = dropout rate
    T = number of diffusion steps
  """

  def __init__(self, in_channels: int, out_channels: int, T: int, p=0.5):
    super().__init__()
    
    self.conv1d_1 = nn.Conv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=1,
      padding=0
    )
    self.conv1d_2 = nn.Conv1d(
      in_channels=out_channels,
      out_channels=out_channels,
      kernel_size=3,
      stride=1,
      padding=0
    )
    self.skip = nn.Identity()
    if in_channels != out_channels:
        self.skip = nn.Conv1d(in_channels, out_channels, 1)

    self.time_embed = SinusoidalEmbeddings(time_steps=T, embed_dim=out_channels)
    self.activation = nn.SiLU(inplace=True)
    self.dropout = nn.Dropout1d(p=p)
    
  def forward(self, x, t):
    x2 = self.activation(normalize(x))
    x2 = self.conv1d_1(x2)
    x2 += self.time_embed(x2, t)
    
    x2 = self.activation(normalize(x2))
    x2 = self.dropout(x2)
    x2 = self.conv1d_2(x2)

    x = self.skip(x)

    return x + x2 # residual connection

class ResLevel(nn.Module):
  def __init__(self, n_res_block: int, in_channels: int, out_channels: int, T: int, p=0.5):
    num_heads = 4
    assert out_channels % num_heads == 0, "Out_channels must be divisible by 4 (num_heads in MultiheadAttention block)"

    super().__init__()

    self.res_blocks = nn.ModuleList(
      Res1DBlock(
        in_channels=in_channels if i == 0 else out_channels,
        out_channels=out_channels,
        T=T,
        p=p
      )
      for i in range(n_res_block)
    )
    self.attention = nn.MultiheadAttention(
      embed_dim=out_channels,
      num_heads=num_heads,
      batch_first=True # expect (N, L, C) dim size 
    )

  def forward(self, x: torch.Tensor, t, attention: bool = False):
    for res_block in self.res_blocks:
      x = res_block(x, t)

    if attention is True:
      x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
      x, _ = self.attention(x, x, x)
      x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)
      
    return x

class EncoderBlock(nn.Module):
  """
    paramaters:
    - n_res_block: number of residual blocks
    - attn_res: resolution level to apply attention mechanism
  """

  def __init__(self, in_channels: List[int], out_channels: List[int], n_res_block: int, T: int, attn_res: int, p=0.5):
    super().__init__()
    self.attn_res = attn_res
    self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    self.res_levels = nn.ModuleList(
      ResLevel(
        n_res_block=n_res_block,
        in_channels=in_channels[i],
        out_channels=out_channels[i],
        T=T,
        p=p
      )
      for i in range(len(out_channels))
    )
    self.out_channels = out_channels

  def forward(self, x: torch.Tensor, t):
    skipped_con = []

    for i, _ in enumerate(self.out_channels):
      if x.size()[2] == self.attn_res: # applied attention mechanism
        x = self.res_levels[i](x, t, True)
      else:
        x = self.res_levels[i](x, t)

      skipped_con.append(x)
      x = self.max_pool(x)

    return x, skipped_con

class DecoderBlock(nn.Module):
  """
    paramaters:
    - n_res_block: number of residual blocks
    - attn_res: resolution level to apply attention mechanism
  """
  def __init__(self, in_channels: List[int], out_channels: List[int], n_res_block: int, T: int, attn_res: int, p=0.5):
    super().__init__()
    self.up_convs = nn.ModuleList( # halves feature maps, increase spatial size
      [
        nn.ConvTranspose1d(
          in_channels=in_channels[i],
          out_channels=out_channels[i],
          kernel_size=2,
          stride=2,
          padding=0
        )
        for i in range(len(out_channels))
      ]
    )
    self.res_levels = nn.ModuleList(
      ResLevel(
        n_res_block=n_res_block,
        in_channels=in_channels[i],
        out_channels=out_channels[i],
        T=T,
        p=p
      )
      for i in range(len(out_channels))
    )
    self.out_channels = out_channels
    self.attn_res = attn_res

  def forward(self, x, t: int, skips):
    for i, _ in enumerate(self.out_channels):
      x = self.up_convs[i](x)
      
      # u-net skip connections
      cropped = crop_image(skips[len(self.out_channels) - i - 1], x)
      x = torch.cat((x, cropped), dim=1)

      if x.size()[2] == self.attn_res:
        x = self.res_levels[i](x, t, True) # applied attention mechanism
      else:
        x = self.res_levels[i](x, t)

    return x
    
class BottleNeck(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, n_res_block: int, T: int, p=0.5):
    super().__init__()

    self.res_level = ResLevel(
      n_res_block=n_res_block,
      in_channels=in_channels,
      out_channels=out_channels,
      T=T,
      p=p
    )

  def forward(self, x, t):
    x = self.res_level(x, t, True)

    return x
  
class Unet1D(nn.Module):
  """
    Make sure to provide the correct out_channels & in_channels values

    paramaters:
    - n_res_block: number of residual blocks
    - attn_res: resolution level to apply attention mechanism
  """
  def __init__(
      self,
      attn_res: int,
      n_res_block: int,
      encoder_out_channels: List[int],
      decoder_in_channels: List[int],
      decoder_out_channels: List[int],
      encoder_in_channels: List[int],
      T:int,
      p=0.5,
      num_classes: int = 1 # since we're only observing log returns
    ):
    super().__init__()

    self.encoder_block = EncoderBlock(
      attn_res=attn_res,
      in_channels=encoder_in_channels,
      out_channels=encoder_out_channels,
      n_res_block=n_res_block,
      T=T,
      p=p
    )
    self.bottleneck = BottleNeck(
      in_channels=encoder_out_channels[-1],
      out_channels=encoder_out_channels[-1],
      n_res_block=n_res_block,
      T=T,
      p=p
    )
    self.decoder_block = DecoderBlock(
      attn_res=attn_res,
      out_channels=decoder_out_channels,
      in_channels=decoder_in_channels,
      n_res_block=n_res_block,
      T=T,
      p=p
    )
    self.output_layer = nn.Conv1d(
      in_channels=decoder_out_channels[-1],
      out_channels=num_classes,
      kernel_size=1
    )
    
  def forward(self, x: torch.Tensor, t: int):
    assert isinstance(t, int), "Argument t must be an int"

    # encoding process
    x, skipped_con = self.encoder_block(x, t)

    # bottleneck
    x = self.bottleneck(x, t)

    # decoding process
    x = self.decoder_block(x, t, skipped_con)

    x = self.output_layer(x)

    return x