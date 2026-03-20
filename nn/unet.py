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
      kernel_size=1,
      stride=1,
      padding=0
    )
    self.conv1d_2 = nn.Conv1d(
      in_channels=out_channels,
      out_channels=out_channels,
      kernel_size=1,
      stride=1,
      padding=0
    )

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

    x = self.conv1d_1(x)

    return x + x2

class ResLevel(nn.Module):
  def __init__(self, res_block: int, in_channels: int, out_channels: int, T: int, p=0.5):
    super().__init__()

    self.res_blocks = nn.ModuleList(
      Res1DBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        T=T,
        p=p
      )
      for _ in range(res_block)
    )

  def forward(self, x, t):
    for res_block in self.res_blocks:
      x = res_block(x, t)

    return x

class EncoderBlock(nn.Module):
  def __init__(self, in_channels: List[int], n_downsampling: int, out_channels: List[int], res_block: int, T: int, res_levels: int = 4, p=0.5):
    super().__init__()

    self.n_downsampling = n_downsampling
    self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    self.res_block = res_block

    self.res_levels = nn.ModuleList(
      ResLevel(
        res_block=res_block,
        in_channels=in_channels[i],
        out_channels=out_channels[i],
        T=T,
        p=p
      )
      for i in range(res_levels)
    )
    self.attention = nn.MultiheadAttention(
      embed_dim=out_channels[len(out_channels) // 2],
      num_heads=4 if out_channels[len(out_channels) // 2] % 2 == 0 else 3,
      batch_first=True # (N, C, L) -> (N, L, C) 
    )

  def forward(self, x):
    skipped_con = []

    for i in range(self.n_downsampling):
      x = self.conv_blocks[i](x)
      skipped_con.append(x)
      x = self.max_pool(x)
      
      if i == (self.n_downsampling // 2) - 1:
        # x = 
        pass

    return x, skipped_con

class DecoderBlock(nn.Module):
  def __init__(self, n_upsampling: int, in_channels: List[int], out_channels: List[int], n_conv=2):
    super().__init__()

    self.n_upsampling = n_upsampling

    self.up_convs = nn.ModuleList( # halves feature maps, increase spatial size
      [
        nn.ConvTranspose1d(
          in_channels=in_channels[i],
          out_channels=out_channels[i],
          kernel_size=2,
          stride=2,
          padding=0
        )
        for i in range(n_upsampling)
      ]
    )
    
    self.conv_blocks = nn.ModuleList(
      [
        Res1DBlock(
          in_channels=in_channels[i],
          out_channels=out_channels[i],
          n_conv=n_conv
        ) 
        for i in range(n_upsampling)
      ]
    )

  def forward(self, x, skips):
    for i in range(self.n_upsampling):
      x = self.up_convs[i](x)
      
      # skip connections
      cropped = crop_image(skips[self.n_upsampling - i - 1], x)
      x = torch.cat((x, cropped), dim=1)

      x = self.conv_blocks[i](x)

    return x
    
class BottleNeck(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, n_conv):
    super().__init__()

    self.conv_block = Res1DBlock(
      in_channels=in_channels,
      out_channels=out_channels,
      n_conv=n_conv
    )

  def forward(self, x):
    x = self.conv_block(x)

    return x
  
class Unet1D(nn.Module):
  """
    Make sure to provide the correct out_channels & in_channels values
  """
  def __init__(
      self,
      n_sampling: int,
      n_conv: int,
      encoder_out_channels: List[int],
      decoder_in_channels: List[int],
      decoder_out_channels: List[int],
      encoder_in_channels: List[int],
      num_classes: int = 1 # since we're only observing log returns
    ):
    super().__init__()

    self.encoder_block = EncoderBlock(
      n_downsampling=n_sampling,
      in_channels=encoder_in_channels,
      out_channels=encoder_out_channels,
      n_conv=n_conv
    )
    self.bottleneck = BottleNeck(
      in_channels=encoder_out_channels[-1],
      out_channels=encoder_out_channels[-1],
      n_conv=n_conv,
    )
    self.decoder_block = DecoderBlock(
      out_channels=decoder_out_channels,
      in_channels=decoder_in_channels,
      n_upsampling=n_sampling,
      n_conv=n_conv,
    )
    self.output_layer = nn.Conv1d(
      in_channels=decoder_out_channels[-1],
      out_channels=num_classes,
      kernel_size=1
    )
    
  def forward(self, x):
    # encoding process
    x, skipped_con = self.encoder_block(x)

    # bottleneck
    x = self.bottleneck(x)

    # decoding process
    x = self.decoder_block(x, skipped_con)

    x = self.output_layer(x)

    return x