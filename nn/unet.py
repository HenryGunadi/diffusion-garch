import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal
from utils import crop_image

class Res1DBlock(nn.Module):
  def __init__(self, n_conv, in_channels: int, out_channels: int):
    super().__init__()
    
    self.block = self.initialize_block(n_conv, in_channels, out_channels)
    
  def initialize_block(self, n_conv, in_channels, out_channels):
    layers = []

    for i in range(n_conv):
      """
        The convolution parameters follow the original U-net architecture
      """
      layers.append(
        nn.Conv1d(
          in_channels=in_channels if i == 0 else out_channels,
          out_channels=out_channels,
          kernel_size=3,
          stride=1,
          padding=0
        )
      )
      layers.append(nn.GroupNorm(num_groups=32, num_channels=out_channels))
      
      if i != n_conv - 1:
        break

      layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
  
  def forward(self, x):
    
    x = self.block(x)


    return x

class EncoderBlock(nn.Module):
  def __init__(self, in_channels: List[int], n_downsampling: int, out_channels: List[int], n_conv=2):
    super().__init__()

    self.n_downsampling = n_downsampling
    self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    self.conv_blocks = nn.ModuleList(
      [
        Res1DBlock(
          in_channels=in_channels[i],
          out_channels=out_channels[i],
          n_conv=n_conv
        ) 
        for i in range(n_downsampling)
      ]
    )

  def forward(self, x):
    skipped_con = []

    for i in range(self.n_downsampling):
      x = self.conv_blocks[i](x)
      skipped_con.append(x) # we store every convolution output for the residual connections
      x = self.max_pool(x)

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