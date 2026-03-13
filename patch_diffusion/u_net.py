import torch
import torch.nn as nn
from typing import List, Literal
import numpy as np

class Conv1DBlock(nn.Module):
  """
    Make sure to provide the correct out_channels values in a list
  """

  def __init__(self, n_conv, in_channels, out_channels: List[int]):
    super().__init__()
    
    self.block = self.initialize_block(n_conv, in_channels, out_channels)
    
  def initialize_block(self, n_conv, in_channels, out_channels: List[int]):
    layers = []

    for i in range(n_conv):
      layers.append(
        nn.Conv1d(
          in_channels=in_channels if i == 0 else out_channels[i-1],
          out_channels=out_channels[i],
          kernel_size=3,
          padding=1,
          stride=1
        )
      )
      layers.append(nn.BatchNorm1d(num_features=out_channels[i]))

    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
  
  def forward(self, x):
    x = self.block(x)
    return x

class EncoderBlock(nn.Module):
  def __init__(self, in_channels: int, n_downsampling: int, out_channels: List[List[int]], n_conv=2):
    super().__init__()

    self.n_downsampling = n_downsampling
    self.max_pool = nn.MaxPool1d(kernel_size=2)
    self.conv_blocks = [
      Conv1DBlock(
        in_channels=in_channels if i == 0 else out_channels[i - 1][-1],
        out_channels=out_channels[i],
        n_conv=n_conv
      ) 
      for i in range(n_downsampling)
    ]

    self.stored_tensors = []
    
  def forward(self, x):
    for i in range(self.n_downsampling):
      x = self.conv_blocks[i](x)
      self.stored_tensors.append(x) # we store every convolution output for the concatenation process
      x = self.max_pool(x)

    return x

class DecoderBlock(nn.Module):
  def __init__(self, n_upsampling: int, in_channels: int, out_channels: List[List[int]], n_conv=2):
    super().__init__()

    self.n_upsampling = n_upsampling
    self.up_conv = nn.ConvTranspose1d(kernel_size=2)
    self.conv_blocks = [
      Conv1DBlock(
        in_channels=in_channels if i == 0 else out_channels[i - 1][-1],
        out_channels=out_channels[i],
        n_conv=n_conv
      ) 
      for i in range(n_upsampling)
    ]

  def forward(self, x):
    for i in range(self.n_upsampling):
      x = self.up_conv(x)
      x = self.conv_blocks[i](x)

    return x
  
class BottleNeck(nn.Module):
  def __init__(self, in_channels, out_channels: List[int], n_conv):
    super().__init__()

    self.conv_block = Conv1DBlock(
      in_channels=in_channels,
      out_channels=out_channels,
      n_conv=n_conv
    )

  def forward(self, x):
    x = self.conv_block(x)

    return x
  
class Unet1D(nn.Module):
  def __init__(
      self,
      n_sampling: int,
      n_conv: int,
      encoder_out_channels: List[List[int]],
      decoder_in_channels: int,
      decoder_out_channels: List[List[int]],
      bottleneck_out_channels: List[List[int]],
      encoder_in_channels: int = 1,
    ):
    super().__init__()

    self.encoder_block = EncoderBlock(
      n_downsampling=n_sampling,
      out_channels=encoder_out_channels,
      in_channels=encoder_in_channels,
      n_conv=n_conv
    )
    self.decoder_out_channels = decoder_out_channels 
    self.bottleneck_out_channels = bottleneck_out_channels
    self.n_conv = n_conv
    self.n_sampling = n_sampling,

  def forward(self, x):
    # encoding process
    x = self.encoder_block(x)

    self.bottleneck = BottleNeck(
      in_channels=x.size()[1],
      out_channels=self.bottleneck_out_channels,
      n_conv=self.n_conv,
    )

    x = self.bottleneck(x)
    
    self.decoder_block = DecoderBlock(
      out_channels=self.decoder_out_channels,
      in_channels=x.size()[1],
      n_upsampling=self.n_sampling,
      n_conv=self.n_conv,
    )

    x = self.decoder_block(x)

    return x
  
def crop_image(original, expected):
    """
      (N, C, L) -> dimensions
      
      Since we're dealing with 1-feature time series data -> 1D
    """

    original_dim = original.size()[-1]
    expected_dim = expected.size()[-1]

    difference = original_dim - expected_dim
    padding = difference // 2

    cropped = original[:, :, padding:original_dim-padding]

    return cropped
# class ConvBlock():
#   @staticmethod
#   def initialized_block(in_channels, out_channels, k, p, s, scale, mode = "nearest", type: Literal["down", "up"] = "down"):
#     type = type.lower()

#     if type == "down":
#       return nn.Sequential(
#         nn.Conv1d(
#           in_channels=in_channels,
#           out_channels=out_channels,
#           kernel_size=k,
#           padding=p,
#           stride=2
#         ),
#         nn.ReLU(),
#         nn.MaxPool1d(kernel_size=2, stride=2),
#       )
#     elif type == "up":
#       return nn.Sequential(
#         nn.ConvTranspose1d(
#           in_channels=in_channels,
#           out_channels=out_channels,
#           kernel_size=k,
#           padding=p,
#           stride=2
#         ),
#         nn.ReLU(),
#         nn.Upsample()
#       )
#     else:
#       raise ValueError("Invalid type arguments")

# class EncoderBlock(nn.Module):
#   def __init__(self, in_channels: int, out_channels: List[int], depth = 4):
#     super().__init__()
#     self.conv_blocks = [conv_block(in_channels, out_channels[i]) for i in range(depth)]

# class DecoderBlock(nn.Module):
#   def __init__(self, in_channels: int, out_channels: List[int], depth = 4):
#     super().__init__()
#     self.conv_blocks = [conv_block(in_channels, out_channels[i]) for i in range(depth)]

class UnetEncoder(nn.Module):
  def __init__(self, in_channels: int, out_channels: List[int]):
    super().__init__()

    # self.conv1 = conv_block(
    #   in_channels=in_channels,
    #   out_channels=out_channels[0]
    # )
    # self.conv1 = conv_block(
    #   in_channels=
    # )

# x = torch.arange(1, 5, dtype=torch.float32).view(1, 2, 2)
# print(x)
# print(x.size())
# print(type(x))

# m = nn.Upsample(scale_factor=3, mode="nearest")
# res = m(x)
# print(res)
# print(res.size())

# m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# in_channels = 2
# kernel = 2
# stride = 2
# padding = 1

# data = torch.arange(1, 100, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# d = nn.Conv1d(
#   in_channels= 1,
#   out_channels= 2,
#   padding=padding,
#   stride=stride,
#   kernel_size=kernel
# )

# print("Before downsampling : ", data.size())
# res_down = d(data)
# print("Downsampling res : ", res_down)
# print("Downsampling res size : ", res_down.size())
# print("Channels : ", res_down.size()[1])

# u = nn.ConvTranspose1d(
#   in_channels=int(res_down.size()[1]),
#   out_channels=int(res_down.size()[1] / 2),
#   padding=padding,
#   stride=stride,
#   kernel_size=kernel
# )

# res_up = u(res_down)
# print("Upsampling res : ", res_up)
# print("Upsampling res size : ", res_up.size())
# print("Channels : ", res_up.size()[1])

# # print("Size : ", data.size())
# # print(input.size())

# # x = torch.randn(1, 3, 2, 2)
# # print(x)

# b = torch.zeros(2, 3, 2, 3)
# print(b.size()[-1])
a = torch.randint(0, 100, size=(3, 2))
b = torch.randint(0, 100, size=(2, 2))
print(a)
print(torch.concatenate((a, b)))


