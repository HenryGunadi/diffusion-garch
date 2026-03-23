import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal
from utils import crop_image, normalize, attn_block
from .pe import SinusoidalEmbeddings

class Res1DBlock(nn.Module):
  """
    p = dropout rate
    T = number of diffusion steps
  """

  def __init__(self, in_ch: int, out_ch: int, T: int, p=0.5):
    super().__init__()
    
    self.conv1d_1 = nn.Conv1d(
      in_ch=in_ch,
      out_ch=out_ch,
      kernel_size=3,
      stride=1,
      padding=1
    )
    self.conv1d_2 = nn.Conv1d(
      in_ch=out_ch,
      out_ch=out_ch,
      kernel_size=3,
      stride=1,
      padding=1
    )
    
    self.skip = nn.Identity()
    if in_ch != out_ch:
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)

    self.time_embed = SinusoidalEmbeddings(time_steps=T, embed_dim=out_ch)
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

    return x + x2

class EncoderBlock(nn.Module):
  """
    paramaters:
    - n_res_block: number of residual blocks
    - attn_res: resolution level to apply attention mechanism
  """

  def __init__(self, in_ch: List[int], out_ch: List[int], n_res_block: int, T: int, attn_res: int, p=0.5, num_heads: int = 4):
    super().__init__()
    self.attn_res = attn_res
    self.down_sammple = nn.Conv1d(stride=2)
    self.out_chs = out_ch
    self.in_chs = in_ch
    self.n_res_block = n_res_block
    self.num_heads = num_heads
    self.T = T
    self.p = p

  def forward(self, x: torch.Tensor, t):
    skipped_con = []

    for i in range(len(self.out_chs)):
      for j in range(self.n_res_block):
        x = Res1DBlock(in_ch=self.in_chs[i] if j == 0 else self.out_chs[i], out_ch=self.out_chs[i], T=self.T, p=self.p)(x, t)

        # apply attention mechanism between res blocks
        if x.size()[-1] == self.attn_res and j != self.n_res_block - 1:
          x = attn_block(out_channels=self.out_chs[i], x=x, num_heads=self.num_heads)

        skipped_con.append(x)

      x = self.max_pool(x)

    return x, skipped_con

class DecoderBlock(nn.Module):
  """
    paramaters:
    - n_res_block: number of residual blocks
    - attn_res: resolution level to apply attention mechanism
  """
  def __init__(self, in_ch: List[int], out_ch: List[int], n_res_block: int, T: int, attn_res: int, p=0.5, num_heads: int = 4):
    super().__init__()
    self.up_convs = nn.ModuleList( # halves feature maps, increase spatial size
      [
        nn.ConvTranspose1d(
          in_ch=in_ch[i],
          out_ch=in_ch[i],
          kernel_size=2,
          stride=2,
          padding=0
        )
        for i in range(len(out_ch))
      ]
    )
    self.out_ch = out_ch
    self.attn_res = attn_res

  def forward(self, x, t: int, skips: List):
    for i, _ in enumerate(self.out_ch):
      x = self.up_convs[i](x)

      # u-net skip connections
      # cropped = crop_image(skips[len(self.out_ch) - i - 1], x)
      # skip = skips[len(self.out_ch) - i - 1]
      skip = skips.pop()
      print("skip u-net : ", skip.size())
      print("x before concat : ", x.size())

      x = torch.cat((x, skip), dim=1)

      print("x after concat : ", x.size())

      if x.size()[2] == self.attn_res:
        x, _ = self.res_levels[i](x, t, True, False) # applied attention mechanism
      else:
        x, _ = self.res_levels[i](x, t, False, False)

    return x
    
class BottleNeck(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, n_res_block: int, T: int, p=0.5, num_heads: int = 4):
    super().__init__()
    self.res_blocks = nn.ModuleList(
      Res1DBlock(
        in_ch=in_ch,
        out_ch=out_ch,
        T=T,
        p=p
      )
      for _ in range(n_res_block)
    )
    self.out_ch = out_ch
    self.num_heads = num_heads

  def forward(self, x: torch.Tensor, t: int):
    for i in range(len(self.res_blocks)):
      res_block = self.res_blocks[i]
      x = res_block(x, t)

      # applied attentio mechanism
      if i != len(self.res_blocks) - 1:
        x = attn_block(out_ch=self.out_ch, x=x, num_heads=self.num_heads)

    return x
  
class Unet1D(nn.Module):
  """
    Make sure to provide the correct out_ch & in_ch values

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
      num_classes: int = 1, # since we're only observing log returns
      num_heads: int = 4
    ):
    super().__init__()

    self.encoder_block = EncoderBlock(
      attn_res=attn_res,
      in_ch=encoder_in_channels,
      out_ch=encoder_out_channels,
      n_res_block=n_res_block,
      T=T,
      p=p,
      num_heads=num_heads
    )
    self.bottleneck = BottleNeck(
      in_ch=encoder_out_channels[-1],
      out_ch=encoder_out_channels[-1],
      n_res_block=n_res_block,
      T=T,
      p=p,
      num_heads=num_heads
    )
    self.decoder_block = DecoderBlock(
      attn_res=attn_res,
      out_ch=decoder_out_channels,
      in_ch=decoder_in_channels,
      n_res_block=n_res_block,
      T=T,
      p=p,
      num_heads=num_heads
    )
    self.output_layer = nn.Conv1d(
      in_ch=decoder_out_channels[-1],
      out_ch=num_classes,
      kernel_size=1
    )
    
  def forward(self, x: torch.Tensor, t: torch.Tensor):
    assert isinstance(t, torch.Tensor), "Argument t must be in torch.Tensor type."

    # encoding process
    x, skipped_con = self.encoder_block(x, t)
    print("passed encoder x : ", x.size())

    # bottleneck
    x = self.bottleneck(x, t)
    print("passed bottleneck x : ", x.size())

    # decoding process
    x = self.decoder_block(x, t, skipped_con)
    print("passed decoder")

    x = self.output_layer(x)
    print("passed output layer")

    return x