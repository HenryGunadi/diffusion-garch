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
      in_channels=in_ch,
      out_channels=out_ch,
      kernel_size=3,
      stride=1,
      padding=1
    )
    self.conv1d_2 = nn.Conv1d(
      in_channels=out_ch,
      out_channels=out_ch,
      kernel_size=3,
      stride=1,
      padding=1
    )
    
    self.skip = nn.Identity()
    if in_ch != out_ch:
        self.skip = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

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

  def __init__(self, in_chs: List[int], out_chs: List[int], n_res_block: int, T: int, attn_res: int, p=0.5, num_heads: int = 4):
    super().__init__()
    self.attn_res = attn_res
    self.out_chs = out_chs
    self.n_res_block = n_res_block
    self.num_heads = num_heads

    self.downsamples = nn.ModuleList([
      nn.Conv1d(
          in_channels=self.out_chs[i],
          out_channels=self.out_chs[i],
          kernel_size=3,
          stride=2,
          padding=1
      )
      for i in range(len(self.out_chs))
    ])
    self.res_blocks = nn.ModuleDict({
      f"res_{i}": nn.ModuleList([
        Res1DBlock(in_ch=in_chs[i] if j == 0 else self.out_chs[i], out_ch=self.out_chs[i], T=T, p=p)
        for j in range(n_res_block)
      ])
      for i in range(len(self.out_chs))
    })

  def forward(self, x: torch.Tensor, t):
    skipped_con = {}

    for idx, (key, res_blocks) in enumerate(self.res_blocks.items()):
      skip_con = []

      for i, res_block in enumerate(res_blocks):
        x = res_block(x, t)

        if x.size()[-1] == self.attn_res and i != self.n_res_block - 1:
          x = attn_block(out_channels=self.out_chs[idx], x=x, num_heads=self.num_heads)

        skip_con.append(x)

      skipped_con[key] = skip_con
      x = self.downsamples[idx](x)

    return x, skipped_con

class DecoderBlock(nn.Module):
  """
    paramaters:
    - n_res_block: number of residual blocks
    - attn_res: resolution level to apply attention mechanism
  """
  def __init__(self, in_chs: List[int], out_chs: List[int], n_res_block: int, T: int, attn_res: int, p=0.5, num_heads: int = 4):
    super().__init__()
    self.out_chs = out_chs
    self.up_convs = nn.ModuleList( # halves feature maps, increase spatial size
      [
        nn.ConvTranspose1d(
          in_channels=in_chs[i],
          out_channels=in_chs[i],
          kernel_size=4,
          stride=2,
          padding=1
        )
        for i in range(len(out_chs))
      ]
    )
    self.res_blocks = nn.ModuleDict({
      f"res_{len(self.out_chs) - i - 1}": nn.ModuleList([
        Res1DBlock(in_ch=in_chs[i] * 2 if j == 0 else (out_chs[i] + in_chs[i]), out_ch=self.out_chs[i], T=T, p=p)
        for j in range(n_res_block)
      ])
      for i in range(len(self.out_chs))
    })
    self.attn_res = attn_res
    self.n_res_block = n_res_block
    self.num_heads = num_heads

  def forward(self, x: torch.Tensor, t: int, skips: nn.ModuleDict):
    for idx, (key, res_blocks) in enumerate(self.res_blocks.items()):
      skip_cons = skips[key].copy()
      x = self.up_convs[idx](x)

      for i, res_block in enumerate(res_blocks):
        skip = skip_cons.pop()
        x = torch.cat((x, skip), dim=1)

        x = res_block(x, t)

        if x.size()[-1] == self.attn_res and i != self.n_res_block - 1:
          x = attn_block(out_channels=self.out_chs[idx], x=x, num_heads=self.num_heads)

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
        x = attn_block(out_channels=self.out_ch, x=x, num_heads=self.num_heads)

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
      in_chs=encoder_in_channels,
      out_chs=encoder_out_channels,
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
      out_chs=decoder_out_channels,
      in_chs=decoder_in_channels,
      n_res_block=n_res_block,
      T=T,
      p=p,
      num_heads=num_heads
    )
    self.output_layer = nn.Conv1d(
      in_channels=decoder_out_channels[-1],
      out_channels=num_classes,
      kernel_size=1
    )
    
  def forward(self, x: torch.Tensor, t: torch.Tensor):
    assert isinstance(t, torch.Tensor), "Argument t must be in torch.Tensor type."

    # encoding process
    x, skipped_con = self.encoder_block(x, t)

    # bottleneck
    x = self.bottleneck(x, t)

    # decoding process
    x = self.decoder_block(x, t, skipped_con)

    x = self.output_layer(x)

    return x