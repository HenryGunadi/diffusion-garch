import torch
import torch.nn as nn

def reverse(xT: torch.Tensor, T: int, betas: torch.Tensor, posterior_betas: torch.Tensor, alpha_bars: torch.Tensor, model: nn.Module):
  xt_prev = xT

  with torch.no_grad():
    for t in reversed(range(0, T)):
      if t > 0:
        z = torch.randn_like(xt_prev)
      else:
        z = torch.zeros_like(xt_prev)

      t_tensor = torch.full((xt_prev.size()[0],), t, device=xt_prev.device, dtype=torch.long)
      alpha_t = (1 - betas[t_tensor])[:, None, None]
      alpha_t_bar = (alpha_bars[t_tensor])[:, None, None]
      epsilon_t = model(xt_prev, t_tensor)
      std_t = (torch.sqrt(posterior_betas[t_tensor]))[:, None, None]

      xt_prev = (1 / torch.sqrt(alpha_t)) * (xt_prev - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)) * epsilon_t) + (std_t * z)
    
    x0 = xt_prev
    return x0