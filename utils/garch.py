from arch import arch_model
import numpy as np

def simulate_garch(res, n_windows, window_length, burn=500):
  sim_data = []

  params = res.params

  model = res.model

  for _ in range(n_windows):
    sim = model.simulate(params, nobs=window_length, burn=burn)
    sim = sim / 100
    sim_series = sim["data"].values
    
    sim_data.append(sim_series)

  return np.array(sim_data)