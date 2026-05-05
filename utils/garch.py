from arch import arch_model
import numpy as np

def simulate_garch_from_windows(windows, mean="zero", p=1, q=1, burn=500):
  """
  windows: list/array of shape (n_windows, window_length)

  Returns:
      sim_data: list of simulated windows (same length as input windows)

  Description:
  - Fits GARCH(1,1) on each window
  - Simulates new data using estimated parameters
  - Includes burn-in to remove initialization bias
  """

  sim_data = []

  for window in windows:
    model = arch_model(window, mean=mean, vol="GARCH", p=p, q=q)
    res = model.fit(disp="off")

    params = res.params.values

    sim_model = arch_model(None, mean=mean, vol="GARCH", p=p, q=q)
    sim = sim_model.simulate(params, nobs=len(window), burn=burn)

    sim_series = sim["data"].values
    sim_data.append(sim_series)

  return sim_data