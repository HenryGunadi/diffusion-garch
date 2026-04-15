from arch import arch_model
import numpy as np

def simulate_garch(omega, alpha, beta, n_data, n_windows, mean="zero", p=1, q=1):
  sim_data = []
  params = np.array([omega, alpha, beta])

  for _ in range(n_windows):
    model = arch_model(
        None,
        mean=mean,
        vol="GARCH",
        p=p,
        q=q
    )
    sim = model.simulate(params, nobs=n_data)
    sim_data.append(sim)

  return sim_data  
