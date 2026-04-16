# reverse_time = time.time()

# scaled_synthetic_data = reverse(
#   xT=xT,
#   T=T,
#   betas=betas,
#   posterior_betas=posterior_betas,
#   alpha_bars=alpha_hats,
#   model=model_v0
# ).squeeze(1).detach().numpy()

# reverse_time = time.time() - reverse_time
# print(f"Reverse process duration : { reverse_time:.2f} seconds")

# # inverse scaling
# synthetic_data = inverse_standard(scaled_synthetic_data, train_snp500)
# synthetic_data[:1]

# SYN_PATH = dir / "data" / "syn_data_256.joblib"
# joblib.dump(synthetic_data, SYN_PATH)