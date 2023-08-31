import numpy as np
import matplotlib.pyplot as plt
import torch

# Continuous beta function
def beta(t, linear_start=1e-4, linear_end=2e-2, t_end=1):
    """ 
    t is a normalized time ranging from 0 to 1. 
    If t=0, it returns linear_start.
    If t=1, it returns linear_end.
    Any value between 0 and 1 will interpolate accordingly.
    """
    intermediate = (linear_start ** 0.5) * (1 - t) + (linear_end ** 0.5) * t
    return intermediate ** 2

# Discretized schedule function
def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    return betas.numpy()

# Generate data for plotting
t_values = np.linspace(0, 1, 1000)
continuous_beta_values = np.array([beta(t) for t in t_values])
discrete_beta_values = make_beta_schedule()

difference = continuous_beta_values - discrete_beta_values
print(difference[:15])

# Plotting
'''
plt.plot(t_values, continuous_beta_values, label='Continuous Beta(t)', color='blue')
plt.plot(t_values, discrete_beta_values, label='Discrete Beta Schedule', color='red', linestyle='--')
plt.legend()
plt.xlabel("Normalized Time")
plt.ylabel("Beta Value")
plt.title("Continuous vs Discrete Beta Schedule")
plt.grid(True)
plt.show()
'''
