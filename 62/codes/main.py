import numpy as np
import matplotlib.pyplot as plt

samples = 10000

theo_var = 1 

sim_samples = np.random.uniform(-np.sqrt(3), np.sqrt(3),size=samples)
sim_var = np.var(sim_samples)

print("Theoretical variance: ",theo_var)
print("Simulated Variance: ",sim_var)