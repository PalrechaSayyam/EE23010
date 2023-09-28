import numpy as np
import matplotlib.pyplot as plt

sets = 1000
samples = 10000

sim_vars = []

theo_var = 1 

for _ in range(sets):
    sim_samples = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=samples)
    sim_var = np.var(sim_samples)
    sim_vars.append(sim_var)

theo_vars = [theo_var] * sets

# Create the scatter plot
yRange = [0.85, 1.15]
plt.figure(figsize=(10, 6))
plt.scatter(range(sets), sim_vars, label='Simulated Variance', color='b', marker='o', s=10)
plt.plot(range(sets), theo_vars, label='Theoretical Variance', linestyle='--', color='r')
plt.xlabel('Simulation Set')
plt.ylabel('Variance')
plt.title('Simulated vs. Theoretical Variance of X')
plt.ylim(yRange[0], yRange[1])
plt.legend()
plt.grid(True)
plt.savefig('../figs/main.png')
plt.show()

