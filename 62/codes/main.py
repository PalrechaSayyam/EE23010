import numpy as np
import matplotlib.pyplot as plt

# Number of samples
samples = 10000

samples_sim = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=samples)

sim_var = np.var(samples_sim)
theo_var = 1  

x_values = np.linspace(-np.sqrt(3), np.sqrt(3), num=1000)
simulated_variance = [sim_var if -np.sqrt(3) <= x <= np.sqrt(3) else 0 for x in x_values]
theoretical_variance = [theo_var if -np.sqrt(3) <= x <= np.sqrt(3) else 0 for x in x_values]

# Create the plot
plt.figure(figsize=(9, 6))
plt.plot(x_values, simulated_variance, label='Simulated Variance', color='g')
plt.plot(x_values, theoretical_variance, label='Theoretical Variance', linestyle='--', color='r')
plt.xlabel('X')
plt.ylabel('Variance')
plt.ylim(0, 1.5) 
plt.title('Simulated vs. Theoretical Variance of X')
plt.legend()
plt.grid(True)
plt.savefig('../figs/main.png')
plt.show()

print(f"Simulated Variance: {sim_var}")
print(f"Theoretical Variance: {theo_var}")

