import numpy as np
import matplotlib.pyplot as plt

def cdf(x):
    if x < -1:
        return 0
    elif x >= -1 and x < 0:
        return (1/4) * (x + 1)
    elif x >= 0 and x < 1:
        return (1/4) * (x + 3)
    elif x >= 1:
        return 1

# Open the file in read mode
with open("des_dist.dat", "r") as file:
    # Read the lines from the file
    lines = file.readlines()

# Process the lines as needed
X = [float(line.strip()) for line in lines]
X = np.array(X)

num_sim = len(X)
n = 100000000

lb = -1/2 - 1/n
ub = 1/n

desired = X[(X > lb) & (X < ub)]

sim_prob = len(desired)/len(X)
act_prob = 5/8

if abs(sim_prob - act_prob) < 0.005:
    print("Statement (B) is true.")
else:
    print("Statement (B) is not true.")

print(f"Calculated probability: {sim_prob}")

x_values = np.linspace(-1, 1, num_sim)

theo_cdf = [cdf(x) for x in x_values]
sorted_X = np.sort(X)
sim_cdf = np.arange(1, num_sim+1) / num_sim

plt.plot(sorted_X, sim_cdf, label='Simulated CDF', marker='o', linestyle='-', markersize=3)
plt.plot(x_values, theo_cdf, label='Theoretical CDF', linestyle='--')

plt.legend()
plt.xlabel('X')
plt.ylabel('CDF')
plt.title('Simulated vs Theoretical CDF')
plt.savefig('../figs/main.png')
plt.grid(True)
plt.show()
