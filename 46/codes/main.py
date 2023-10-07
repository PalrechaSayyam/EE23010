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

def pdf(u):
    if u < -1:
        return 0
    elif u >= -1 and u < 0:
        return 1/4
    elif u == 0:
        return 1/2
    elif u > 0 and u < 1:
        return 1/4
    elif u >= 1:
        return 0

num_sim = 2000

prob_zero = pdf(0)

prob_rem = 1 - prob_zero

# Generate random values strictly between -1 and 1, excluding 0
sample_one = np.random.uniform(-1, 0, size=int(num_sim * prob_rem/2))
sample_two = np.random.uniform(0, 1, size=int(num_sim * prob_rem/2))
sample = np.concatenate((sample_one, sample_two))

zeros = np.zeros(int(num_sim * prob_zero))

X = np.concatenate((sample, zeros))
np.random.shuffle(X)

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

x_values = np.linspace(-1, 1, 1000)

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
