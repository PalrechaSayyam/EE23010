import numpy as np

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

num_sim = 1000000

prob_zero = pdf(0)

prob_rem = 1 - prob_zero

# Generate random values strictly between -1 and 1, excluding 0
sample_one = np.random.uniform(-1, 0, size=int(num_sim * prob_rem/2))
sample_two = np.random.uniform(0, 1, size=int(num_sim * prob_rem/2))
sample = np.concatenate((sample_one, sample_two))

zeros = np.zeros(int(num_sim * prob_zero))

X = np.concatenate((sample, zeros))
np.random.shuffle(X)

n = 1000000

lb = -1/2 - 1/n
ub = 1/n

desired = X[(X > lb) & (X < ub)]

sim_prob = len(desired)/len(X)
act_prob = 5/8

if abs(sim_prob - act_prob) <= 0.001:
    print("Statement (B) is true.")
else:
    print("Statement (B) is not true.")

print(f"Calculated probability: {sim_prob}")
