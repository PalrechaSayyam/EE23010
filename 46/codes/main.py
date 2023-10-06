import numpy as np
import random 

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

num_sim = 100000

samples = np.random.choice([np.random.uniform(-1,1),np.random.uniform(-1,1)], size=num_sim)
total = np.array([pdf(t) for t in samples])

X = np.random.choice([np.random.uniform(-2,-1),np.random.uniform(1,2)], size=num_sim)

for i, value in enumerate(total):
    if value == 1/4:
        x_value = np.random.choice([np.random.uniform(-1,0),np.random.uniform(0,1)], size=1)
        
        while x_value == 0:
            x_value = np.random.choice([np.random.uniform(-1,0),np.random.uniform(0,1)], size=1)
            
        X[i] = x_value
        
    elif value == 1/2:
        X[i] = 0

n = 100000

lb = -1/2 - 1/n
ub = 1/n

desired = X[(X > lb) & (X < ub)]

prob = len(desired)/len(X)

if abs(prob - 5/8) < 0.001:
    print("Statement (B) is true.")
else:
    print("Statement (B) is not true.")

print(f"Calculated probability: {prob}")

