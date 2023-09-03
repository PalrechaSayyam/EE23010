import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

#Probability of the event of getting same number on both dice, i.e., Z=0 is 1/6
prob = 1/6
size = int(1/prob)

#Number of samples is 10^(size-1), 0 denotes same number on dice
simlen=int(100000)

#Generating sample date using Bernoulli r.v.
data_bern = bernoulli.rvs(size=simlen,p=prob)

#Calculating the number of favourable outcomes-simulation
#err_ind = np.nonzero(data_bern == 1)
#calculating the probability
#err_n = np.size(err_ind)/simlen
#Different numbers on both dice
#Using simulation
#b_sim=simlen*(1-err_n)
#Using exact probability
#b_act=simlen*(1-prob)
#same number on both dice
#Using simulation
#a_sim=simlen-b_sim
#Using exact probability
#a_act=simlen-b_act
#Theory vs simulation
#print("Probability-simulation,actual (X=1):",err_n,prob)
#print("Probability-simulation,actual (X=0):",1-err_n,1-prob)
#print("Different numbers on both dice-simulation,actual:",b_sim,b_act)
#print("Different numbers on both dice-simulation,actual:",a_sim,a_act)

print("Samples generated:",data_bern)

def generate_pmf(size):
	array = np.full(size, prob)
	return array

pmf_X = generate_pmf(size)
pmf_Y = generate_pmf(size)
pmf_Z = np.convolve(pmf_X, pmf_Y, mode='full')

# Calculate the actual probability of getting the same number on both dice (Z = 0)
act_prob_same_no = pmf_Z[5]  # Probability at Z = 0
# Calculate the actual probability of getting different numbers on both dice (|Z| > 0)
act_prob_diff_nos = 1 - act_prob_same_no

# Simulate the random variables X and Y by rolling two dice
num_samples = simlen
X_simulated = np.random.randint(1, size + 1, num_samples)
Y_simulated = np.random.randint(1, size + 1, num_samples)

# Calculate Z for each pair of X and Y
Z_simulated = X_simulated - Y_simulated

# Calculate the simulated probability of getting the same number on both dice (Z = 0)
sim_prob_same_no = np.sum(Z_simulated == 0) / num_samples

# Calculate the simulated probability of getting different numbers on both dice (|Z| > 0)
sim_prob_diff_nos = np.sum(Z_simulated != 0) / num_samples

print("Actual Probability (Z = 0):", act_prob_same_no)
print("Simulated Probability (Z = 0):", sim_prob_same_no)

if np.isclose(act_prob_same_no, act_prob_same_no):
    print("Hence verified for same number on both dice")
else:
    print("error")

print("Actual Probability (Z ≠ 0):", act_prob_diff_nos)
print("Simulated Probability (Z ≠ 0):", sim_prob_diff_nos)

if np.isclose(act_prob_diff_nos, act_prob_diff_nos):
    print("Hence verified for different numbers on both dice")
else:
    print("error")
    
k_values = np.arange(-size + 1, size)
plt.stem(k_values, pmf_Z)
plt.xlabel("k")
plt.ylabel("Probability")
plt.title("Theoretical PMF of Z")
plt.grid(True)
plt.savefig('/home/sayyam/EE23010/ncertExemplar/figs/main_act.png')
plt.show()

# Plot the simulated PMF of Z
plt.hist(Z_simulated, bins=np.arange(-size + 0.5, size + 0.5, 1), density=True, rwidth=0.8, alpha=0.75)
plt.xlabel("Z")
plt.ylabel("Probability Density")
plt.title("Simulated PMF of Z")
plt.grid(True)
plt.savefig('/home/sayyam/EE23010/ncertExemplar/figs/main_sim.png')
plt.show()

