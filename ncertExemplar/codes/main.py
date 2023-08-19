import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

#Number of samples is 36, 1 denotes same number on dice and 0 denotes different numbers on dice
simlen=int(36)

#Probability of the event of getting same number on both dice, i.e., X=1 is 1/6
prob = 1/6

#Generating sample date using Bernoulli r.v.
data_bern = bernoulli.rvs(size=simlen,p=prob)
#Calculating the number of favourable outcomes-simulation
err_ind = np.nonzero(data_bern == 1)
#calculating the probability
err_n = np.size(err_ind)/simlen
#Different numbers on both dice
#Using simulation
b_sim=simlen*(1-err_n)
#Using exact probability
b_act=simlen*(1-prob)
#same number on both dice
#Using simulation
a_sim=simlen-b_sim
#Using exact probability
a_act=simlen-b_act
#Theory vs simulation
print("Probability-simulation,actual (X=1):",err_n,prob)
print("Probability-simulation,actual (X=0):",1-err_n,1-prob)
print("Different numbers on both dice-simulation,actual:",b_sim,b_act)
print("Different numbers on both dice-simulation,actual:",a_sim,a_act)
print("Samples generated:",data_bern)
