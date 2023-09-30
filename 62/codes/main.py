import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

samples = 10000
x = sp.symbols('x')
prob = (1 / (2 * sp.sqrt(3)))

f_x = prob * sp.Piecewise((1, (x >= -sp.sqrt(3)) & (x <= sp.sqrt(3))), (0, True))
theo_var = sp.integrate(x**2 * f_x, (x, -sp.oo, sp.oo))

sim_samples = np.random.uniform(-np.sqrt(3), np.sqrt(3),size=samples)
sim_var = np.var(sim_samples)

print("Theoretical variance: ",theo_var)
print("Simulated Variance: ",sim_var)
