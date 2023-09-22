import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# Parameters for the distribution
n = 4
p = 0.5
q = 1 - p

# Create an array of possible values for Y
y_values = np.arange(0, n + 1)

# Calculate the binomial PMF for Y
binomial_pmf = binom.pmf(y_values, n, p)

# Calculate the normal PDF for Y using the mean and variance of the binomial distribution
mean = n * p
variance = n * p * q
normal_pdf = norm.pdf(y_values, loc=mean, scale=np.sqrt(variance))

# Plot the results
plt.figure(figsize=(8, 6))
plt.stem(y_values, binomial_pmf, markerfmt='bo', linefmt='b-', basefmt=' ', label='Binomial PMF')
plt.plot(y_values, normal_pdf, marker='o', linestyle='-', color='r', label='Normal PDF')
plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Stem Plot of Binomial PMF and Plot of Normal PDF')
plt.xticks(y_values)
plt.legend()
plt.grid(True)
plt.savefig('../figs/main.png')
plt.show()

