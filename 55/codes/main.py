import numpy as np
import matplotlib.pyplot as plt

# Number of simulations and defining the values on cards
num_simulations = 100000
cards = [1, 2, 3, 4, 5]

# Initialize array to store simulated values of X
X_simulated = []

# Simulate the scenario and calculate X for each simulation
for _ in range(num_simulations):
    cards = np.arange(1, 6)
    np.random.shuffle(cards)
    selected_cards = np.random.choice(cards, size=2, replace=False)
    X_simulated.append(np.sum(selected_cards))

X_simulated = np.array(X_simulated)

# Calculate the simulated mean and variance
simulated_mean = np.mean(X_simulated)
simulated_variance = np.var(X_simulated)

# Create all possible combinations of two cards without replacement
combinations = np.array(np.meshgrid(cards, cards)).T.reshape(-1, 2)
combinations = combinations[combinations[:, 0] != combinations[:, 1]]

# Calculate the PMF using NumPy
p_X_theoretical, _ = np.histogram(combinations.sum(axis=1), bins=np.arange(2.5, 10.5), density=True)

# Values of X range from 3 to 9
k_values = np.array(range(3, 10))
hist_bins = np.arange(2.5, 10.5)

# Plot the PMFs as a stem plot
plt.stem(k_values, p_X_theoretical, linefmt='b-', markerfmt='bo', basefmt=' ', label='X_theoretical', use_line_collection=True)
plt.stem(k_values, np.histogram(X_simulated, bins=hist_bins, density=True)[0], linefmt='r-', markerfmt='ro', basefmt=' ', label='X_simulated', use_line_collection=True)
plt.xlabel('X')
plt.ylabel('Probability')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.savefig('../figs/main.png')
plt.show()

print("Theoretical Mean:", np.dot(k_values, p_X_theoretical))
print("Simulated Mean:", simulated_mean)
print("Theoretical Variance:", np.dot(k_values**2, p_X_theoretical) - (np.dot(k_values, p_X_theoretical))**2)
print("Simulated Variance:", simulated_variance)

