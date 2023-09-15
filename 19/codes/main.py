import random
import numpy as np

# Number of simulations
num_simulations = 100000

# Initialize an array to store the sum of two card numbers for each simulation
sums = np.zeros(num_simulations)

# Simulate the scenario
for i in range(num_simulations):
    # Create a deck of cards with numbers 1 to 5
    deck = [1, 2, 3, 4, 5]
    
    # Draw two cards without replacement
    cards_drawn = random.sample(deck, 2)
    
    # Calculate the sum of the two card numbers
    sum_of_cards = sum(cards_drawn)
    
    # Store the sum in the array
    sums[i] = sum_of_cards

# Calculate the simulated mean and variance
simulated_mean = np.mean(sums)
simulated_variance = np.var(sums)

# Theoretical mean and variance (calculated previously)
theoretical_mean = 6
theoretical_variance = 3

# Print the results
print("Simulated Mean:", simulated_mean)
print("Theoretical Mean:", theoretical_mean)
print("Simulated Variance:", simulated_variance)
print("Theoretical Variance:", theoretical_variance)

