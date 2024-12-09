import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

'''
This script simulates the estimation of the parameter 𝜃* 
using random variables and plots the results over multiple runs.
'''

# Setting the parameters for the random variables
shots = 10000  # Number of samples on a quantum computer
variance0 = 10  # some constant; it is sigma_{0}^2
std_dev = np.sqrt(variance0 / shots)  # Variance for the normal distribution

# Defining the number of iterations
num_runs = 20

# Creating subplots
fig, axes = plt.subplots(4, 5, figsize=(20, 16))  # Modify to a 4x5 subplot matrix

for i in range(num_runs):
    # Setting the expected values of the random variables
    # Corresponding to f(0), f(pi/2), f(pi)
    eaxct_f_val_1, eaxct_f_val_2, eaxct_f_val_3 = np.random.rand(3)
    
    # Generating samples of the random variables
    N = 10000  # Number of samples
    X1 = np.random.normal(eaxct_f_val_1, std_dev, N)
    X2 = np.random.normal(eaxct_f_val_2, std_dev, N)
    X3 = np.random.normal(eaxct_f_val_3, std_dev, N)

    # Calculating the new random variable THETA_star
    a = (X1 + X3) / 2
    b = (X1 - X3) / 2
    c = X2 - a
    THETA_star = np.arctan(c / b)

    # Calculating the position of the vertical line
    exact_theta_star = np.arctan((2 * eaxct_f_val_2 - eaxct_f_val_1 - eaxct_f_val_3) / (eaxct_f_val_1 - eaxct_f_val_3))

    # Plotting the approximate probability density function of THETA_star on the subplot
    ax = axes[i // 5, i % 5]
    ax.hist(THETA_star, bins=100, density=True, alpha=0.6, color='g', label='Empirical PDF')

    # Using kernel density estimation to plot the smoothed PDF
    kde = gaussian_kde(THETA_star)
    x_vals = np.linspace(min(THETA_star), max(THETA_star), 1000)
    ax.plot(x_vals, kde(x_vals), color='r', label='KDE')

    # Adding a vertical line
    ax.axvline(x=exact_theta_star, color='b', linestyle='--', label=f'theta* = {exact_theta_star:.2f}')

    # Adding the expected values as annotations
    caption = f"f(0) = {eaxct_f_val_1:.2f}\nf(pi/2) = {eaxct_f_val_2:.2f}\nf(pi) = {eaxct_f_val_3:.2f}"
    ax.text(0.05, 0.95, caption, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    ax.set_xlabel('Estimated_theta*')
    ax.set_ylabel('Density')
    ax.set_title(f'Run {i+1}')
    ax.legend()

plt.tight_layout()
plt.show()
