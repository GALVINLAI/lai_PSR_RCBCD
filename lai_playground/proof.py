import numpy as np

def D_star(x, R):
    terms = [1/(2*R), 1/(2*R) * np.cos(R * x)] + [1/R * np.cos(l * x) for l in range(1, R)]
    return sum(terms)

def tilde_D_mu(x, x_mu, R):
    D_star_diff = D_star(x - x_mu, R) - D_star(x + x_mu, R)
    expected_diff = (1/R) * np.cos(x_mu) * (0.5 * np.sin(R * x) + sum(np.sin(l * x) for l in range(1, R)))
    return D_star_diff, expected_diff

# Parameters
R = 5
x = np.linspace(-np.pi, np.pi, 3)
x = np.random.rand()
mu = 3
x_mu = (2 * mu - 1) / (2 * R) * np.pi

# Calculate the differences
D_star_diff_values, expected_diff_values = tilde_D_mu(x, x_mu, R)

# Return the values to verify
print(np.linalg.norm(D_star_diff_values - expected_diff_values))
