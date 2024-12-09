import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

def interp_matrix(theta_vals):
    # create interpolation matrix
    return np.array([[1, np.cos(val), np.sin(val)] for val in theta_vals])

def objective(theta_vals, x):
    A = interp_matrix(theta_vals)
    Var = np.linalg.inv(A.T @ A)
    part = Var[1:3, 1:3]  # Fixed to select the correct submatrix
    v = np.array([-np.sin(x), np.cos(x)])
    return np.dot(np.dot(v.T, part), v)  # Fixed the order of np.dot for correct matrix multiplication

def objective2(theta_vals, x):
    A = interp_matrix(theta_vals)
    Var = np.linalg.inv(A.T)
    t = np.array([0, -np.sin(x), np.cos(x)])
    At = np.dot(Var, t)
    return np.dot(At, At)  # Fixed the order of np.dot for correct matrix multiplication


# Example usage
# np.random.seed(4)
x = np.random.uniform(0, 2 * np.pi, size=1)[0]
x = 0
# Define the bounds for theta_vals
bounds = [(0, 2 * np.pi) for _ in range(3)]

# Initial guess for theta_vals
# initial_theta_vals = np.random.uniform(0, 2 * np.pi, size=3)
initial_theta_vals = [0, np.pi*2/3, np.pi*4/3]

# # List of optimization methods
# methods = ['TNC', 'SLSQP', 'Nelder-Mead', 'Powell']

# # Optimize using different methods and print results
# for method in methods:
#     result = minimize(objective2, initial_theta_vals, args=(x,), bounds=bounds, method=method)
#     print(f"Method: {method}")
#     print("Optimized theta_vals:", result.x)
#     print("Minimum value of the objective function:", result.fun)
#     print("Optimization success:", result.success)
#     print("Optimization message:", result.message)
#     print()

# Global optimization using differential evolution
result_de = differential_evolution(objective2, bounds, args=(x,))
print("Global Optimization Method: Differential Evolution")
print("Optimized theta_vals:", result_de.x)
print("Minimum value of the objective function:", result_de.fun)
print("Optimization success:", result_de.success)
print("Optimization message:", result_de.message)
print()

# Global optimization using differential evolution
result_de = dual_annealing(objective2, bounds, args=(x,))
print("Global Optimization Method: dual_annealing")
print("Optimized theta_vals:", result_de.x)
print("Minimum value of the objective function:", result_de.fun)
print("Optimization success:", result_de.success)
print("Optimization message:", result_de.message)
print()