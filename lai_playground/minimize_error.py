import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Lists to store the values for plotting
objective_values = []
singular_values_list = []
condition_numbers = []

def create_matrix(x1, x2, x3):
    return np.array([
        [1, np.cos(x1), np.sin(x1)],
        [1, np.cos(x2), np.sin(x2)],
        [1, np.cos(x3), np.sin(x3)]
    ])

def objective(x):
    A = create_matrix(x[0], x[1], x[2])
    return np.trace(np.linalg.inv(A.T @ A))

def callback(xk):
    A = create_matrix(xk[0], xk[1], xk[2])
    singular_values = np.linalg.svd(A, compute_uv=False)
    objective_value = objective(xk)
    condition_number = np.linalg.cond(A)
    
    # Store the values
    objective_values.append(objective_value)
    singular_values_list.append(singular_values)
    condition_numbers.append(condition_number)
    
    # Print current values
    # print(f"Current x: {xk}")
    # print(f"Objective function value: {objective_value}")
    # print(f"Singular values of A: {singular_values}")
    # print(f"Condition number of A: {condition_number}\n")

# Initial guess for x1, x2, x3
x0 = np.random.uniform(0, 2 * np.pi, 3)
# x0 =  [0.0, 2 * np.pi/ 3 , 4 * np.pi/ 3 ]

# List of solvers to use
# solvers = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'CG', 'BFGS']
solvers = ['BFGS']
for solver in solvers:
    print('=' * 150)
    print(f"Solver: {solver}")
    if solver in ['L-BFGS-B', 'TNC', 'CG', 'BFGS']:
        result = minimize(objective, x0, method=solver, callback=callback)
    else:
        result = minimize(objective, x0, method=solver)
    optimal_x = result.x
    min_trace = result.fun
    A_optimal = create_matrix(optimal_x[0], optimal_x[1], optimal_x[2])
    cond_number = np.linalg.cond(A_optimal)
    
    print(f"Optimal x: {optimal_x}")
    print(f"Minimum trace of (A^T A)^-1: {min_trace}")
    print(f"Condition number of A: {cond_number}\n")

# Plotting the data
iterations = np.arange(1, len(objective_values) + 1)
plt.figure(figsize=(12, 4))

# Plot objective function values
plt.subplot(1, 3, 1)
plt.plot(iterations, objective_values, marker='o', linestyle='-')
plt.title('Objective Function Value per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')

# Plot singular values
plt.subplot(1, 3, 2)
for idx in range(len(singular_values_list[0])):
    plt.plot(iterations, [sv[idx] for sv in singular_values_list], marker='o', linestyle='-', label=f'Singular Value {idx+1}')
plt.title('Singular Values of A per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Singular Values')
plt.legend()

# Plot condition numbers
plt.subplot(1, 3, 3)
plt.plot(iterations, condition_numbers, marker='o', linestyle='-')
plt.title('Condition Number of A per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Condition Number')

plt.tight_layout()
plt.show()
