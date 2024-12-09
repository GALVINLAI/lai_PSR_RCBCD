import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg
import os, shutil
from tqdm import trange
from lai_utils import is_unitary, is_hermitian
from lai_utils import generate_random_unitary, generate_random_hermitian, generate_random_H_paulis
from lai_utils import create_uniform_superposed_state, create_ket_zero_state
from lai_utils import U_j

'''
Solve a random toy problem using three algorithms.
This code only uses numpy, not jax.numpy.
'''

###################################################### algorithms

# Block Coordinate Descent method
def block_coordinate_descent(V_list, H_list, M, input_state, theta_init, 
                             sigma=0.01,
                             max_iter=10, 
                             plot_subproblem=False, 
                             PRINT_INFO=False):
    theta = np.array(theta_init)
    m = len(theta)
    
    # Record function value changes
    # Record initial state as iteration 0, so all methods start from the same point in the plot
    f_values = [exact_cost(theta, V_list, H_list, M, input_state)]
    iterations = [0]
    selected_coordinates = [None]
    theta_values = [theta.copy()]
    
    if plot_subproblem:
        # Delete and recreate the folder to save plot results
        output_dir = 'bcd/bcd_vqa_toy_plots'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    t = trange(max_iter, leave=True)
    for k in t:
        theta_old = theta.copy()
        
        # Randomly select an index
        j = random.choice(range(m))

        # Calculate coefficients a, b, c
        theta_copy = theta.copy()
        theta_copy[j] = 0
        exact_hat_f1 = exact_cost(theta_copy, V_list, H_list, M, input_state)
        hat_f1 = exact_hat_f1 + np.random.randn() * sigma
        
        theta_copy = theta.copy()
        theta_copy[j] = np.pi / 2
        exact_hat_f2 = exact_cost(theta_copy, V_list, H_list, M, input_state)
        hat_f2 = exact_hat_f2 + np.random.randn() * sigma
        
        theta_copy = theta.copy()
        theta_copy[j] = np.pi
        exact_hat_f3 = exact_cost(theta_copy, V_list, H_list, M, input_state)
        hat_f3 = exact_hat_f3 + np.random.randn() * sigma
        
        a = (hat_f1 + hat_f3) / 2
        b = (hat_f1 - hat_f3) / 2
        c = hat_f2 - a
        
        # The following decision only considers finding the minimum point
        if np.isclose(b,0) and np.isclose(c,0):
            # Constant function. Any value is a minimum point, so no change.
            pass
        elif np.isclose(b,0) and not np.isclose(c,0):
            # sin function, minimum point affected by amplitude c
            theta[j] = (3/2) * np.pi if c > 0 else (1/2) * np.pi
        elif not np.isclose(b,0) and np.isclose(c,0):
            # cos function, minimum point affected by amplitude b
            theta[j] = np.pi if b > 0 else np.zeros(1)
        else:
            # Both b and c are non-zero, which is more complex.
            # Obtain the analytical solution for coordinate j
            _theta_star = np.arctan(c / b)
            theta[j] = _theta_star
            exact_hat_f_theta_star = exact_cost(theta, V_list, H_list, M, input_state)
            hat_f_theta_star = exact_hat_f_theta_star + np.random.randn() * sigma
            IS_MINIMIZER = hat_f_theta_star < a
            IS_POSITIVE =  _theta_star > 0 
            if IS_POSITIVE and IS_MINIMIZER:
                pass
            elif IS_POSITIVE and not IS_MINIMIZER:
                theta[j] += np.pi
            elif not IS_POSITIVE and IS_MINIMIZER:
                theta[j] += 2*np.pi
            elif not IS_POSITIVE and not IS_MINIMIZER:
                theta[j] += np.pi

        # Record current iteration information
        current_f_value = exact_cost(theta, V_list, H_list, M, input_state)
        f_values.append(current_f_value)
        iterations.append(k+1)
        selected_coordinates.append(j)
        theta_values.append(theta.copy())
        
        message = f"Iteration: {k+1}, Value: {current_f_value}"
        t.set_description("[BCD] Progress: %s" % message)
        t.refresh()  # to show immediately the update

        if PRINT_INFO:
            print(f"[BCD] Iteration {k+1}: cost values = {current_f_value}, selected coordinate = {j}, theta = {theta}")

        #############################################################################
        if plot_subproblem:
            # The following plots the single_variable_function and saves the function image
            # Used to verify the correctness of the theory
            def single_variable_function(theta_j):
                theta_var = theta_old.copy()
                theta_var[j] = theta_j
                return exact_cost(theta_var, V_list, H_list, M, input_state)
              
            def plot_single_variable_function():
                x_range = 2 * np.pi
                x = np.linspace(0, x_range, 1000)
                y = np.array([single_variable_function(value) for value in x])
                old_theta_j = theta_old[j]
                new_theta_j = theta[j]
                theta_j_x = np.array([old_theta_j, new_theta_j])
                theta_j_y = np.array([single_variable_function(value) for value in theta_j_x])
                # Plot the function image
                plt.plot(x, y, label=r'f(theta_j)')
                # Add the first point "old theta j"
                plt.scatter(theta_j_x[0], theta_j_y[0], color='red', s=100, label='Old theta_j')
                # Add the second point "new theta j"
                plt.scatter(theta_j_x[1], theta_j_y[1], color='blue', s=100, label='New theta_j')
                # Set legend
                plt.legend()
                # Add labels and title
                plt.xlabel(r'theta_j')
                plt.ylabel(r'f(theta_j)')
                plt.title(f'Iter # {k+1}, Chosen Coord j: {j}')
                plt.grid(True)
                # Save the image
                plt.savefig(f'bcd/bcd_vqa_toy_plots/iter_{k+1}_cor_{j}.png')
                plt.close()

            plot_single_variable_function()

    # Return final theta and recorded data
    return theta, f_values, iterations, selected_coordinates, theta_values

# Random Coordinate Descent method (PSR)
def random_coordinate_descent(V_list, H_list, M, input_state, theta_init,
                              alpha=0.1, decay_step=30, decay_rate=-1.0, decay_threshold=1e-4,
                              sigma=0.01,
                              max_iter=100,
                              CHECK_GRADIENT=False,
                              PRINT_INFO=False):
    theta = np.array(theta_init)
    m = len(theta)
    
    # Record function value changes
    f_values = [exact_cost(theta, V_list, H_list, M, input_state)]
    iterations = [0]
    selected_coordinates = [None]
    theta_values = [theta.copy()]
    
    t = trange(max_iter, leave=True)
    for k in t:
        j = random.choice(range(m))
        
        # Decay step size
        if decay_rate > 0 and (k + 1) % decay_step == 0:
            alpha *= decay_rate
            alpha = max(alpha, decay_threshold)

        # Use Parameter-shift Rule to compute partial derivative, where s = np.pi / 2
        theta_copy = theta.copy()
        theta_copy[j] += np.pi / 2
        f_forward = exact_cost(theta_copy, V_list, H_list, M, input_state) + np.random.randn() * sigma
        
        theta_copy = theta.copy()
        theta_copy[j] -= np.pi / 2
        f_backward = exact_cost(theta_copy, V_list, H_list, M, input_state) + np.random.randn() * sigma
        
        ps_gradient = (f_forward - f_backward) / 2

        # Move along the negative gradient
        theta[j] -= alpha * ps_gradient   
        
        # Record current iteration information
        current_f_value = exact_cost(theta, V_list, H_list, M, input_state)
        f_values.append(current_f_value)
        iterations.append(k+1)
        selected_coordinates.append(j)
        theta_values.append(theta.copy())
        
        message = f"Iteration: {k+1}, Value: {current_f_value}"
        t.set_description("[RCD] Progress: %s" % message)
        t.refresh()  # to show immediately the update

        if PRINT_INFO:
            print(f"[RCD] Iteration {k+1}: cost values = {current_f_value}, selected coordinate = {j}, theta = {theta}")

    # Return final theta and recorded data
    return theta, f_values, iterations, selected_coordinates, theta_values

# Gradient Descent method (PSR)
def gradient_descent(V_list, H_list, M, input_state, theta_init,
                     alpha=0.01,
                     sigma=0.01,
                     max_iter=100,
                     PRINT_INFO=False):
    theta = np.array(theta_init)
    m = len(theta)
    
    # Record function value changes
    f_values = [exact_cost(theta, V_list, H_list, M, input_state)]
    iterations = [0]
    theta_values = [theta.copy()]
    
    t = trange(max_iter, leave=True)
    for k in t:
        full_gradient = np.zeros(m)
        
        # Calculate gradients for all coordinates
        for j in range(m):
            theta_copy = theta.copy()
            theta_copy[j] += np.pi / 2
            f_forward = exact_cost(theta_copy, V_list, H_list, M, input_state) + np.random.randn() * sigma
            
            theta_copy = theta.copy()
            theta_copy[j] -= np.pi / 2
            f_backward = exact_cost(theta_copy, V_list, H_list, M, input_state) + np.random.randn() * sigma
            
            full_gradient[j] = (f_forward - f_backward) / 2
        
        # Update all parameters along the negative gradient direction
        theta -= alpha * full_gradient
        
        # Record current iteration information
        current_f_value = exact_cost(theta, V_list, H_list, M, input_state)
        f_values.append(current_f_value)
        iterations.append(k)
        theta_values.append(theta.copy())
        
        message = f"Iteration: {k+1}, Value: %s" % current_f_value
        t.set_description("[GD] Progress: %s" % message)
        t.refresh()  # to show immediately the update

        if PRINT_INFO:
            print(f"[GD] Iteration {k+1}: cost values = {current_f_value}, theta = {theta}")

    # Return final theta and recorded data
    return theta, f_values, iterations, theta_values


###################################################### problem setup

# Pauli-X matrix
X = np.array([[0, 1], [1, 0]], dtype=complex)

# Pauli-Y matrix
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

# Pauli-Z matrix
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Identity matrix
I = np.array([[1, 0], [0, 1]], dtype=complex)

PAULIS = [X, Y, Z, I]

def generate_random_pauli(n):
    """
    Generate a random n-qubit Hermitian matrix by Kronecker product of Pauli matrices.
    """
    H = PAULIS[random.choice(range(4))]
    for _ in range(1, n):
        H = np.kron(H, PAULIS[random.choice(range(4))])
    return H

def generate_random_unitary(dim):
    """
    Generate a random dim x dim unitary matrix.
    """
    random_matrix = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    q, r = np.linalg.qr(random_matrix)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph
    return q

def U_j(theta_j, H, method):
    """
    Calculate the matrix U_j(theta_j) based on the specified method. (rotation-like gates)
    - 'exponential': e^{-(i / 2) H_j \theta_j}
    - 'trigonometric': cos(theta_j / 2) * I - i * sin(theta_j / 2) * H
    """
    if method == 'exponential':
        exponent = -1j / 2 * H * theta_j
        return scipy.linalg.expm(exponent)
    elif method == 'trigonometric':
        return np.cos(theta_j / 2) * np.eye(H.shape[0]) - 1j * np.sin(theta_j / 2) * H
    else:
        raise ValueError("Unknown method: choose 'exponential' or 'trigonometric'")

# Define the objective function
def exact_cost(theta, V_list, H_list, M, input_state):
    U_theta = np.eye(V_list[0].shape[0])
    for i in range(len(V_list)):
        U_theta = V_list[i] @ U_j(theta[i], H_list[i], 'trigonometric') @ U_theta
    U_theta_dagger = U_theta.conjugate().T
    return np.real(np.vdot(input_state, U_theta_dagger @ M @ U_theta @ input_state))

# Example initialization
n = 4  # Number of qubits
m = 20  # Number of parameters
theta_init = np.random.rand(m)  # Initial parameters
input_state = np.eye(2**n)[:, 0] # |0> state as input

# Generate random V and H matrices
# np.random.seed(42)

# Ensure the matrices in V_list are random unitary matrices by using the QR decomposition method.
# Generate a random matrix and then perform QR decomposition, using the Q matrix as the unitary matrix.
V_list = [generate_random_unitary(2**n) for _ in range(m)]
H_list = [generate_random_pauli(n) for _ in range(m)]

# Define the M matrix (observable)
# M = np.random.rand(2**n, 2**n)
M = np.random.rand(2**n, 2**n) + 1j * np.random.rand(2**n, 2**n)
M = (M + M.conjugate().T) / 2  # Make it Hermitian

###################################################### solve problem

MAX_ITER = 500
SIGMA = 0.01

# Block Coordinate Descent optimization BCD
theta_opt_bcd, f_values_bcd, iterations_bcd, selected_coordinates_bcd, theta_values_bcd = block_coordinate_descent(
    V_list, H_list, M, input_state, theta_init, 
    sigma=SIGMA,
    max_iter=MAX_ITER,
    plot_subproblem=False,
    PRINT_INFO=False
)

# Random Coordinate Descent optimization RCD
theta_opt_rcd, f_values_rcd, iterations_rcd, selected_coordinates_rcd, theta_values_rcd = random_coordinate_descent(
    V_list, H_list, M, input_state, theta_init,
    alpha=0.01, decay_step=30, decay_rate=-1, decay_threshold=1e-4,
    sigma=SIGMA,
    max_iter=MAX_ITER,
    CHECK_GRADIENT=False,
    PRINT_INFO=False
)

# Gradient Descent optimization GD
theta_opt_gd, f_values_gd, iterations_gd, theta_values_gd = gradient_descent(
    V_list, H_list, M, input_state, theta_init, 
    alpha=0.01, 
    sigma=SIGMA,
    max_iter=MAX_ITER,
    PRINT_INFO=False
)

###################################################### plot results

# Create a figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot the first subplot: compare function value changes of the three algorithms
axs[0].plot(iterations_bcd, f_values_bcd, label='BCD')
axs[0].plot(iterations_rcd, f_values_rcd, label='RCD')
axs[0].plot(iterations_gd, f_values_gd, label='GD')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Function Value')
axs[0].set_title('Function Value over Iterations')
axs[0].legend()
axs[0].grid(True)

# Plot the second subplot: compare function value changes of the three algorithms (by function evaluation counts)
num_fun_eval_bcd = np.arange(len(iterations_bcd)) * 3
num_fun_eval_rcd = np.arange(len(iterations_rcd)) * 2
num_fun_eval_gd = np.arange(len(iterations_gd)) * 2 * m
axs[1].plot(num_fun_eval_bcd, f_values_bcd, label='BCD')
axs[1].plot(num_fun_eval_rcd, f_values_rcd, label='RCD')
axs[1].plot(num_fun_eval_gd, f_values_gd, label='GD')
axs[1].set_xlabel('Number of function evaluations')
axs[1].set_ylabel('Function Value')
axs[1].set_title('Function Value over # function evaluations')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(0, min(num_fun_eval_bcd[-1], num_fun_eval_rcd[-1], num_fun_eval_gd[-1]))

# Adjust layout and display the image
plt.tight_layout()
plt.show()
plt.close()
