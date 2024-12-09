import numpy as np
import jax  # jax is used for automatic differentiation and numerical computation
'''
JAX Library
JAX is a library for numerical computation that provides powerful automatic differentiation capabilities,
allowing for the computation of gradients, Hessian matrices, and more.
Additionally, JAX supports GPU and TPU acceleration, and its JIT compilation can significantly improve computation speed.
'''
import jax.random as jrd
from tqdm import trange  # tqdm is used for displaying progress bars.

def gradient_descent(f, initial_point, learning_rate, num_iterations, sigma, key):
    """
    Minimize the function f using the gradient descent algorithm.
    
    Args:
        f (Callable[[Any], float]): The function to minimize, which takes a parameter and returns a float.
        initial_point (Any): The starting point, matching the parameter type of the function f.
        learning_rate (float): The learning rate used to update parameters.
        num_iterations (int): The number of iterations for gradient descent.
        sigma (float): The standard deviation of the random noise added to the gradient.
        key (jax.random.PRNGKey): The random number generator key for JAX.
        skip_hessian (bool, optional): Whether to skip the computation of the Hessian matrix. Default is False.
    
    Returns:
        tuple: A tuple containing the following five elements:
            - function_values (List[float]): A list of the function f's values after each iteration.
            - x (Any): The final parameter values after gradient descent, matching the parameter type of f.
            - function_value (float): The final value of the function f after gradient descent.
            - eigen_values (List[np.ndarray]): A list of eigenvalues of the Hessian matrix (only returned if skip_hessian is False).
            - lip_diag_values (List[np.ndarray]): A list of diagonal elements of the Hessian matrix (only returned if skip_hessian is False).
    """
    # jax.grad(f) returns a function that computes the gradient of f.
    # jax.hessian(f) returns a function that computes the Hessian matrix of f.
    # jax.jit compiles these computation functions to improve their runtime efficiency.
    grad_f = jax.jit(jax.grad(f))
    # JIT compilation is a dynamic compilation technique that compiles code to machine code during execution, improving runtime speed.
    # JAX provides the jax.jit function to JIT compile computation processes.

    x = initial_point

    # Print a separator before the progress bar
    print("-"*100)

    # Create a progress bar to display the iterations
    t = trange(num_iterations, desc="Bar desc", leave=True)

    function_values = [f(initial_point)]

    # Perform gradient descent
    for i in t:
        # Generate a new random key
        # Use jax.random.split to generate a new random key to ensure the independence of random numbers generated in each iteration.
        key, subkey = jrd.split(key)

        # Generate the random noise
        # Use jax.random.split to generate a new random key to ensure the independence of random numbers generated in each iteration.
        noise = jrd.normal(subkey, x.shape) * sigma

        # Compute the gradient of f at the current value of x
        gradient = grad_f(x) + noise
    
        # Update x using the gradient and the learning rate
        x = x + learning_rate * gradient

        function_value = f(x)
        function_values.append(function_value)
        message = f"Iteration: {i}, Value: {function_value}"
             
        # Update the progress bar
        t.set_description("[GD] Processing %s" % message)
        t.refresh()  # to show the update immediately

    # Print the final value of x
    # print("[GD] Final value of x:", x)
    print("[GD] Final value of f:", function_value)

    return x, function_value, function_values
