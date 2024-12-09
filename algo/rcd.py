import numpy as np
import jax
import jax.random as jrd
from tqdm import trange

def random_coordinate_descent(f, initial_point, alpha, num_iterations, sigma, key,
                              decay_step=30, 
                              decay_rate=-1.0, 
                              decay_threshold=1e-4):
    """
    Random Coordinate Descent algorithm for optimizing a function f using automatic differentiation.

    Parameters:
        - f: function to optimize
        - initial_point: starting point for the algorithm
        - alpha: step size for the algorithm
        - num_iterations: number of iterations to run the algorithm for

    Returns:
        - best_point: the point that achieves the minimum value of f found during the algorithm
        - best_value: the minimum value of f found during the algorithm
    """
    grad_f = jax.jit(jax.grad(f))

    current_point = initial_point
    best_point = current_point
    best_value = f(current_point)

    function_values = [best_value]

    # Print a separator before the progress bar
    print("-"*100)
    
    t = trange(num_iterations, desc="Bar desc", leave=True)
    for i in t:
        # Generate new keys
        key, subkey1, subkey2 = jrd.split(key, 3)

        j = jrd.randint(subkey1, (), 0, len(current_point))  # choose a random coordinate

        if decay_rate > 0 and (i + 1 ) % decay_step == 0:
            alpha = alpha * decay_rate
            alpha = max(alpha, decay_threshold)
        
        # Move in the negative direction of the chosen coordinate
        def update_fn(x):
            # Generate noise
            noise = jrd.normal(subkey2, ()) * sigma

            # Compute the gradient of the chosen coordinate
            xj_grad = grad_f(x)[j] + noise

            # Update the chosen coordinate
            x = x.at[j].add(alpha * xj_grad)
            
            return x

        next_point = update_fn(current_point)

        # Update the best point and value if the new point is better
        next_value = f(next_point)

        # notice that this is maximization
        if next_value > best_value:
            best_point = next_point
            best_value = next_value

        current_point = next_point

        function_values.append(next_value)

        message = f"Iteration: {i}, Value: {next_value}"

        t.set_description("[RCD] Processing %s" % message)
        t.refresh()  # to show the update immediately

    # Print the final value of x
    # print("[RCD] Final value of x:", next_point)
    print("[RCD] Final value of f:", best_value)
    
    return best_point, best_value, function_values
