import numpy as np
import jax
import jax.random as jrd
from tqdm import trange

def random_coordinate_descent_mini_batch(f, initial_point, alpha, num_iterations, sigma, key,
                                          decay_step=30, 
                                          decay_rate=-1.0, 
                                          decay_threshold=1e-4,
                                          batch_size=10):
    """
    Random Coordinate Descent algorithm for optimizing a function f using automatic differentiation.

    Parameters:
        - f: function to optimize
        - initial_point: starting point for the algorithm
        - alpha: step size for the algorithm
        - num_iterations: number of iterations to run the algorithm for
        - batch_size: size of the mini-batches used for updating coordinates

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
    print("-" * 100)

    t = trange(num_iterations, desc="Bar desc", leave=True)
    for i in t:
        # Generate new keys
        key, subkey1, subkey2 = jrd.split(key, 3)

        m = len(current_point)  # Length of the current point
        num_batches = m // batch_size  # Number of mini-batches

        # Ensure we randomly shuffle the coordinates into mini-batches
        indices = jrd.permutation(subkey1, m)  # Random permutation of indices
        batches = [indices[j * batch_size: (j + 1) * batch_size] for j in range(num_batches)]
        
        if m % batch_size != 0:
            # If m is not divisible by batch_size, include the leftover part as a batch
            batches.append(indices[num_batches * batch_size:])

        if decay_rate > 0 and (i + 1) % decay_step == 0:
            alpha = alpha * decay_rate
            alpha = max(alpha, decay_threshold)

        # Update for each mini-batch
        def update_fn(x, batch):
            # Generate noise
            noise = jrd.normal(subkey2, (len(batch),)) * sigma

            # Compute the gradient for each coordinate in the mini-batch
            grads = grad_f(x)[batch] + noise

            # Update the chosen coordinates
            x = x.at[batch].add(alpha * grads)

            return x

        # For each mini-batch, update the corresponding coordinates
        for batch in batches:
            current_point = update_fn(current_point, batch)

        # Update the best point and value if the new point is better
        next_value = f(current_point)

        # notice that this is maximization
        if next_value > best_value:
            best_point = current_point
            best_value = next_value

        function_values.append(next_value)

        message = f"Iteration: {i}, Value: {next_value}"

        t.set_description("[RCD_mini_batch] Processing %s" % message)
        t.refresh()  # to show the update immediately

    # Print the final value of f
    print("[RCD_mini_batch] Final value of f:", best_value)
    
    return best_point, best_value, function_values
