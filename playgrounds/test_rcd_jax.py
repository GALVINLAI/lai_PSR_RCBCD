import jax
import jax.numpy as jnp
import random
from jax import random as jrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def random_coordinate_descent(f, initial_point, alpha, num_iterations):
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
    grad_f = jax.grad(f)

    current_point = initial_point
    best_point = current_point
    best_value = f(current_point)

    function_values = []


    for i in range(num_iterations):
        j = random.randint(0, len(current_point)-1)  # choose a random coordinate

        # move in the negative direction of the chosen coordinate
        def update_fn(x):
            xj = x[j]
            xj_grad = grad_f(x)[j]
            xj_new = xj - alpha * xj_grad
            return jax.ops.index_update(x, j, xj_new)

        next_point = update_fn(current_point)

        # update the best point and value if the new point is better
        next_value = f(next_point)
        if next_value < best_value:
            best_point = next_point
            best_value = next_value

        current_point = next_point

        function_values.append(next_value)
    
    plt.plot(function_values)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.xlim(0, 100)
    plt.title('Value vs Iteration')
    plt.savefig('value_vs_iteration_jax.png')

    return best_point, best_value


if __name__ == '__main__':
    # test the random coordinate descent algorithm on a simple function
    key = jrd.PRNGKey(0)

    def f(x):
        return jnp.sum(jnp.square(x))

    initial_point = jrd.normal(key, (2,))
    print('initial point:', initial_point)
    alpha = 0.1
    num_iterations = 1000

    best_point, best_value = random_coordinate_descent(f, initial_point, alpha, num_iterations)

    print("Best point:", best_point)
    print("Best value:", best_value)
