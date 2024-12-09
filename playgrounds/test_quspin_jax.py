import numpy as np

np.random.seed(42)
import random

import click
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import matplotlib
from sysflow.utils.common_utils.file_utils import load

from jax.config import config

config.update("jax_enable_x64", True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

# TODO: change this to be argument related
N = 3
p = 6

# load the data from the pre-dumped data
data_dict = load("quspin_data/quspin_data.pkl")

psi0_input = data_dict["psi0_input"]
psi1_input = data_dict["psi1_input"]
H0 = data_dict["H0"]
H1 = data_dict["H1"]

# convert the data to jax array
psi0_input = jnp.array(psi0_input)
psi1_input = jnp.array(psi1_input)
H0 = jnp.array(H0)
H1 = jnp.array(H1)

# get the eigenvalues and eigenvectors
H0_eval, H0_evec = jla.eigh(H0)
H1_eval, H1_evec = jla.eigh(H1)
imag_unit = jnp.complex64(1.0j)


def get_reward(protocol):
    """Get the fidelity of the protocol
    Arguments:
        protocol -- The alpha's and beta's for a given protocol
    Returns:
        fildeity -- scalar between 0 and 1
    """
    u = psi0_input
    for i in range(len(protocol)):
        if i % 2 == 0:
            u = jnp.matmul(H0_evec.conj().T, u)
            u = jnp.multiply(jnp.exp(-protocol[i] * imag_unit * H0_eval), u)
            u = jnp.matmul(H0_evec, u)
        else:
            u = jnp.matmul(H1_evec.conj().T, u)
            u = jnp.multiply(jnp.exp(-protocol[i] * imag_unit * H1_eval), u)
            u = jnp.matmul(H1_evec, u)

    return jnp.absolute(jnp.dot(psi1_input.T.conjugate(), u)) ** 2


def gradient_descent(f, initial_point, learning_rate, num_iterations):
    grad_f = jax.grad(f)
    x = initial_point
    t = trange(num_iterations, desc="Bar desc", leave=True)
    function_values = []
    # Perform gradient descent
    for i in t:
        # Compute the gradient of f at the current value of x
        gradient = grad_f(x)

        # Update x using the gradient and the learning rate
        x = x + learning_rate * gradient

        function_value = f(x)
        function_values.append(function_value)
        message = f"Iteration: {i}, Value: {function_value}"

        t.set_description("Processing %s" % message)
        t.refresh()  # to show immediately the update

    # Print the final value of x
    print("Final value of x:", x)
    return function_values


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

    t = trange(num_iterations, desc="Bar desc", leave=True)
    for i in t:
        j = random.randint(0, len(current_point) - 1)  # choose a random coordinate

        # move in the negative direction of the chosen coordinate
        def update_fn(x):
            xj = x[j]
            xj_grad = grad_f(x)[j]
            xj_new = xj + alpha * xj_grad
            return jax.ops.index_update(x, j, xj_new)

        next_point = update_fn(current_point)

        # update the best point and value if the new point is better
        next_value = f(next_point)
        if next_value < best_value:
            best_point = next_point
            best_value = next_value

        current_point = next_point

        function_values.append(next_value)

        message = f"Iteration: {i}, Value: {next_value}"

        t.set_description("Processing %s" % message)
        t.refresh()  # to show immediately the update

    return best_point, best_value, function_values


@click.command()
@click.option("--p", default=6, help="The dimension of the problem")
def main(p):
    # try the gradient descent algorithm
    learning_rate = 0.01

    # Define the initial value for x
    x = jnp.array(
        [
            1.0,
        ]
        * p
    )

    # Set the number of iterations for gradient descent
    num_iterations = 1000

    # Run gradient descent
    function_values1 = gradient_descent(get_reward, x, learning_rate, num_iterations)

    # Run random coordinate descent
    _, _, function_values2 = random_coordinate_descent(
        get_reward, x, learning_rate, num_iterations
    )

    plt.plot(function_values1, label="Gradient Descent")
    plt.plot(function_values2, label="Random Coordinate Descent")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(f"Value vs Iteration (dimension = {p})")
    plt.legend()
    plt.savefig(f"value_vs_iteration_jax_p_{p}.png")


if __name__ == "__main__":
    main()
