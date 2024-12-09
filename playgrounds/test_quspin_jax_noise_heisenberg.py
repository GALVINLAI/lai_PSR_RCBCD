import numpy as np

np.random.seed(42)
import random

import click
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import matplotlib
from sysflow.utils.common_utils.file_utils import load, dump, make_dir

from jax.config import config
import os

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

matplotlib.use("Agg")
from tqdm import trange
import numpy as np

# TODO: change this to be argument related
N = 4
p = 6

# load the data from the pre-dumped data
data_dict = load("quspin_data/quspin_data_heisenberg.pkl")

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


def gradient_descent(f, initial_point, learning_rate, num_iterations, sigma):
    grad_f = jax.jit(jax.grad(f))
    hess_f = jax.jit(jax.hessian(f))
    x = initial_point
    t = trange(num_iterations, desc="Bar desc", leave=True)
    function_values = []
    eigen_values = []
    lip_diag_values = []
    # Perform gradient descent
    for i in t:
        # Compute the gradient of f at the current value of x
        gradient = grad_f(x) * (1 + jnp.array(np.random.randn(*x.shape)) * sigma )

        # Update x using the gradient and the learning rate
        x = x + learning_rate * gradient

        function_value = f(x)
        function_values.append(function_value)
        message = f"Iteration: {i}, Value: {function_value}"

        # adding the hessian info
        hessian_mat = hess_f(x)
        vals, vecs = np.linalg.eig(np.array(hessian_mat))
        eigen_values.append(vals)
        lip_diag_values.append(np.diag(hessian_mat))
        
        t.set_description("Processing %s" % message)
        t.refresh()  # to show immediately the update

    # Print the final value of x
    print("Final value of x:", x)
    return function_values, x, function_value, eigen_values, lip_diag_values


def random_coordinate_descent(f, initial_point, alpha, num_iterations, sigma):
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
    hess_f = jax.jit(jax.hessian(f))

    current_point = initial_point
    best_point = current_point
    best_value = f(current_point)

    function_values = []
    eigen_values = []
    lip_diag_values = []
    
    t = trange(num_iterations, desc="Bar desc", leave=True)
    for i in t:
        j = random.randint(0, len(current_point) - 1)  # choose a random coordinate

        # move in the negative direction of the chosen coordinate
        def update_fn(x):
            xj = x[j]
            xj_grad = grad_f(x)[j] * ( 1 + jnp.array(np.random.randn()) * sigma )
            xj_new = xj + alpha * xj_grad
            return jax.ops.index_update(x, j, xj_new)

        next_point = update_fn(current_point)

        # update the best point and value if the new point is better
        next_value = f(next_point)
        if next_value > best_value:
            best_point = next_point
            best_value = next_value

        current_point = next_point

        function_values.append(next_value)

        message = f"Iteration: {i}, Value: {next_value}"

        # adding the hessian info
        hessian_mat = hess_f(current_point)
        vals, vecs = np.linalg.eig(np.array(hessian_mat))
        eigen_values.append(vals)
        lip_diag_values.append(np.diag(hessian_mat))
        
        t.set_description("Processing %s" % message)
        t.refresh()  # to show immediately the update

    return best_point, best_value, function_values, eigen_values, lip_diag_values


@click.command()
@click.option("--p", default=6, help="The dimension of the problem")
@click.option("--sigma", default=0.01, help='the sigma for the gaussian noise of the gradient. Note that this is multiplied by the gradient')
@click.option("--repeat", default=10, help='the number of times to repeat the experiment')
@click.option("--lr_gd", default=0.001, help='the learning rate for the gradient descent')
@click.option("--lr_rcd", default=0.01, help='the learning rate for the random coordinate descent')
def main(p, sigma, repeat, lr_gd, lr_rcd):
    # try the gradient descent algorithm
    learning_rate = 0.01    



    make_dir('exp_noise_heisenberg')
    make_dir(f'exp_noise_heisenberg/p_{p}')
    for exp_i in range(repeat):
        # Define the initial value for x
        x = jnp.array(
            [
                1.0,
            ]
            * p
        )
        
        
        if os.path.exists(f'exp_noise_heisenberg/lr_{lr_gd}/p_{p}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl'):
            continue


        # Set the number of iterations for gradient descent
        num_iterations = 10000
        

        # Run gradient descent
        function_values1, x1, f_x1, eigen_values1, lip_diag_values1 = gradient_descent(get_reward, x, lr_gd, num_iterations, sigma)

        # Run random coordinate descent
        x2, f_x2, function_values2, eigen_values2, lip_diag_values2 = random_coordinate_descent(
            get_reward, x, lr_rcd, num_iterations, sigma
        )
        
        
        make_dir(f'exp_noise_heisenberg/lr_{lr_gd}/p_{p}/sigma_{sigma}/exp_{exp_i}')
        data_dict = {
            'function_values1': function_values1,
            'function_values2': function_values2,
            'x1': x1,
            'x2': x2,
            'f_x1': f_x1,
            'f_x2': f_x2,
            'eigen_values1': eigen_values1,
            'eigen_values2': eigen_values2,
            'lip_diag_values1': lip_diag_values1,
            'lip_diag_values2': lip_diag_values2,
        }
        
        dump(data_dict, f'exp_noise_heisenberg/lr_{lr_gd}/p_{p}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl')

if __name__ == "__main__":
    main()
