import numpy as np

np.random.seed(42)
import random

import click
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.random as jrd
import matplotlib
from sysflow.utils.common_utils.file_utils import load, dump

from jax.config import config

config.update("jax_enable_x64", True)
# Global flag to set a specific platform, must be used at startup.
config.update('jax_platform_name', 'cpu')


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

from icecream import ic
import os 

# TODO: change this to be argument related
N = 3
p = 18

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

# get the second derivative for a second differentiable function
def second_derivative(func, argnums):
    return jax.grad(jax.grad(func, argnums=argnums), argnums=argnums)


def first_derivative(func, argnums):
    return jax.grad(func, argnums=argnums)


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


def get_reward2(*protocol):
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


def make_plots(): 
    for p1 in os.listdir("exp_noise"):
        p = int(p1.split("_")[1])
        print(f"p = {p}")

        for p2 in os.listdir(f"exp_noise/{p1}/"):
            sigma = float(p2.split("_")[1])
            print(f"\tsigma = {sigma}")
            f1_list = []
            f2_list = []
            for p3 in os.listdir(f"exp_noise/{p1}/{p2}"):
                data = load(f"exp_noise/{p1}/{p2}/{p3}/data_dict.pkl")
                f1 = data["function_values1"]
                f2 = data["function_values2"]
                f1_list.append(f1)
                f2_list.append(f2)

            f1_mean = np.mean(f1_list, axis=0)
            f1_std = np.std(f1_list, axis=0)
            f2_mean = np.mean(f2_list, axis=0)
            f2_std = np.std(f2_list, axis=0)

            # this is to add the shade around the data.
            plt.figure()
            plt.plot(f1_mean, label="gradient descent")
            plt.fill_between(
                range(len(f1_mean)), f1_mean - f1_std, f1_mean + f1_std, alpha=0.2
            )
            plt.plot(f2_mean, label="random coordinate descent")
            plt.fill_between(
                range(len(f2_mean)), f2_mean - f2_std, f2_mean + f2_std, alpha=0.2
            )
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel("fidelity")
            plt.title(f"GD and RCDM comparisons with p = {p}, sigma = {sigma}")
            plt.grid()
            plt.tight_layout()
            make_dir("figs_exp_noise")
            plt.savefig(f"figs_exp_noise/plot_p_{p}_sigma_{sigma}.png")
            plt.close()



def compute_Lipschitz_constant(): 
    for p1 in os.listdir("exp_noise"):
        p = int(p1.split("_")[1])
        print(f"p = {p}")
        sigma = 0.001
        p2 = f'sigma_{sigma}'
        print(f"\tsigma = {sigma}")
        p3 = 'exp_0'
        data = load(f"exp_noise/{p1}/{p2}/{p3}/data_dict.pkl")
        x1 = data["x1"]
        f_x1 = data["f_x1"]

        # compute the Lipschitz constant
        x = x1
        lip_lip_list = []
        for _ in trange(100): 
            x_new = x + 0.01 * jnp.array([random.uniform(-1, 1) for _ in range(p)])
            lip_list = []
            for i in range(p): 
                lip_list.append( second_derivative(get_reward2, i)(*x_new).item() )
            lip_lip_list.append(lip_list)
        
        lip_lip_list = list(zip(*lip_lip_list))
        for i in range(p): 
            print(f'L{i}', abs(max(lip_lip_list[i])), abs(min(lip_lip_list[i])))





def compute_Lipschitz_constant_hessian(): 
    for p1 in os.listdir("exp_noise"):
        p = int(p1.split("_")[1])
        print(f"p = {p}")
        sigma = 0.001
        p2 = f'sigma_{sigma}'
        print(f"\tsigma = {sigma}")
        p3 = 'exp_0'
        data = load(f"exp_noise/{p1}/{p2}/{p3}/data_dict.pkl")
        x1 = data["x1"]
        f_x1 = data["f_x1"]

        # compute the Lipschitz constant
        x = x1
        lip_list = []
        for _ in trange(100): 
            x_new = x + 0.01 * jnp.array([random.uniform(-1, 1) for _ in range(p)])
            hessian_mat = jax.hessian(get_reward)(x_new)
            vals, vecs = jnp.linalg.eig(hessian_mat)
            lip_list.append(abs(min(vals)))
        
        print(f'L', abs(min(lip_list)), abs(max(lip_list)))




if __name__ == "__main__":
    # plot the shaded curves for gradient descent and random coordinate descent
    make_plots()
    
    # print out the Lipschitz constant for each p and sigma
    compute_Lipschitz_constant()
    
    # print out the Lipschitz constant for the vector
    compute_Lipschitz_constant_hessian()