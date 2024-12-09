import argparse  # Used for parsing command-line arguments
import os
import sys
import matplotlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.random as jrd
from utils import load, dump, make_dir
from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from algo.bcd import block_coordinate_descent

# Set up configurations
np.random.seed(42)
# config.update("jax_enable_x64", True)

# Set the matplotlib backend to 'Agg'
matplotlib.use("Agg")

def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add the system size argument
    parser.add_argument('--N', type=int, default=3, help='System size')

    # Add the problem dimension argument
    parser.add_argument('--dim', type=int, default=6, help='The dimension of the problem')
    
    # Add the sigma argument
    parser.add_argument('--sigma', type=float, default=0.01, help='The sigma for the Gaussian noise of the gradient. Note that this is multiplied by the gradient')
    
    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=10, help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.001, help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.01, help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=1000, help='The number of iterations for the optimization algorithm')

    return parser

# Parse command-line arguments and print them
args = create_parser().parse_args()
print("Run the QAOA algorithm for the TFIM model")
print(f"System size: {args.N}")
print(f"Problem dimension: {args.dim}")
print(f"Sigma: {args.sigma}")
print(f"Repeat count: {args.repeat}")
print(f"Gradient descent learning rate: {args.lr_gd}")
print(f"Random coordinate descent learning rate: {args.lr_rcd}")
print(f"Number of iterations: {args.num_iter}")

N = args.N
dim = args.dim
sigma = args.sigma
repeat = args.repeat
lr_gd = args.lr_gd
lr_rcd = args.lr_rcd
num_iter = args.num_iter

# Load the data from the pre-dumped data
try:
    data_dict = load(f"quspin_data/tfim_N_{N}.pkl")
except FileNotFoundError:
    print("Unable to load the data file. Please run the code `python generate_tfim_ham.py` to generate the data.")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit()

psi0_input = data_dict["psi0_input"]
psi1_input = data_dict["psi1_input"]
H0 = data_dict["H0"]
H1 = data_dict["H1"]

# Convert the data to JAX arrays
psi0_input = jnp.array(psi0_input)
psi1_input = jnp.array(psi1_input)
H0 = jnp.array(H0)
H1 = jnp.array(H1)

# Get the eigenvalues and eigenvectors
H0_eval, H0_evec = jla.eigh(H0)
H1_eval, H1_evec = jla.eigh(H1)
imag_unit = jnp.complex64(1.0j)  # Imaginary unit

def get_reward(protocol):
    """Get the fidelity of the protocol
    Arguments:
        protocol -- The alpha's and beta's for a given protocol
    Returns:
        fidelity -- scalar between 0 and 1
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

def main():
    make_dir('exp/tfim')
    make_dir(f'exp/tfim/dim_{dim}')
    
    random_keys = jrd.split(jrd.PRNGKey(42), repeat)

    for exp_i in range(repeat):
        x = jnp.array([1.0] * dim)

        if os.path.exists(f'exp/tfim/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl'):
            continue

        # # Run gradient descent
        # function_values_gd, x_gd, f_x_gd, eigen_values_gd, lip_diag_values_gd = gradient_descent(get_reward, x, lr_gd, num_iter, sigma, random_keys[exp_i])

        # # Run random coordinate descent
        # x_rcd, f_x_rcd, function_values_rcd, eigen_values_rcd, lip_diag_values_rcd = random_coordinate_descent(
        #     get_reward, x, lr_rcd, num_iter, sigma, random_keys[exp_i]
        # )

        # Run block coordinate descent
        x_bcd, f_x_bcd, function_values_bcd, eigen_values_bcd, lip_diag_values_bcd = block_coordinate_descent(
            get_reward, x, num_iter, sigma, random_keys[exp_i],
            problem_name='tfim',
            opt_goal='max', 
            opt_method='analytic',
            skip_hessian=False, 
            plot_subproblem=True,
            cyclic_mode=False
        )
        
        # Save data
        make_dir(f'exp/tfim/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}')
        data_dict = {
            'function_values_bcd': function_values_bcd,
            'x_bcd': x_bcd,
            'f_x_bcd': f_x_bcd,
            'eigen_values_bcd': eigen_values_bcd,
            'lip_diag_values_bcd': lip_diag_values_bcd,
        }

        dump(data_dict, f'exp/tfim/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl')

if __name__ == "__main__":
    main()
