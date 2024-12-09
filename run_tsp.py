import argparse
import os, shutil
import matplotlib
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from utils import dump, make_dir, hamiltonian_to_matrix

from algo.gd import gradient_descent
from algo.rcd import random_coordinate_descent
from algo.rcd_mini_batch import random_coordinate_descent_mini_batch


# from algo.bcd_dev import block_coordinate_descent
# from algo.oicd import oicd

# Set up configurations
matplotlib.use("Agg")  # Set the matplotlib backend to 'Agg'
# np.random.seed(42)

def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add the system size argument
    parser.add_argument('--N', type=int, default=4, help='System size')

    # Add the problem dimension argument
    parser.add_argument('--dim', type=int, default=20, help='The dimension of the problem')
    
    # Add the sigma argument
    parser.add_argument('--sigma', type=float, default=0.1, help='The sigma for the Gaussian noise of the gradient. Note that this is multiplied by the gradient')
    
    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=10, help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.0001, help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.001, help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter', type=int, default=1000, help='The number of iterations for the optimization algorithm')

    return parser

args = create_parser().parse_args()
print("Run the QAOA algorithm for the TSP model")
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

ham_str = '600303.0 -100069.5 * z0 -100055.5 * z4 + 12.0 * z4 * z0 -100069.5 * z1 -100055.5 * z5 + 12.0 * z5 * z1 -100069.5 * z2 -100055.5 * z3 + 12.0 * z3 * z2 -100077.0 * z7 + 22.75 * z7 * z0 -100077.0 * z8 + 22.75 * z8 * z1 -100077.0 * z6 + 22.75 * z6 * z2 + 12.0 * z3 * z1 + 12.0 * z4 * z2 + 12.0 * z5 * z0 + 15.75 * z7 * z3 + 15.75 * z8 * z4 + 15.75 * z6 * z5 + 22.75 * z6 * z1 + 22.75 * z7 * z2 + 22.75 * z8 * z0 + 15.75 * z6 * z4 + 15.75 * z7 * z5 + 15.75 * z8 * z3 + 50000.0 * z3 * z0 + 50000.0 * z6 * z0 + 50000.0 * z6 * z3 + 50000.0 * z4 * z1 + 50000.0 * z7 * z1 + 50000.0 * z7 * z4 + 50000.0 * z5 * z2 + 50000.0 * z8 * z2 + 50000.0 * z8 * z5 + 50000.0 * z1 * z0 + 50000.0 * z2 * z0 + 50000.0 * z2 * z1 + 50000.0 * z4 * z3 + 50000.0 * z5 * z3 + 50000.0 * z5 * z4 + 50000.0 * z7 * z6 + 50000.0 * z8 * z6 + 50000.0 * z8 * z7'
H = hamiltonian_to_matrix(ham_str)

print(jnp.linalg.eigh(H)[0][0])
print(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0]))
print(bin(int(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0])[0][0]))[2:])

ans = list(str(bin(int(jnp.nonzero(jnp.linalg.eigh(H)[1][:, 0])[0][0]))[2:]))[::-1]
ans = jnp.array([int(i) for i in ans] + [0] * (9 - len(ans)))
print(ans)

def get_tsp_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray): Binary string as numpy array.

    Returns:
        list[int]: Sequence of cities to traverse.
            The i-th item in the list corresponds to the city which is visited in the i-th step.
            The list for an infeasible answer e.g. [[0,1],1,] can be interpreted as
            visiting [city0 and city1] as the first city, then visit city1 as the second city,
            then visit nowhere as the third city).
    """
    n = int(np.sqrt(len(x)))
    z = []
    for p__ in range(n):
        p_th_step = []
        for i in range(n):
            if x[i * n + p__] >= 0.999:
                p_th_step.append(i)
        if len(p_th_step) == 1:
            z.extend(p_th_step)
        else:
            z.append(p_th_step)
    return z

print(get_tsp_solution(ans))

E_gs = jnp.linalg.eigh(H)[0][0]

def ry(theta):
    """Create a rotation matrix for a rotation about the y-axis."""
    return jnp.array([[jnp.cos(theta / 2), -jnp.sin(theta / 2)],
                      [jnp.sin(theta / 2), jnp.cos(theta / 2)]])

def cz():
    """Create a controlled-Z gate."""
    return jnp.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])

def _apply_gate(state, gate, targets, n_qubits):
    """Apply a gate to a state."""
    operators = [jnp.eye(2)] * n_qubits
    
    if len(targets) == 1:
        operators[targets[0]] = gate
    else:
        operators[targets[0]: targets[1] + 1] = [gate]
    
    gate = operators[0]
    for operator in operators[1:]:
        gate = jnp.kron(gate, operator)
    return jnp.dot(gate, state)

def apply_gates(state, n_qubits, reps, ry_angles):
    """Apply the sequence of RY and CZ gates."""
    for rep in range(reps):
        for i in range(n_qubits):
            state = _apply_gate(state, ry(ry_angles[rep * n_qubits + i]), [i], n_qubits)
        for i in range(n_qubits - 1):
            state = _apply_gate(state, cz(), [i, i + 1], n_qubits)
    return state

@jax.jit
def get_energy_ratio(psi):
    return -jnp.vdot(psi, jnp.dot(H, psi)) / E_gs

# Define the number of qubits
n_qubits = 3 ** 2

# Define the number of repetitions
reps = 10

dim = n_qubits * reps

# Define the rotation angles for the RY gates
ry_angles = [jnp.pi / 4] * dim  # Using pi/4 for each qubit as an example

# Create the initial state
state = jnp.eye(2 ** n_qubits)[:, 0]

# Apply the gates
state = apply_gates(state, n_qubits, reps, ry_angles)

@jax.jit
def objective(params):
    psi = apply_gates(state, n_qubits, reps, params)
    return get_energy_ratio(psi)

# The final state is now stored in the 'state' variable
print(objective(ry_angles))

######################## Solver Setup ########################

def main():
    # Try the gradient descent algorithm
    
    make_dir('exp/tsp')
    # make_dir(f'exp/tsp/dim_{dim}')

    # Check if the folder exists and delete it if it does.
    # Note that we delete everything inside the folder
    dir_path = f'exp/tsp/lr_{lr_gd}/dim_{dim}/sigma_{sigma}'
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed existing directory: {dir_path}")

    random_keys = jrd.split(jrd.PRNGKey(42), repeat)

    for exp_i in range(repeat):
        print('='*100)
        print(f'Experiment # {exp_i} begins.')
        # Define the initial value for x
        # Uniformly distributed between [0, 2pi]
        # Ensure that the initial point of each experiment exp_i is random
        init_x = jrd.uniform(random_keys[exp_i], (dim,)) * 2 * np.pi

        # Initialize data_dict
        data_dict = {}

        ############################################################
        # Run gradient descent
        x_gd, f_x_gd, function_values_gd = gradient_descent(
            objective, init_x, lr_gd, num_iter, sigma, random_keys[exp_i]
        )
        data_dict.update({
            'x_gd': x_gd,
            'f_x_gd': f_x_gd,
            'function_values_gd': function_values_gd,
        })

        # Run random coordinate descent
        x_rcd, f_x_rcd, function_values_rcd = random_coordinate_descent(
            objective, init_x, lr_rcd, num_iter, sigma, random_keys[exp_i], 
            decay_step=30, decay_rate=0.85
        )
        data_dict.update({
            'x_rcd': x_rcd,
            'f_x_rcd': f_x_rcd,
            'function_values_rcd': function_values_rcd,
        })


        # Run random coordinate descent - mini-batch

        x_rcd_batch, f_x_rcd_batch, function_values_rcd_batch = random_coordinate_descent_mini_batch(
            objective, init_x, lr_rcd, num_iter, sigma, random_keys[exp_i], 
            decay_step=30, decay_rate=0.85,
            batch_size=10
        )
        data_dict.update({
            'x_rcd_batch': x_rcd_batch,
            'f_x_rcd_batch': f_x_rcd_batch,
            'function_values_rcd_batch': function_values_rcd_batch,
        })

        ############################################################
        # global setting for OICD

        # problem_name='tsp'
        # opt_goal='max'
        # plot_subproblem=False
        # cyclic_mode=True
        # solver_flag = True
        # # solver_flag = False

        # opt_interp_points = np.array([0, np.pi*2/3, np.pi*4/3])

        # inverse_interp_matrix = np.array([
        #             [np.sqrt(2)/3, np.sqrt(2)/3, np.sqrt(2)/3],
        #             [2/3, -1/3, -1/3],
        #             [0, 1/np.sqrt(3), -1/np.sqrt(3)]
        #             ])
        
        # # omega_set = [2]
        # omega_set = [1] 

        # generators_dict = {}

        # for i in range(dim):
        #     generators_dict[f"Generator_{i}"] = {
        #         'opt_interp_points': opt_interp_points,
        #         'omega_set': omega_set,
        #         'inverse_interp_matrix': inverse_interp_matrix
        #     }

        # # Run block coordinate descent (classical thetas)
        # x_oicd, f_x_oicd, function_values_oicd = oicd(
        #     objective, generators_dict, init_x, num_iter, sigma, random_keys[exp_i],
        #     problem_name=problem_name,
        #     opt_goal=opt_goal, 
        #     plot_subproblem=plot_subproblem,
        #     cyclic_mode=cyclic_mode,
        #     subproblem_iter=20,
        #     solver_flag = solver_flag,
        # )
        # data_dict.update({
        #     'x_oicd': x_oicd,
        #     'f_x_oicd': f_x_oicd,
        #     'function_values_oicd': function_values_oicd,
        # })


        ############################################################
        # # global setting for block coordinate descent methods
        # problem_name='tsp'
        # opt_goal='max'
        # plot_subproblem=False
        # cyclic_mode=True

        # # Run block coordinate descent (classical thetas)
        # x_bcd_c, f_x_bcd_c, function_values_bcd_c = block_coordinate_descent(
        #     objective, init_x, num_iter, sigma, random_keys[exp_i],
        #     problem_name=problem_name,
        #     opt_goal=opt_goal, 
        #     plot_subproblem=plot_subproblem,
        #     cyclic_mode=cyclic_mode,
        #     mode='classical'
        # )
        # data_dict.update({
        #     'x_bcd_c': x_bcd_c,
        #     'f_x_bcd_c': f_x_bcd_c,
        #     'function_values_bcd_c': function_values_bcd_c,
        # })

        # # Run block coordinate descent (general thetas)
        # x_bcd_g, f_x_bcd_g, function_values_bcd_g = block_coordinate_descent(
        #     objective, init_x, num_iter, sigma, random_keys[exp_i],
        #     problem_name=problem_name,
        #     opt_goal=opt_goal,
        #     plot_subproblem=plot_subproblem,
        #     cyclic_mode=cyclic_mode,
        #     mode='general'
        # )
        # data_dict.update({
        #     'x_bcd_g': x_bcd_g,
        #     'f_x_bcd_g': f_x_bcd_g,
        #     'function_values_bcd_g': function_values_bcd_g,
        # })

        # # Run block coordinate descent (regression)
        # fevl_num_each_iter = 5
        # x_bcd_reg, f_x_bcd_reg, function_values_bcd_reg = block_coordinate_descent(
        #     objective, init_x, num_iter, sigma, random_keys[exp_i],
        #     problem_name=problem_name,
        #     opt_goal=opt_goal,  
        #     plot_subproblem=plot_subproblem,
        #     cyclic_mode=cyclic_mode,
        #     fevl_num_each_iter=fevl_num_each_iter,
        #     mode='reg'
        # )
        # data_dict.update({
        #     'x_bcd_reg': x_bcd_reg,
        #     'f_x_bcd_reg': f_x_bcd_reg,
        #     'function_values_bcd_reg': function_values_bcd_reg,
        #     'fevl_num_each_iter_reg': fevl_num_each_iter,
        # })

        # # Run block coordinate descent with optimized random coordinate descent
        # x_bcd_opt_rcd, f_x_bcd_opt_rcd, function_values_bcd_opt_rcd = block_coordinate_descent(
        #     objective, init_x, num_iter, sigma, random_keys[exp_i],
        #     problem_name=problem_name,
        #     opt_goal=opt_goal, 
        #     plot_subproblem=plot_subproblem,
        #     cyclic_mode=cyclic_mode,
        #     mode='opt_rcd'
        # )
        # data_dict.update({
        #     'x_bcd_opt_rcd': x_bcd_opt_rcd,
        #     'f_x_bcd_opt_rcd': f_x_bcd_opt_rcd,
        #     'function_values_bcd_opt_rcd': function_values_bcd_opt_rcd,
        # })



        make_dir(f'exp/tsp/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}')

        # Save data_dict to file (or perform further operations)
        # For example:
        # with open(f'exp/tsp/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data.pkl', 'wb') as f:
        #     pickle.dump(data_dict, f)
        dump(data_dict, f'exp/tsp/lr_{lr_gd}/dim_{dim}/sigma_{sigma}/exp_{exp_i}/data_dict.pkl')

if __name__ == "__main__":
    main()