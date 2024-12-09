# the code is adapted from https://minatoyuichiro.medium.com/variational-quantum-factoring-using-qaoa-and-vqe-on-blueqat-29c6f4f195f1. https://arxiv.org/pdf/1411.6758.pdf
import jax
import jax.numpy as np
import jax.random as random
import numpy as onp
from jax.scipy.linalg import expm
from tqdm import trange

from utils import make_dir, dump, hamiltonian_to_matrix


# Number of qubits in the system
n_qubits = 4

hamiltonian_str = '-3.0 + 0.5 * Z0 + 0.25 * Z1 + 0.25 * Z2 + 0.5 * Z3 + 0.75 * Z0*Z2 - 0.25 * Z1*Z2 + 0.25 * Z0*Z1 + 0.25 * Z0*Z3 + 0.75 * Z1*Z3 + 0.25 * Z2*Z3 - 0.25 * Z0*Z1*Z2 - 0.25 * Z1*Z2*Z3'
H = hamiltonian_to_matrix(hamiltonian_str)

# print out the hamiltonian
print(H)

# check the ground state 
for x in np.linalg.eigh(H)[1][:, :2].T:
    print(np.nonzero(x))
print(int('0110', 2), int('1001', 2))

E_gs = np.linalg.eigh(H)[0][0]


# Number of qubits
n_qubits = 4

hamiltonian_str = 'X0 + X1 + X2 + X3'
total_X_operator = hamiltonian_to_matrix(hamiltonian_str)

# initial state
psi0 = np.ones(2 ** 4) / 2 ** 2

def qaoa_ansatz(params):
    psi = psi0
    for i, param in enumerate(params): 
        if i % 2 == 0:
            psi = expm(-1j * param * H) @ psi
        else: 
            psi = expm(-1j * param * total_X_operator) @ psi
    return psi

def get_energy_ratio(psi):
    return -np.absolute(np.vdot(psi, np.dot(H, psi))) / E_gs

@jax.jit
def objective(params):
    psi = qaoa_ansatz(params)
    return get_energy_ratio(psi)





num_layer = 20
for num_layer in [5, 8, 10, 15, 20]:
    
    # Gradient of the function
    grad_f = jax.jit(jax.grad(objective))

    # Initialize parameters
    x0 = random.uniform(random.PRNGKey(42), (2 * num_layer,)) * 2 * np.pi
    
    num_steps = 1000

    make_dir(f"results/qaoa/{num_layer}")
    for learning_rate in np.linspace(0.1, 0.5, 51):
        learning_rate = round(learning_rate, 3)

        x = x0
        t = trange(num_steps, desc="GD", leave=True)
        x_list = []
        f_list = []
        x_list.append(x)
        f_list.append(objective(x))
        # Gradient descent
        for i in t:
            x += learning_rate * grad_f(x)
            x_list.append(x)
            f_list.append(objective(x))
            
            
            
            message = f"Iteration: {i}, Value: {objective(x)}"
            t.set_description("Processing %s" % message)
            t.refresh()  # to show immediately the update

        make_dir(f"results/qaoa/layer_{num_layer}/lr_{learning_rate}")
        dump({'x': x_list, 'f': f_list}, f"results/qaoa/layer_{num_layer}/lr_{learning_rate}/gd.pkl")
        print(f"Minimum at: {x}")

        # Create a PRNG key
        key = random.PRNGKey(0)
        
        x_list = []
        f_list = []
        
        x = x0
        x_list.append(x)
        f_list.append(objective(x))
        
        t = trange(num_steps, desc="RCD", leave=True)
        # Random coordinate descent
        for i in t:
            # Split the PRNG key
            key, subkey = random.split(key)
            
            # Choose random index using the subkey
            idx = random.randint(subkey, (), 0, x.shape[0])
            
            # Construct one-hot encoded vector for chosen index
            mask = np.zeros(x.shape[0])
            mask = mask.at[idx].set(1)
            
            # Update only the chosen coordinate
            x += learning_rate * mask * grad_f(x)
            x_list.append(x)
            f_list.append(objective(x))
            
            message = f"Iteration: {i}, Value: {objective(x)}"
            t.set_description("Processing %s" % message)
            t.refresh()  # to show immediately the update
        dump({'x': x_list, 'f': f_list}, f"results/qaoa/layer_{num_layer}/lr_{learning_rate}/rcd.pkl")
        
        print(f"Minimum at: {x}")
