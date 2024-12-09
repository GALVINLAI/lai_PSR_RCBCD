import jax
import jax.numpy as np
import jax.random as random
import numpy as onp
from jax.scipy.linalg import expm
from tqdm import trange

from sysflow.utils.common_utils.file_utils import make_dir, dump


# Define the 2x2 Pauli matrices
I = np.eye(2)
Z = np.array([[1, 0], [0, -1]])

# Define a 4-qubit identity for 4-qubit Hamiltonian
I_4 = np.eye(16)

# Define a function to perform tensor product for Pauli matrices
def multi_kron(matrices):
    result = np.eye(1)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

# Define the Hamiltonian
def hamiltonian():
    # 0.5*I term
    H = 0.5 * I_4
    # -3.0*Z[0] term
    H -= 3.0 * multi_kron([Z, I, I, I])
    # Interaction terms
    for i in range(3):
        # 0.5*Z[0]*Z[i+1] term
        H += 0.5 * multi_kron([Z if j == 0 or j == i+1 else I for j in range(4)])
    # 0.5*Z[1]*Z[2] term
    H += 0.5 * multi_kron([Z if j == 1 or j == 2 else I for j in range(4)])
    # 0.5*Z[2]*Z[3] term
    H += 0.5 * multi_kron([Z if j == 2 or j == 3 else I for j in range(4)])
    return H

# Print the Hamiltonian
H = hamiltonian()

# verify the Hamiltonian
print(np.nonzero(np.linalg.eigh(H)[1][:, 0]))
print(int('0101', 2))

E_gs = np.linalg.eigh(H)[0][0]

def ry(theta):
    """Create a rotation matrix for a rotation about the y-axis."""
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])

def cz():
    """Create a controlled-Z gate."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def apply_gate(state, gate, targets, n_qubits):
    """Apply a gate to a state."""
    operators = [np.eye(2)] * n_qubits
    
    if len(targets) == 1:
        operators[targets[0]] = gate
    else:
        operators[targets[0]: targets[1]+ 1] = [gate]
    
    gate = operators[0]
    for operator in operators[1:]:
        gate = np.kron(gate, operator)
    return np.dot(gate, state)

def apply_gates(state, n_qubits, reps, ry_angles):
    """Apply the sequence of RY and CZ gates."""
    for rep in range(reps):
        for i in range(n_qubits):
            state = apply_gate(state, ry(ry_angles[rep * n_qubits + i]), [i], n_qubits)
        for i in range(n_qubits - 1):
            state = apply_gate(state, cz(), [i, i+1], n_qubits)
    return state


def get_energy_ratio(psi):
    return -np.absolute(np.vdot(psi, np.dot(H, psi))) / E_gs


# define the number of qubits
n_qubits = 4

# define the number of repetitions
reps = 5

# define the rotation angles for the RY gates
ry_angles = [np.pi/4] * n_qubits * reps  # using pi/4 for each qubit as an example

# create the initial state
state = np.eye(2**n_qubits)[:, 0]

# apply the gates
state = apply_gates(state, n_qubits, reps, ry_angles)



@jax.jit
def objective(params):
    psi = apply_gates(state, n_qubits, reps, params)
    return get_energy_ratio(psi)


# the final state is now stored in the 'state' variable
# print( objective(ry_angles))
# exit()



for num_layer in [5]:
    
    # Gradient of the function
    grad_f = jax.jit(jax.grad(objective))

    # Initialize parameters
    x0 = random.uniform(random.PRNGKey(42), ( n_qubits * reps ,)) * 2 * np.pi
    
    num_steps = 1000

    make_dir(f"results/maxcut/{num_layer}")
    for learning_rate in np.linspace(1.0, 4.0, 51):
    # for learning_rate in [0.1]:
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

        make_dir(f"results/maxcut/layer_{num_layer}/lr_{learning_rate}")
        dump({'x': x_list, 'f': f_list}, f"results/maxcut/layer_{num_layer}/lr_{learning_rate}/gd.pkl")
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
        dump({'x': x_list, 'f': f_list}, f"results/maxcut/layer_{num_layer}/lr_{learning_rate}/rcd.pkl")
        
        print(f"Minimum at: {x}")
