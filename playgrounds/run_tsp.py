import jax
import jax.numpy as np
import jax.random as random
import numpy as onp
from jax.scipy.linalg import expm
from tqdm import trange

from sysflow.utils.common_utils.file_utils import make_dir, dump

# Define Pauli Z matrix
Z = np.array([[1, 0], [0, -1]])

# Define identity operator
I = np.eye(2)

def tensor_product(i, operator):
    """Compute the tensor product of `operator` on the `i`-th qubit."""
    matrices = [I]*9  # for 9 qubits
    matrices[i] = operator
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Construct Hamiltonian
H = 600303.0*np.eye(2**9)  # The first term is special: 600303.0*I
H -= 100055.5*tensor_product(3, Z)
H -= 100055.5*tensor_product(4, Z)
H -= 100055.5*tensor_product(5, Z)
H -= 100077.0*tensor_product(6, Z)
H -= 100077.0*tensor_product(7, Z)
H -= 100077.0*tensor_product(8, Z)
H += 50000.0*np.dot(tensor_product(0, Z), tensor_product(1, Z))
H += 50000.0*np.dot(tensor_product(0, Z), tensor_product(2, Z))
H -= 100069.5*tensor_product(0, Z)
H += 50000.0*np.dot(tensor_product(0, Z), tensor_product(3, Z))
H += 12.0*np.dot(tensor_product(0, Z), tensor_product(4, Z))
H += 12.0*np.dot(tensor_product(0, Z), tensor_product(5, Z))
H += 50000.0*np.dot(tensor_product(0, Z), tensor_product(6, Z))
H += 22.75*np.dot(tensor_product(0, Z), tensor_product(7, Z))
H += 22.75*np.dot(tensor_product(0, Z), tensor_product(8, Z))
H += 50000.0*np.dot(tensor_product(1, Z), tensor_product(2, Z))
H -= 100069.5*tensor_product(1, Z)
H += 12.0*np.dot(tensor_product(1, Z), tensor_product(3, Z))
H += 50000.0*np.dot(tensor_product(1, Z), tensor_product(4, Z))
H += 12.0*np.dot(tensor_product(1, Z), tensor_product(5, Z))
H += 22.75*np.dot(tensor_product(1, Z), tensor_product(6, Z))
H += 50000.0*np.dot(tensor_product(1, Z), tensor_product(7, Z))
H += 22.75*np.dot(tensor_product(1, Z), tensor_product(8, Z))
H -= 100069.5*tensor_product(2, Z)
H += 12.0*np.dot(tensor_product(2, Z), tensor_product(3, Z))
H += 12.0*np.dot(tensor_product(2, Z), tensor_product(4, Z))
H += 50000.0*np.dot(tensor_product(2, Z), tensor_product(5, Z))
H += 22.75*np.dot(tensor_product(2, Z), tensor_product(6, Z))
H += 22.75*np.dot(tensor_product(2, Z), tensor_product(7, Z))
H += 50000.0*np.dot(tensor_product(2, Z), tensor_product(8, Z))
H += 50000.0*np.dot(tensor_product(3, Z), tensor_product(4, Z))
H += 50000.0*np.dot(tensor_product(3, Z), tensor_product(5, Z))
H += 50000.0*np.dot(tensor_product(3, Z), tensor_product(6, Z))
H += 15.75*np.dot(tensor_product(3, Z), tensor_product(7, Z))
H += 15.75*np.dot(tensor_product(3, Z), tensor_product(8, Z))
H += 50000.0*np.dot(tensor_product(4, Z), tensor_product(5, Z))
H += 15.75*np.dot(tensor_product(4, Z), tensor_product(6, Z))
H += 50000.0*np.dot(tensor_product(4, Z), tensor_product(7, Z))
H += 15.75*np.dot(tensor_product(4, Z), tensor_product(8, Z))
H += 15.75*np.dot(tensor_product(5, Z), tensor_product(6, Z))
H += 15.75*np.dot(tensor_product(5, Z), tensor_product(7, Z))
H += 50000.0*np.dot(tensor_product(5, Z), tensor_product(8, Z))
H += 50000.0*np.dot(tensor_product(6, Z), tensor_product(7, Z))
H += 50000.0*np.dot(tensor_product(6, Z), tensor_product(8, Z))
H += 50000.0*np.dot(tensor_product(7, Z), tensor_product(8, Z))

print(H)
print(np.linalg.eigh(H)[0][0])
print(np.nonzero(np.linalg.eigh(H)[1][:, 0]))
print(bin(int(np.nonzero(np.linalg.eigh(H)[1][:, 0])[0][0]))[2:])

ans = list(str(bin(int(np.nonzero(np.linalg.eigh(H)[1][:, 0])[0][0]))[2:]))[::-1]
ans = np.array([int(i) for i in ans]+ [0] * (9 - len(ans)))
print(ans)
def get_tsp_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        list[int]: sequence of cities to traverse.
            The i-th item in the list corresponds to the city which is visited in the i-th step.
            The list for an infeasible answer e.g. [[0,1],1,] can be interpreted as
            visiting [city0 and city1] as the first city, then visit city1 as the second city,
            then visit no where as the third city).
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
# this is the sanity check!


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

@jax.jit
def get_energy_ratio(psi):
    # return -np.absolute(np.vdot(psi, np.dot(H, psi))) / E_gs
    return np.absolute(np.vdot(psi, np.dot(H, psi))) / E_gs



# define the number of qubits
n_qubits = 3 ** 2

# define the number of repetitions
reps = 10

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
print( objective(ry_angles))
# exit()



for num_layer in [10]:
    
    # Gradient of the function
    grad_f = jax.jit(jax.grad(objective))

    # Initialize parameters
    x0 = random.uniform(random.PRNGKey(42), ( n_qubits * reps ,)) * 2 * np.pi
    
    num_steps = 10000

    make_dir(f"results/tsp/{num_layer}")
    for learning_rate in np.linspace(5e-4, 5e-3, 51):
    # for learning_rate in np.linspace(0.5, 5, 51):
        
    # for learning_rate in [0.1]:
        # learning_rate = round(learning_rate, 3)

        x = x0
        t = trange(num_steps, desc="GD", leave=True)
        x_list = []
        f_list = []
        x_list.append(x)
        f_list.append(objective(x))
        # Gradient descent
        for i in t:
            x -= learning_rate * grad_f(x)
            x_list.append(x)
            f_list.append(objective(x))
            
            
            
            message = f"Iteration: {i}, Value: {objective(x)}"
            t.set_description("Processing %s" % message)
            t.refresh()  # to show immediately the update

        make_dir(f"results/tsp/layer_{num_layer}/lr_{learning_rate}")
        dump({'x': x_list, 'f': f_list}, f"results/tsp/layer_{num_layer}/lr_{learning_rate}/gd.pkl")
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
            x -= learning_rate * mask * grad_f(x)
            x_list.append(x)
            f_list.append(objective(x))
            
            message = f"Iteration: {i}, Value: {objective(x)}"
            t.set_description("Processing %s" % message)
            t.refresh()  # to show immediately the update
        dump({'x': x_list, 'f': f_list}, f"results/tsp/layer_{num_layer}/lr_{learning_rate}/rcd.pkl")
        
        print(f"Minimum at: {x}")
