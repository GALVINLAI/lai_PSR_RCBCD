# the code is adapted from https://minatoyuichiro.medium.com/variational-quantum-factoring-using-qaoa-and-vqe-on-blueqat-29c6f4f195f1. https://arxiv.org/pdf/1411.6758.pdf
import jax
import jax.numpy as np
import numpy as onp
import jax.random as random
import numpy as onp
from jax.scipy.linalg import expm
from tqdm import trange, tqdm
import pickle
from matplotlib import pyplot as plt

from sysflow.utils.common_utils.file_utils import make_dir, dump

# Define the 2x2 Identity and Pauli-Z matrices
I_2 = np.eye(2)
Z_2 = np.array([[1, 0], [0, -1]])

# Function to calculate the operator on n qubits at position pos
def operator_on_qubits(op_2, pos, n):
    op_n = 1
    for i in range(n):
        op_n = np.kron(op_n, op_2 if i == pos else I_2)
    return op_n

# Number of qubits in the system
n_qubits = 4

# Initialize Hamiltonian as zero matrix
H = np.zeros((2**n_qubits, 2**n_qubits))

# -3.0*I
H += -3.0 * np.eye(2**n_qubits)

# +0.5*Z[0] + 0.25*Z[1] + 0.25*Z[2] + 0.5*Z[3]
for i, coef in enumerate([0.5, 0.25, 0.25, 0.5]):
    H += coef * operator_on_qubits(Z_2, i, n_qubits)

# +0.75*Z[0]*Z[2] - 0.25*Z[1]*Z[2] + 0.25*Z[0]*Z[1] + 0.25*Z[0]*Z[3] + 0.75*Z[1]*Z[3] + 0.25*Z[2]*Z[3]
for i, j, coef in [(0, 2, 0.75), (1, 2, -0.25), (0, 1, 0.25), (0, 3, 0.25), (1, 3, 0.75), (2, 3, 0.25)]:
    H += coef * operator_on_qubits(Z_2, i, n_qubits) * operator_on_qubits(Z_2, j, n_qubits)

# -0.25*Z[0]*Z[1]*Z[2] - 0.25*Z[1]*Z[2]*Z[3]
for i, j, k, coef in [(0, 1, 2, -0.25), (1, 2, 3, -0.25)]:
    H += coef * operator_on_qubits(Z_2, i, n_qubits) * operator_on_qubits(Z_2, j, n_qubits) * operator_on_qubits(Z_2, k, n_qubits)

# print out the hamiltonian
print(H)

# check the ground state 
for x in np.linalg.eigh(H)[1][:, :2].T:
    print(np.nonzero(x))
print(int('0110', 2), int('1001', 2))

E_gs = np.linalg.eigh(H)[0][0]


# Define the 2x2 Pauli-X operator
X = np.array([[0, 1], [1, 0]])

# Define the 2x2 identity operator
I = np.array([[1, 0], [0, 1]])

# Function to get operator for n qubits at position pos
def get_operator(total_qubits, op, pos):
    # Initialize operator as identity
    operator = 1
    for i in range(total_qubits):
        operator = np.kron(operator, op if i == pos else I)
    return operator

# Number of qubits
n_qubits = 4

# Initialize total operator as zero matrix
total_X_operator = np.zeros((2**n_qubits, 2**n_qubits))

# Create total X operator by applying X gate to each qubit
for qubit in range(n_qubits):
    total_X_operator += get_operator(n_qubits, X, qubit)

print(total_X_operator)

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


f = objective
grad_f = jax.jit(jax.grad(f))
hess_f = jax.jit(jax.hessian(f))


def get_lip_info(x_list):
    eigen_values = []
    lip_diag_values = []
    # Perform gradient descent
    for x in tqdm(x_list):

        # adding the hessian info
        hessian_mat = hess_f(x)
        vals, vecs = onp.linalg.eig(onp.array(hessian_mat))
        eigen_values.append(vals)
        lip_diag_values.append(onp.diag(hessian_mat))
        
    return eigen_values, lip_diag_values

def load_data(num_layer, learning_rate):
    rcd_path = f"results/qaoa/layer_{num_layer}/lr_{learning_rate}/rcd.pkl"
    gd_path = f"results/qaoa/layer_{num_layer}/lr_{learning_rate}/gd.pkl"

    with open(rcd_path, 'rb') as rcd_file:
        rcd_data = pickle.load(rcd_file)['x']
    with open(gd_path, 'rb') as gd_file:
        gd_data = pickle.load(gd_file)['x']

    return rcd_data, gd_data

num_layer = 20
learning_rate = 0.18800000846385956
folder = f'plots/factor/qaoa/layer_{num_layer}/lr_{learning_rate}'


rcd_data, gd_data = load_data(num_layer, learning_rate)

eigen_values_rcd, lip_diag_values_rcd = get_lip_info(rcd_data)
eigen_values_gd, lip_diag_values_gd = get_lip_info(gd_data)

f1, f2 = eigen_values_rcd, eigen_values_gd
f3, f4 = lip_diag_values_rcd, lip_diag_values_gd

f1 = np.array(list(map(lambda x: np.max(np.abs(x)), f1)))
f2 = np.array(list(map(lambda x: np.max(np.abs(x)), f2)))

f3 = np.array(list(map(lambda x: np.mean(np.abs(x)), f3)))
f4 = np.array(list(map(lambda x: np.mean(np.abs(x)), f4)))

f5 = f1 / f3
f6 = f2 / f4

plt.plot(f5,  color='b', label='RCD')
plt.plot(f6, color='r', label='GD')


plt.title('Lipschitz Constant Ratio')
plt.xlabel('Steps')
plt.ylabel('L / L_avg')
plt.legend()
plt.grid()
# plt.show()
plt.savefig('lip_factor.png')
