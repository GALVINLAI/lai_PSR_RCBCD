import numpy as np
np.random.seed(42)
import scipy as sp
import scipy.optimize as opt

import scipy.linalg as la
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

from sysflow.utils.common_utils.file_utils import load, dump, make_dir

# the policy gradient parts
def grad_descent(x, lr, iters):
    """Apply gradient descent to find the minimum of a function.
    Args:
        x: initial guess
        lr: learning rate
        iters: number of iterations
    Returns:
        x: the final guess
    """
    x_list = []
    for i in range(iters):
        x = x - lr * dfunc(x)
        x_list.append(x)
    return x, x_list


class QuManager():
    # quantum env for the agent
    def __init__(self, psi0, psi1, H0, H1):
        """quantum environment for the RL agents 
        Arguments:
            psi0 -- initial quantum state
            psi1 -- target quantum state
            H0 -- the first hamiltonian gate 
            H1 -- the second hamiltonian gate 
        """
        self.psi0_input = np.expand_dims(psi0, axis=1)
        self.imag_unit = np.complex64(1.0j)
        self.psi1 = psi1.reshape(psi1.shape[0], 1)

        self.H1 = H1
        self.H0 = H0

        self.H0_eval, self.H0_evec = la.eigh(H0)
        self.H0_eval = np.expand_dims(self.H0_eval, axis=-1)

        self.H1_eval, self.H1_evec = la.eigh(H1)
        self.H1_eval = np.expand_dims(self.H1_eval, axis=-1)

    def get_reward(self, protocol):
        """Get the fidelity of the protocol
        Arguments:
            protocol -- The alpha's and beta's for a given protocol
        Returns:
            fildeity -- scalar between 0 and 1
        """
        u = np.copy(self.psi0_input)
        for i in range(len(protocol)):
            if i % 2 == 0:
                np.matmul(self.H0_evec.conj().T, u, out=u)
                np.multiply(
                    np.exp(-protocol[i] * self.imag_unit * self.H0_eval), u, out=u)
                np.matmul(self.H0_evec, u, out=u)
            else:
                np.matmul(self.H1_evec.conj().T, u, out=u)
                np.multiply(
                    np.exp(-protocol[i] * self.imag_unit * self.H1_eval), u, out=u)
                np.matmul(self.H1_evec, u, out=u)

        return np.absolute(np.dot(self.psi1.T.conjugate(), u))[0][0] ** 2


# TODO: change this to be argument related
# this is to add the heisenberg model here 
N = 4
p = 6
heisen_delta = 0.5

L = N
basis = spin_basis_1d(L, S="1", kblock=0, pblock=1)

coupling = [[+1.0, i] for i in range(L)]
coupling2 = [[+1.0, i, i] for i in range(L)]

# real-valued
Sz = hamiltonian(
    [["z", coupling],], [], basis=basis, check_herm=False, dtype=np.float64
)

# initial state
psi_i = np.zeros(basis.Ns)
psi_i[basis.index(f"20" * int(L // 2))] = 1.0

# QAOA terms
list_H = [[heisen_delta, i, (i + 1) % L] for i in range(L)]
list_H_kin = [[0.5, i, (i + 1) % L] for i in range(L)]
static_1 = [
    ["+-", list_H_kin],
    ["-+", list_H_kin],
    ["zz", list_H],
]

# S*S, spin-1 Heisenberg model
H_star = hamiltonian(static_1, [], basis=basis, dtype=np.float64)

E_star, psi_star = H_star.eigsh(k=1, which="SA")

print("ground state energy: ", E_star / N)

print(-H_star.eigh()[0] / N)

psi_i = psi_i.reshape((-1,))
psi_star = psi_star.reshape((-1,))

psi0_input = psi_i.astype(complex)
psi1_input = psi_star.astype(complex)


H_dict = {}
sym_dict = {}

# QAOA terms

list_H_kin = [[0.5, i, (i + 1) % L] for i in range(L)]
static_1 = [["+-", list_H_kin], ["-+", list_H_kin]]
H_new = hamiltonian(static_1, [], basis=basis, dtype=np.float64)

i = 0
H_dict[i] = H_new.toarray()


coupling2 = [[heisen_delta, i, (i + 1) % L] for i in range(L)]

# real-valued
Szz = hamiltonian(
    [["zz", coupling2],],
    [],
    basis=basis,
    check_herm=False,
    dtype=np.float64,
)

i += 1
H_dict[i] = Szz.toarray()



make_dir('quspin_data')
data_dict = {
    'psi0_input': psi0_input,
    'psi1_input': psi1_input,
    'H0': H_dict[0],
    'H1': H_dict[1]
}
dump(data_dict, 'quspin_data/quspin_data_heisenberg.pkl')

exit()

quma = QuManager(psi0_input, psi1_input, H0, H1)
# print(quma.get_reward([0.1, 0.2]*p))


# plot the 2d landscape along the two directions
x0 = np.random.normal(loc=0.5, scale=0.1, size=2 * p)
# setting the boundary
bound = 2
bnds = tuple((0, bound) for _ in range(2 * p))

iter_i = 0

def mycallback(x):
    global iter_i
    iter_i += 1
    score = quma.get_reward(x)
    print("iter: {}, loss: {}, mean_reward: {}, max_reward: {},  test_reward: {}, his_reward: {}, entropy: {}".
                format(iter_i, -score, score, score, score, score, 0.0))


sol = opt.minimize(lambda x: -quma.get_reward(x), x0, bounds=bnds,
                    tol=1e-8,  method='SLSQP', options={'disp': True}, callback=mycallback)

x = sol.x



