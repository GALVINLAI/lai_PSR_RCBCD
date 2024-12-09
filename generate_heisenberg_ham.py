import argparse
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from utils import dump, make_dir

np.random.seed(42)

def create_parser():
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument('--N', type=int, default=3, help='System size')
    return parser

def create_initial_state(basis, L):
    # Create an initial quantum state psi_i, setting the value of a specific state to 1.0 in the basis vector, and the rest to 0.

    # basis.Ns is an attribute in the QuSpin library representing the number of states in the Hamiltonian basis.
    # Specifically, for a given system basis, Ns is the total number of possible quantum states.
    psi_i = np.zeros(basis.Ns)

    # int(L // 2) takes the integer part of half the system size (floor division).
    # Suppose L = 6:
    # f"20" * int(L // 2) generates the string "202020".  
    # basis.index("202020") returns the index of this state in the basis, for example, the index is 5.
    # psi_i[5] = 1.0 sets the 5th element of the initial state vector psi_i to 1.0.
    psi_i[basis.index(f"20" * int(L // 2))] = 1.0

    return psi_i

def create_coupling_terms(L, heisen_delta):
    # Create coupling terms for the Hamiltonian, including isotropic zz terms and kinetic terms +- and -+.

    # i in range(L)
    # So for i=0, ..., L-2, (i + 1) % L = i + 1
    # For i=L-1, (i + 1) % L = 0

    list_H = [[heisen_delta, i, (i + 1) % L] for i in range(L)]

    # Suppose L = 4 and heisen_delta = 0.5:
    # list_H content: [[0.5, 0, 1], [0.5, 1, 2], [0.5, 2, 3], [0.5, 3, 0]]
    # This indicates the zz coupling term between each spin and its adjacent spin (with periodic boundary conditions), with a coupling constant of 0.5.

    list_H_kin = [[0.5, i, (i + 1) % L] for i in range(L)]
    # "kin" is kinetic energy

    # i is the position in the spin chain.
    # (i + 1) % L is the position under periodic boundary conditions, handling the coupling between the end and the start of the chain.

    # These coupling terms are then used to create the Hamiltonian:

    return [["+-", list_H_kin], ["-+", list_H_kin], ["zz", list_H]], list_H_kin
    # This function returns two values.
    # The first return value is a list containing three sets of coupling terms:
    # static_1 = ["+-", list_H_kin] and ["-+", list_H_kin] represent the kinetic terms +- and -+, respectively.
    # ["zz", list_H] represents the zz coupling term.
    # The second return value is list_H_kin.

def create_hamiltonian(basis, static_terms):
    return hamiltonian(static_terms, [], basis=basis, dtype=np.float64)

def get_ground_state(H_star):
    E_star, psi_star = H_star.eigsh(k=1, which="SA")
    # k=1 means we only need to find one (the smallest) eigenvalue and its corresponding eigenvector.
    # ‘SA’: Smallest (algebraic) eigenvalues.

    return E_star, psi_star

def main():
    parser = create_parser()
    args = parser.parse_args()

    N = args.N
    heisen_delta = 0.5

    basis = spin_basis_1d(N, S="1", kblock=0, pblock=1)

    # Initial state vector
    psi_i = create_initial_state(basis, N)

    static_1, list_H_kin = create_coupling_terms(N, heisen_delta)
    # static_1 contains kinetic terms and zz coupling terms.
    # list_H_kin contains only kinetic terms.

    # H_star is the H in the paper (section 4.3.2)
    H_star = create_hamiltonian(basis, static_1)

    # Target state vector
    E_star, psi_star = get_ground_state(H_star)

    print(f"System size: {N}")
    print("ground state energy: ", E_star / N)

    # (-1,) reshapes the array to a one-dimensional array, where -1 means automatically calculate the size of the dimension to match the total number of elements in the original array.
    psi_i = psi_i.reshape((-1,))
    psi_star = psi_star.reshape((-1,))

    H0 = hamiltonian([["+-", list_H_kin], ["-+", list_H_kin]], [], basis=basis, dtype=np.float64).toarray()
    H1 = hamiltonian([["zz", [[heisen_delta, i, (i + 1) % N] for i in range(N)]]], [], basis=basis, dtype=np.float64).toarray()
    # Note, H0 + H1 = H_star

    make_dir('quspin_data')

    # psi0_input: Initial state vector
    # psi1_input: Target state vector

    data_dict = {'psi0_input': psi_i.astype(complex),
                 'psi1_input': psi_star.astype(complex),
                 'H0': H0,
                 'H1': H1,
                 'H': H_star.toarray(),
                 'E_gs': E_star[0],
                 }
    dump(data_dict, f'quspin_data/heisenberg_N_{N}.pkl')

if __name__ == '__main__':
    main()
