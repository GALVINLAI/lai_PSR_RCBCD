import argparse
import numpy as np
import scipy.linalg as la
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from utils import dump, make_dir

# Set the random seed for reproducibility
np.random.seed(42)

def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add the system size argument
    parser.add_argument('--N', type=int, default=3, help='System size')
    
    return parser

def create_hamiltonian(N):
    # Compute Hilbert space basis
    basis = spin_basis_1d(L=N, pauli=True, pblock=1)

    # Define Hamiltonian terms
    zz_term = [[1.0, i, i + 1] for i in range(N - 1)]
    z_term = [[1.0, i] for i in range(N)]
    x_term = [[1.0, i] for i in range(N)]

    # Compute operator string lists
    static_C = [["zz", zz_term], ["z", z_term]]
    static_B = [["x", x_term]]

    # Create Hamiltonian operators
    H_C = hamiltonian(static_C, [], basis=basis, dtype=np.float64)
    H_B = hamiltonian(static_B, [], basis=basis, dtype=np.float64)
    
    return H_C, H_B

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Access the system size
    print(f"System size: {args.N}")
    N = args.N

    # Compute Hamiltonian operators
    H_C, H_B = create_hamiltonian(N)
    
    # Define H0 and H1 operators
    # .toarray(): Returns copy of a hamiltonian object at time time as a dense array.
    H0 = -H_C.toarray() + 4.0 * H_B.toarray()
    H1 = -H_C.toarray() - 4.0 * H_B.toarray()
    
    H0_i = -H_C.toarray() + 2.0 * H_B.toarray()
    H1_i = -H_C.toarray() - 2.0 * H_B.toarray()

    # Find (all) eigenvalues and eigenvectors
    w, v = la.eig(H0_i)
    ww, vv = la.eig(H1_i)

    #  initial state
    idx = np.argmin(w)
    psi0_input = v[:, idx].astype(complex)

    # desired target state
    idx = np.argmin(ww)
    psi1_input = vv[:, idx].astype(complex)

    # Print psi_inputs and H operators
    print(f"psi0_input: {psi0_input}, shape: {psi0_input.shape}")
    print(f"psi1_input: {psi1_input}, shape: {psi1_input.shape}")
    print(f"H0: {H0}, shape: {H0.shape}")
    print(f"H1: {H1}, shape: {H1.shape}")
    
    # Create directory for data
    make_dir('quspin_data')
    data_dict = {
        'psi0_input': psi0_input,
        'psi1_input': psi1_input,
        'H0': H0,
        'H1': H1
    }

    # Save data to pickle
    dump(data_dict, f'quspin_data/tfim_N_{N}.pkl')

if __name__ == "__main__":
    main()
