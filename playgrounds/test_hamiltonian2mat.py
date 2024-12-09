import sys
sys.path.append("..") # Adds higher directory to python modules path.
import jax.numpy as np
from utils import hamiltonian_to_matrix



def test_factor():
    def old_code():
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
        # print(H)

        # check the ground state 
        for x in np.linalg.eigh(H)[1][:, :2].T:
            print(np.nonzero(x))
        # print(int('0110', 2), int('1001', 2))

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

        # print(total_X_operator)
        return H, total_X_operator

    def new_code():
        hamiltonian_str = "-3.0 + 0.5 * Z0 + 0.25 * Z1 + 0.25 * Z2 + 0.5 * Z3 + 0.75 * Z0*Z2 - 0.25 * Z1*Z2 + 0.25 * Z0*Z1 + 0.25 * Z0*Z3 + 0.75 * Z1*Z3 + 0.25 * Z2*Z3 - 0.25 * Z0*Z1*Z2 - 0.25 * Z1*Z2*Z3"
        H = hamiltonian_to_matrix(hamiltonian_str)
        
        hamiltonian_str = 'X0 + X1 + X2 + X3'
        total_X_operator = hamiltonian_to_matrix(hamiltonian_str)
        return H, total_X_operator
    
    # Run both codes
    H_old, total_X_operator_old = old_code()
    H_new, total_X_operator_new = new_code()

    # Check the result
    assert np.allclose(H_old, H_new), "Hamiltonian Matrices are not equal"
    assert np.allclose(total_X_operator_old, total_X_operator_new), "Total X operator Matrices are not equal"

    print("The test passed successfully!")



def test_tsp():
    def old_code():
            
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

        return H
    
    def new_code():
        data = """
        IIIIIIIIZ       (-100069.5+0j)
        IIIIZIIII       (-100055.5+0j)
        IIIIZIIIZ       (12+0j)
        IIIIIIIZI       (-100069.5+0j)
        IIIZIIIII       (-100055.5+0j)
        IIIZIIIZI       (12+0j)
        IIIIIIZII       (-100069.5+0j)
        IIIIIZIII       (-100055.5+0j)
        IIIIIZZII       (12+0j)
        IZIIIIIII       (-100077+0j)
        IZIIIIIIZ       (22.75+0j)
        ZIIIIIIII       (-100077+0j)
        ZIIIIIIZI       (22.75+0j)
        IIZIIIIII       (-100077+0j)
        IIZIIIZII       (22.75+0j)
        IIIIIZIZI       (12+0j)
        IIIIZIZII       (12+0j)
        IIIZIIIIZ       (12+0j)
        IZIIIZIII       (15.75+0j)
        ZIIIZIIII       (15.75+0j)
        IIZZIIIII       (15.75+0j)
        IIZIIIIZI       (22.75+0j)
        IZIIIIZII       (22.75+0j)
        ZIIIIIIIZ       (22.75+0j)
        IIZIZIIII       (15.75+0j)
        IZIZIIIII       (15.75+0j)
        ZIIIIZIII       (15.75+0j)
        IIIIIZIIZ       (50000+0j)
        IIZIIIIIZ       (50000+0j)
        IIZIIZIII       (50000+0j)
        IIIIZIIZI       (50000+0j)
        IZIIIIIZI       (50000+0j)
        IZIIZIIII       (50000+0j)
        IIIZIIZII       (50000+0j)
        ZIIIIIZII       (50000+0j)
        ZIIZIIIII       (50000+0j)
        IIIIIIIZZ       (50000+0j)
        IIIIIIZIZ       (50000+0j)
        IIIIIIZZI       (50000+0j)
        IIIIZZIII       (50000+0j)
        IIIZIZIII       (50000+0j)
        IIIZZIIII       (50000+0j)
        IZZIIIIII       (50000+0j)
        ZIZIIIIII       (50000+0j)
        ZZIIIIIII       (50000+0j)
        """

        # Split the data into lines
        lines = data.strip().split("\n")

        # Prepare empty lists to hold the extracted data
        first_column = []
        second_column = []

        n_qubits = 9

        # Process each line
        for line in lines:
            # Split the line into columns
            columns = line.split()
            
            # Get the first column string
            str_val = columns[0]

            # Find the indices of 'Z' characters
            z_indices = [i for i, char in enumerate(str_val, start=0) if char == 'Z']

            # Output 'zi' or 'zizj' depending on the number of 'Z's
            if len(z_indices) == 1:
                op_str = f"z{n_qubits - 1 - int(z_indices[0])}"
            elif len(z_indices) == 2:
                op_str = f"z{n_qubits -1 - int(z_indices[0])} * z{n_qubits -1 - int(z_indices[1])}"

            # Append the data to the corresponding lists
            first_column.append(op_str)
            second_column.append(float(eval(columns[1]).real))

        # Now, 'first_column' contains the strings from the first column,
        # and 'second_column' contains the float numbers from the second column.

        # print("First column:", first_column)
        # print("Second column:", second_column)

        ham_str = ''
        for op, coeff in zip(first_column, second_column):
            if ham_str == '':
                ham_str = f'{coeff} * {op} '
                continue
            if coeff < 0: 
                ham_str = ham_str + f'{coeff} * {op} '
            else:
                ham_str = ham_str + f'+ {coeff} * {op} '

        ham_str = '600303.0 ' + ham_str

        H = hamiltonian_to_matrix(ham_str, n_qubits)
        return H
    
    # Run both codes
    H_old = old_code()
    H_new = new_code()

    # Check the result
    assert np.allclose(H_old, H_new), "Hamiltonian Matrices are not equal"

    print("The test passed successfully!")
    
def test_maxcut():
    def old_code():   
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
        return H 
    
    def new_code():
        data = """
        IIZZ    (0.5+0j)
        IZIZ    (0.5+0j)
        IZZI    (0.5+0j)
        ZIIZ    (0.5+0j)
        ZZII    (0.5+0j)
        """

        # Split the data into lines
        lines = data.strip().split("\n")

        # Prepare empty lists to hold the extracted data
        first_column = []
        second_column = []

        n_qubits = 4

        # Process each line
        for line in lines:
            # Split the line into columns
            columns = line.split()
            
            # Get the first column string
            str_val = columns[0]

            # Find the indices of 'Z' characters
            z_indices = [i for i, char in enumerate(str_val, start=0) if char == 'Z']

            # Output 'zi' or 'zizj' depending on the number of 'Z's
            if len(z_indices) == 1:
                op_str = f"z{n_qubits - 1 - int(z_indices[0])}"
            elif len(z_indices) == 2:
                op_str = f"z{n_qubits -1 - int(z_indices[0])} * z{n_qubits -1 - int(z_indices[1])}"

            # Append the data to the corresponding lists
            first_column.append(op_str)
            second_column.append(float(eval(columns[1]).real))

        # Now, 'first_column' contains the strings from the first column,
        # and 'second_column' contains the float numbers from the second column.

        # print("First column:", first_column)
        # print("Second column:", second_column)

        ham_str = ''
        for op, coeff in zip(first_column, second_column):
            if ham_str == '':
                ham_str = f'{coeff} * {op} '
                continue
            if coeff < 0: 
                ham_str = ham_str + f'{coeff} * {op} '
            else:
                ham_str = ham_str + f'+ {coeff} * {op} '

        ham_str = '0.5 - 3 * z0 ' + ' + ' + ham_str
        
        H = hamiltonian_to_matrix(ham_str, n_qubits)
        return H
    
    # Run both codes
    H_old = old_code()
    H_new = new_code()

    # Check the result
    assert np.allclose(H_old, H_new), "Hamiltonian Matrices are not equal"

    print("The test passed successfully!")
            



if __name__ == "__main__":
    # Run the test
    test_factor()
    test_tsp()
    test_maxcut()