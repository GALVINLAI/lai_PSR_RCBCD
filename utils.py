import os
import pickle  # Used for serializing and deserializing Python objects.
import re
from functools import reduce
import jax.numpy as np

def make_dir(path):
    """
    Create a new directory if it does not exist.

    Parameters:
    path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        # The parameter exist_ok=True indicates that if the directory already exists, no error will be raised, and the operation will be simply ignored.

def load(filename):
    """
    Load data from a pickle file.

    Parameters:
    filename (str): The name of the pickle file.

    Returns:
    loaded: The data loaded from the pickle file.
    """
    assert filename.endswith('.pkl'), "File must be a '.pkl' file"
    
    with open(filename, 'rb') as file:
        # The 'rb' parameter indicates opening the file in binary read mode, which is necessary for reading pickle files.
        loaded = pickle.load(file)
        
    return loaded

def dump(content, filename):
    """
    Save data to a pickle file.

    Parameters:
    content : The data to be saved.
    filename (str): The name of the pickle file.
    """
    assert filename.endswith('.pkl'), "File must be a '.pkl' file"
    
    with open(filename, 'wb') as file:
        # The 'wb' parameter indicates opening the file in binary write mode, which is necessary for writing pickle files.
        pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)
        # The parameter protocol=pickle.HIGHEST_PROTOCOL specifies using the highest protocol version to ensure the serialized data can be compatible with future Python versions.

# Define Pauli matrices and identity matrix
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# Create a mapping from string to corresponding matrix
MATRIX_MAP = {'x': X, 'y': Y, 'z': Z, 'i': I}

# Example to compute the matrix representation of X0*Y1*Z2, where X0 denotes applying Pauli-X matrix to the 0th qubit,
# Y1 denotes applying Pauli-Y matrix to the 1st qubit, and Z2 denotes applying Pauli-Z matrix to the 2nd qubit
def term_to_matrix(term, n_qubits):
    """Convert a term to its matrix representation."""
    # Initialize list with identity operators
    operators = [I] * n_qubits

    for component in term.split("*"):
        # Split the term by asterisk * into multiple components. For example, 'X0*Y1*Z2' is split into ['X0', 'Y1', 'Z2'].
        component = component.strip()
        if not component:  # Skip if the component is empty.
            continue
        operator = component[0].lower()  # Extract the operator (e.g., 'x', 'y', 'z', 'i') and convert to lowercase.
        qubit = int(component[1:])  # Extract the qubit number on which the operator acts.
        operators[qubit] = MATRIX_MAP[operator]  # Replace the identity matrix at the corresponding position with the operator matrix.

    # Compute the tensor product of the operators
    return reduce(np.kron, operators)  # Compute the tensor product X ⊗ Y ⊗ Z to get the final matrix representation.
    # reduce is a function in Python, imported from the functools module, used for cumulative computation on elements in a sequence.
    # The working principle of reduce is to apply the specified binary function to the first two elements of the sequence,
    # then apply the result to the next element, and so on until all elements in the sequence are processed.

def hamiltonian_to_matrix(hamiltonian_str, n_qubits=-1):
    """
    Convert a Hamiltonian to its matrix representation.
    """

    # Handle the minus signs in the string, converting them to addition terms with a negative sign.
    hamiltonian_str = '+-'.join(map(str.strip, hamiltonian_str.split('-')))
    # hamiltonian_str.split('-'): Split the original Hamiltonian string by minus signs '-', generating a list of substrings.
    # For example, if the original string is "-3.0 + 0.5 * Z0 * Z2 - 0.25 * Z1 * Z0", it will be split into ['-3.0 + 0.5 * Z0 * Z2 ', ' 0.25 * Z1 * Z0'].

    # map(str.strip, ...): Apply the strip method to each substring after splitting to remove leading and trailing whitespace.
    # Continuing the previous example, it becomes ['-3.0 + 0.5 * Z0 * Z2', '0.25 * Z1 * Z0'].

    # '+-'.join(...): Reconnect the processed substrings with '+-'.
    # This step makes each minus sign explicit as part of a negative number.
    # For example, the final result will be '-3.0 + 0.5 * Z0 * Z2 +- 0.25 * Z1 * Z0'.

    terms = list(filter(None, map(str.strip, hamiltonian_str.split('+'))))
    # Split the Hamiltonian string by plus signs '+' to generate a list of substrings.
    # map(str.strip, ...): Apply the strip method to each substring after splitting to remove leading and trailing whitespace.
    # Continuing the previous example, it becomes ['-3.0', '0.5 * Z0 * Z2', '- 0.25 * Z1 * Z0'].

    # list(filter(None, ...)): Filter out empty strings.
    # Although empty strings are not generated in this particular example, this is a safety measure in case there are extraneous plus signs or spaces in the input string.

    if n_qubits < 0:  # This code aims to automatically calculate the number of qubits involved when the user does not specify it.
        # If n_qubits is less than 0, it means the user did not specify the number of qubits, and it needs to be calculated automatically.
        # Get the number of qubits involved
        qubit_indices = [int(qubit[1:]) for term in terms for qubit in re.findall(r'[x|y|z|X|Y|Z|i|I]\d+', term)]
        # for term in terms: This is a loop that iterates through each element in the terms list.
        # for qubit in ...: This is a nested loop that further iterates over all qubit representations matching the regular expression in each term.

        # Extract the indices of the qubits involved in all terms. The slice qubit[1:] considers multi-digit numbers, hence the use of 1:.
        n_qubits = max(qubit_indices) + 1

    # Initialize the Hamiltonian matrix
    H = np.zeros((2**n_qubits, 2**n_qubits))

    for term in terms:
        coef = 1.0
        ops = []  # Initialize the operator list as empty.
        for component in term.split("*"):  # Split the term by asterisk * into multiple components.
            try:
                coef *= float(component)
                # Try to convert the component to a float and multiply it into the coefficient.
                # If the component is a number, this step will succeed and update the coefficient.
            except ValueError:
                # If the component is not a number (i.e., it's an operator), add it to the operator list ops.
                ops.append(component)
        # Add the term to the Hamiltonian matrix
        # The term_to_matrix function converts a single term to its matrix representation.
        # Convert the operator list to its matrix representation, multiply by the coefficient, and add to the Hamiltonian matrix H.
        H += coef * term_to_matrix("*".join(ops), n_qubits)

    return H

# Test the function
# hamiltonian_str = "-3.0 + 0.5 * Z0 + 0.25 * Z1 + 0.25 * Z2 + 0.5 * Z3 + 0.75 * Z0*Z2 - 0.25 * Z1*Z2 + 0.25 * Z0*Z1 + 0.25 * Z0*Z3 + 0.75 * Z1*Z3 + 0.25 * Z2*Z3 - 0.25 * Z0*Z1*Z2 - 0.25 * Z1*Z2*Z3"
# print(hamiltonian_to_matrix(hamiltonian_str).astype(np.float64))
