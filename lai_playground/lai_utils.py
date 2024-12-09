import jax.numpy as jnp
import jax
import numpy as np

# Pauli-X matrix
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)

# Pauli-Y matrix
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)

# Pauli-Z matrix
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

# Identity matrix
I = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)

PAULIS = [X, Y, Z, I]

def is_unitary(matrix, atol=1e-6, rtol=1e-6):
    """
    Verify if the given matrix is unitary.

    Parameters:
    - matrix (jnp.ndarray): The matrix to be checked.
    - atol (float): Absolute tolerance.
    - rtol (float): Relative tolerance.

    Returns:
    - bool: True if the matrix is unitary, False otherwise.
    """
    product = jnp.dot(matrix, matrix.conj().T)
    return jnp.allclose(product, jnp.eye(matrix.shape[0]), atol=atol, rtol=rtol)

def is_hermitian(matrix, atol=1e-6, rtol=1e-6):
    """
    Verify if the given matrix is Hermitian.

    Parameters:
    - matrix (jnp.ndarray): The matrix to be checked.
    - atol (float): Absolute tolerance.
    - rtol (float): Relative tolerance.

    Returns:
    - bool: True if the matrix is Hermitian, False otherwise.
    """
    return jnp.allclose(matrix, matrix.conj().T, atol=atol, rtol=rtol)

def generate_random_unitary(dim, prng_key, complex=True):
    """
    Generate a random dim x dim unitary matrix using QR decomposition.

    Parameters:
    - dim (int): Dimension of the matrix.
    - prng_key (jax.random.PRNGKey): Random key for reproducibility.
    - complex (bool): Whether to include complex numbers.

    Returns:
    - jnp.ndarray: A dim x dim unitary matrix.
    """
    subkey1, subkey2 = jax.random.split(prng_key)
    random_matrix = jax.random.normal(subkey1, (dim, dim)).astype(jnp.complex64)
    if complex:
        random_matrix += 1j * jax.random.normal(subkey2, (dim, dim))
    q, r = jnp.linalg.qr(random_matrix)
    d = jnp.diagonal(r)
    ph = d / jnp.abs(d)
    q *= ph
    return q

def generate_random_hermitian(dim, prng_key, complex=True):
    """
    Generate a random dim x dim Hermitian matrix.

    Parameters:
    - dim (int): Dimension of the matrix.
    - prng_key (jax.random.PRNGKey): Random key for reproducibility.
    - complex (bool): Whether to include complex numbers.

    Returns:
    - jnp.ndarray: A dim x dim Hermitian matrix.
    """
    subkey1, subkey2 = jax.random.split(prng_key)
    random_matrix = jax.random.normal(subkey1, (dim, dim)).astype(jnp.complex64)
    if complex:
        random_matrix += 1j * jax.random.normal(subkey2, (dim, dim))
    Hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2
    return Hermitian_matrix

def generate_random_H_paulis(n, prng_key):
    """
    Generate a random n-qubit Hermitian matrix using the Kronecker product of Pauli matrices.
    To ensure H is Hermitian and Involutory, we randomly select from I, X, Y, Z to construct H.

    Parameters:
    - n (int): Number of qubits.
    - prng_key (jax.random.PRNGKey): Pseudorandom number generator key.

    Returns:
    - jnp.ndarray: Random n-qubit Hermitian matrix.
    """
    subkeys = jax.random.split(prng_key, n)
    H = PAULIS[jax.random.choice(subkeys[0], 4)]
    for i in range(1, n):
        H = jnp.kron(H, PAULIS[jax.random.choice(subkeys[i], 4)])
    return H

def U_j(theta_j, H, method='exponential'):
    """
    Calculate the rotation-like gate matrix U_j(theta_j) based on the specified method:
    
    Parameters:
    - theta_j (float): The rotation angle.
    - H (jnp.ndarray): The Hermitian matrix to be used in the gate.
    - method (str): The method to use for constructing the gate. Options are:
      - 'exponential': Uses the formula e^{-(i / 2) H_j \theta_j}.
      - 'trigonometric': Uses the formula cos(theta_j / 2) * I - i * sin(theta_j / 2) * H.
    
    Returns:
    - jnp.ndarray: The resulting rotation-like gate matrix.
    
    Raises:
    - ValueError: If the method specified is not 'exponential' or 'trigonometric'.
    """
    if method == 'exponential':
        exponent = -1j / 2 * H * theta_j
        return jax.scipy.linalg.expm(exponent)
    elif method == 'trigonometric':
        return jnp.cos(theta_j / 2) * jnp.eye(H.shape[0]) - 1j * jnp.sin(theta_j / 2) * H
    else:
        raise ValueError("Unknown method: choose 'exponential' or 'trigonometric'")


def create_uniform_superposed_state(n):
    """
    Create a uniform superposed state |+⟩^⊗n.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - jnp.ndarray: The uniform superposed state.
    """
    initial_state = jnp.zeros(2**n)
    initial_state = initial_state.at[0].set(1)

    # Apply Hadamard gate to each qubit
    H = (1/jnp.sqrt(2)) * jnp.array([[1, 1], [1, -1]])
    H_n = H

    # Tensor product to create H⊗n
    for _ in range(n - 1):
        H_n = jnp.kron(H_n, H)

    # Apply H⊗n to the initial state
    superposed_state = jnp.dot(H_n, initial_state)

    return superposed_state

def create_ket_zero_state(n):
    """
    Create the initial state |0⟩^⊗n.

    Parameters:
    - n (int): Number of qubits.

    Returns:
    - jnp.ndarray: The initial state |0⟩^⊗n.
    """
    return jnp.eye(2**n)[:, 0]


