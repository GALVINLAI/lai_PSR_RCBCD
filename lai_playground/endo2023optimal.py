import numpy as np

# Define the function h^(4)(q)
def h_4(q):
    w, x, y, z = q
    return np.array([w**2, x**2, y**2, z**2, 
                     np.sqrt(2) * w * x, np.sqrt(2) * w * y, np.sqrt(2) * w * z, 
                     np.sqrt(2) * x * y, np.sqrt(2) * x * z, np.sqrt(2) * y * z])

# Define the vectors q_i satisfying ||q_i||^2 = 1
q_vectors = [np.random.randn(4) for _ in range(10)]
q_vectors = [q / np.linalg.norm(q) for q in q_vectors]

# Create matrix A^(4)
A_4 = np.array([h_4(q) for q in q_vectors])

# Define 1_d vector for d=4
d = 4
one_d = np.zeros(d*(d+1)//2)
one_d[:d] = 1

# Calculate B
I = np.eye(len(one_d))
B = np.linalg.inv(np.outer(one_d, one_d) + 2 * I).dot(A_4.T @ A_4)

# Check if B is symmetric
is_symmetric = np.allclose(B, B.T)

is_symmetric
