import numpy as np

# Define the function h^(4)(q)
def h_2(q):
    w, x = q
    return np.array([w**2, x**2,
                     np.sqrt(2) * w * x])

# Define the vectors q_i satisfying ||q_i||^2 = 1
q_vectors = [np.random.randn(2) for _ in range(3)]

angle=np.random.uniform(0,2*np.pi) # Random angle between 0 and 2*pi
Rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.sin(angle)]])
q_vectors = [[1,0],[-1/2,np.sqrt(3)/2],[-1/2,-np.sqrt(3)/2]]
q_vectors = [q / np.linalg.norm(q) for q in q_vectors]

# Create matrix A^(2)
A_2 = np.array([h_2(q) for q in q_vectors])

# Define 1_d vector for d=4
d = 2
one_d = np.zeros(d*(d+1)//2)
one_d[:d] = 1

# Calculate B
I = np.eye(len(one_d))
B = np.linalg.inv(np.outer(one_d, one_d) + 2 * I).dot(A_2.T @ A_2)

# Check if B is symmetric
is_symmetric = np.allclose(B, B.T)

print(is_symmetric)
print(B)
print(A_2.T @ A_2)

C=np.array([[0, 1/2, 1.j/2],[1/2, 0, 0],[0, 1/2, -1.j/2]])
C=np.array([[0, 1/2, 1.j/2],[1/np.sqrt(2), 0, 0],[0, 1/2, -1.j/2]])
invC=np.linalg.inv(C)
print(invC)
print(invC.conj().T @ invC) 
# print(C @ C.T)