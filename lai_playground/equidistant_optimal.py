import numpy as np

def interp_matrix(theta_vals):
    # create interpolation matrix
    return np.array([[1, np.cos(val), np.sin(val)] for val in theta_vals])

def mse(theta_vals):
    # create interpolation matrix
    A = interp_matrix(theta_vals)
    return np.trace(np.linalg.inv(A.T @ A))

theta_vals1 = [0, np.pi*2/3, np.pi*4/3]
A1=interp_matrix(theta_vals1)

theta_vals2 = np.linspace(-np.pi, np.pi, 3, endpoint=False)
A2=interp_matrix(theta_vals2)

shift=np.random.rand()
theta_vals3 = theta_vals2 + shift
A3=interp_matrix(theta_vals3)

print(np.linalg.cond(A1))
print(mse(theta_vals1))

print(np.linalg.cond(A2))
print(mse(theta_vals2))

print(np.linalg.cond(A3))
print(mse(theta_vals3))




