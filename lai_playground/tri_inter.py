import numpy as np

def generate_data(a, b, c, x_values):
    return [a + b * np.cos(x) + c * np.sin(x) for x in x_values]

def solve_coefficients(x_values, y_values):
    A = np.array([[1, np.cos(x), np.sin(x)] for x in x_values])
    B = np.array(y_values)
    return np.linalg.solve(A, B), A

# Test values
a_true = 2
b_true = 3
c_true = 1

# Generate random x values (distinct modulo 2*pi)
# np.random.seed(0)  # for reproducibility
x_values = np.random.uniform(0, 2*np.pi, 3)

# Generate corresponding y values
y_values = generate_data(a_true, b_true, c_true, x_values)

# Solve for coefficients and get the matrix
(coeffs, A) = solve_coefficients(x_values, y_values)

# Print the results
a_hat, b_hat, c_hat = coeffs
print("True coefficients: a = {}, b = {}, c = {}".format(a_true, b_true, c_true))
print("Estimated coefficients: a = {}, b = {}, c = {}".format(a_hat, b_hat, c_hat))
print("Condition number of the matrix A: {:.2f}".format(np.linalg.cond(A)))

# Verify the solution with another set of random x values
x_values2 = np.random.uniform(0, 2*np.pi, 3)
y_values2 = generate_data(a_true, b_true, c_true, x_values2)
(coeffs2, A2) = solve_coefficients(x_values2, y_values2)

# Print the results for the second set of x values
a_hat2, b_hat2, c_hat2 = coeffs2
print("Estimated coefficients with another set of x values: a = {}, b = {}, c = {}".format(a_hat2, b_hat2, c_hat2))
print("Condition number of the matrix A2: {:.2f}".format(np.linalg.cond(A2)))
