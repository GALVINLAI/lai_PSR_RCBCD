import numpy as np

def interp_matrix(theta_vals, max_n):
    # 创建插值矩阵，每行依次包含 1, cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(nθ), sin(nθ)
    return np.array([[1] + [func(k * val) for k in range(1, max_n + 1) for func in (np.cos, np.sin)] for val in theta_vals])

max_n = 5
# 生成2n+1个等距theta值，从0到2π，不包括终点2π
theta_vals = np.linspace(0, 2 * np.pi, 2*max_n+1, endpoint=False)
A = interp_matrix(theta_vals, max_n)

# 打印插值矩阵
print("插值矩阵 A:")
print(A)

# 打印矩阵 A 的转置乘以 A
print("\n矩阵 A 的转置乘以 A (应该是对角矩阵):")
ATA = A.T @ A
print(ATA)

# 计算并打印 A.T @ A 的逆矩阵的迹
print("\n矩阵 A.T @ A 的逆矩阵的迹:")
trace_inv_ATA = np.trace(np.linalg.inv(ATA))
print(trace_inv_ATA)

# 计算并打印矩阵 A 的条件数
print("\n矩阵 A 的条件数 （因该是√2）:")
cond_A = np.linalg.cond(A)
print(cond_A)

# 随机选择A的两行并计算内积
rows = np.random.choice(A.shape[0], 2, replace=False)
row1 = A[rows[0]]
row2 = A[rows[1]]
inner_product = np.dot(row1, row2)
print(f"\nInner product of row {rows[0]} and row {rows[1]} of matrix A: {inner_product}")