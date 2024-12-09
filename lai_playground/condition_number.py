import numpy as np

#  定义矩阵 A
def create_matrix(x1, x2, x3):
    return np.array([
        [1, np.cos(x1), np.sin(x1)],
        [1, np.cos(x2), np.sin(x2)],
        [1, np.cos(x3), np.sin(x3)]
    ])

# 输入 x1, x2, x3 的值
x1 = 0.0
x2 = np.pi / 2
x3 = np.pi 

# 生成矩阵 A
A = create_matrix(x1, x2, x3)

# 计算矩阵 A 的条件数
cond_number = np.linalg.cond(A)
print(cond_number)

#####################################################
# 输入 x1, x2, x3 的值
x1 = np.random.uniform(0, 2 * np.pi)
x2 = np.random.uniform(0, 2 * np.pi)
x3 = np.random.uniform(0, 2 * np.pi)

# 生成矩阵 A
A = create_matrix(x1, x2, x3)

# 计算矩阵 A 的条件数
cond_number = np.linalg.cond(A)
print(cond_number)

#####################################################

from scipy.optimize import minimize

# 定义目标函数，计算条件数
def condition_number(x):
    A = create_matrix(x[0], x[1], x[2])
    return np.linalg.cond(A)

# 初始猜测的 x1, x2, x3
initial_guess = [0.0, np.pi / 2, np.pi]

# 使用优化方法找到使得条件数最小的 x1, x2, x3
result = minimize(condition_number, initial_guess, bounds=[(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])

# 最优解
x1_opt, x2_opt, x3_opt = result.x

# 生成最优矩阵 A
A_opt = create_matrix(x1_opt, x2_opt, x3_opt)

# 计算最优矩阵 A 的条件数
cond_number_opt = np.linalg.cond(A_opt)

print(f"Optimal x1: {x1_opt}, x2: {x2_opt}, x3: {x3_opt}")
print(f"Condition number: {cond_number_opt}")
