import numpy as np
from scipy.optimize import minimize

# 定义矩阵 A
def create_matrix(x1, x2, x3):
    return np.array([
        [1, np.cos(x1), np.sin(x1)],
        [1, np.cos(x2), np.sin(x2)],
        [1, np.cos(x3), np.sin(x3)]
    ])

# 定义目标函数，计算条件数
def condition_number(x):
    A = create_matrix(x[0], x[1], x[2])
    return np.linalg.cond(A)

# 初始猜测的 x1, x2, x3
# initial_guess = [0.0, np.pi / 2, np.pi]
initial_guess = [np.pi / 4, np.pi / 2, np.pi]
# initial_guess =  np.random.uniform(0, 2 * np.pi, 3)
# 使用不同的求解器
solvers = ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'CG', 'BFGS']

results = []
# initial_guess =  np.random.uniform(0, 2 * np.pi, 3)

for solver in solvers:
    result = minimize(condition_number, initial_guess, method=solver)
    x1_opt, x2_opt, x3_opt = result.x
    A_opt = create_matrix(x1_opt, x2_opt, x3_opt)
    cond_number_opt = np.linalg.cond(A_opt)
    results.append({
        'solver': solver,
        'x1': x1_opt,
        'x2': x2_opt,
        'x3': x3_opt,
        'condition_number': cond_number_opt
    })

# 输出结果
for res in results:
    print(f"Solver: {res['solver']}")
    print(f"Optimal x1: {res['x1']}, x2: {res['x2']}, x3: {res['x3']}")
    print(f"Condition number: {res['condition_number']}\n")

