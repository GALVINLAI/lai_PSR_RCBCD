
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成自变量 x
x = np.linspace(0, 10, 100)

# 生成因变量 y，假设真实的线性关系为 y = 3x + 7，并添加一些随机噪声
y = 3 * x + 7 + np.random.randn(100) * 2

# 构建矩阵 A，其中一列是自变量 x，另一列是常数 1
A = np.vstack([x, np.ones(len(x))]).T

# 使用线性最小二乘法求解线性方程组
solution = np.linalg.lstsq(A, y, rcond=None)
slope, intercept = solution[0]

print(f"Slope: {slope}, Intercept: {intercept}")

# 使用拟合参数生成拟合直线
y_fit = slope * x + intercept

# 可视化原始数据和拟合直线
plt.scatter(x, y, label='Data Points')
plt.plot(x, y_fit, color='red', label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Least Squares Fitting')
plt.legend()
plt.show()
