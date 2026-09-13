import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 创建网格
w_values = np.arange(0.0, 4.1, 0.1)
b_values = np.arange(0.0, 4.1, 0.1)

# 存储结果
mse_matrix = np.zeros((len(w_values), len(b_values)))

# 计算每个 (w, b) 组合的 MSE
for i, w in enumerate(w_values):
    for j, b in enumerate(b_values):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, w, b)
        mse_matrix[i, j] = l_sum / 3

# 创建网格坐标
W, B = np.meshgrid(w_values, b_values)

# 3D 可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(W, B, mse_matrix.T, cmap='viridis', alpha=0.8)

# 添加标签
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE Loss')
ax.set_title('Loss Surface for Linear Regression')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()