import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate points
n_points = 10000
points = []

while len(points) < n_points:
    x1, x2, x3 = np.sort(np.random.uniform(0, np.pi, 3))
    if x1 < x2 < x3:
        points.append([x1, x2, x3])

points = np.array(points)

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=10)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Visualization of the Convex Set: 0 ≤ x1 < x2 < x3 ≤ π')

plt.colorbar(ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=10), 
             label='x3 value')

plt.tight_layout()
plt.show()