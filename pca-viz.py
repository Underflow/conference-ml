import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plane definition
orig = [0, 0, 0]
direction = [[0, 1, 0],
             [1, 0, 1]]

n = 200
for _ in range(n):
    (a, b) = (np.random.rand(), np.random.rand())
    disp = np.random.normal(0, 0.05, size = 3)
    point = [orig[0] + a * direction[0][0] + b * direction[1][0] + disp[0],
             orig[1] + a * direction[0][1] + b * direction[1][1] + disp[1],
             orig[2] + a * direction[0][2] + b * direction[1][2] + disp[2]]
    print(point)
    ax.scatter(point[0], point[1], point[2], c='r', marker='o')


ax.set_xlabel('Feature A')
ax.set_ylabel('Feature B')
ax.set_zlabel('Feature C')

point  = np.array([2, 2, 2])
normal = np.array([-1, 0, 1])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(2), range(2))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
ax.plot_surface(xx, yy, z, alpha=0.2)

plt.show()

