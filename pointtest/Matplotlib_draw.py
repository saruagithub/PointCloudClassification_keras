import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pointtest import provider
current_data, current_label = provider.loadDataFile('pointtest/modelnet40_ply_hdf5_2048/ply_data_test1.h5')

points = current_data[51]
# skip = 1   # Skip every n points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# point_range = range(0, points.shape[0], skip) # skip points to prevent crash
point_range = range(0, points.shape[0]) # skip points to prevent crash
ax.scatter(points[point_range, 0],   # x
           points[point_range, 1],   # y
           points[point_range, 2],   # z
           c=points[point_range, 2], # height data for color
           cmap='spectral',
           marker="x")
ax.axis('scaled')  # {equal, scaled}
plt.show()