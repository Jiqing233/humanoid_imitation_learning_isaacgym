import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Backend: {matplotlib.get_backend()}")
print(f"Backend location: {matplotlib.backends.backend_tkagg.__file__}")

# Test if 3D plotting works
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0, 1], [0, 1], [0, 1])
plt.show()