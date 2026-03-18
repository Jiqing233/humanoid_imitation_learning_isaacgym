from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ..core import BasePlotterTask, BasePlotterTasks
from ..plt_plotter import Matplotlib3DPlotter
from ..simple_plotter_tasks import Draw3DDots, Draw3DLines

print("\n=== Creating Test Tasks ===")
print("Creating lines task...")
task = Draw3DLines(task_name="test", 
    lines=np.array([[[0, 0, 0], [0, 0, 1]], [[0, 1, 1], [0, 1, 0]]]), color="blue")
print(f"Lines data shape: {task.raw_data.shape}")

print("\nCreating dots task...")
task2 = Draw3DDots(task_name="test2", 
    dots=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]), color="red")
print(f"Dots data shape: {task2.raw_data.shape}")

print("\nCombining tasks...")
task3 = BasePlotterTasks([task, task2])
print(f"Number of tasks: {len(list(task3))}")

print("\nCreating plotter...")
plotter = Matplotlib3DPlotter(cast(BasePlotterTask, task3))
print("Plotter created")

# Create artists
plotter._create_impl(list(task3))

# Create a new figure for testing
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Get the first artist from the cache
for task_name, artists in plotter._artist_cache.items():
    if isinstance(artists, list):
        # Get the first line artist
        artist = artists[0]
        print(f"\nPlotting artist from task {task_name}:")
        print(f"Type: {type(artist)}")
        print(f"Data: {artist.get_data_3d()}")
        
        # Get the data from the artist
        x, y, z = artist.get_data_3d()
        
        # Plot the line
        ax.plot(x, y, z, color=artist.get_color(), linewidth=artist.get_linewidth())
        break  # Only plot the first artist

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Single Artist Test')

# Set equal aspect ratio
ax.set_box_aspect([1,1,1])

# Set a specific view angle
ax.view_init(elev=20, azim=45)

# Show the plot
plt.show()

# Keep the window open
input("Press Enter to close the plot...")