def test_simple_with_draw():
    """Test simple 3D plotting using draw() method instead of plt.show()"""
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    import numpy as np

    # Create figure and 3D axis using the same method as Matplotlib3DPlotter
    fig = plt.figure()
    # ax = p3.Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # Create some test data
    x = np.array([0, 0])
    y = np.array([0, 0])
    z = np.array([0, 1])

    # Plot the line
    ax.plot(x, y, z, 'b-', linewidth=2)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Simple 3D Line Test with draw()')

    # Use draw() method instead of plt.show()
    def draw():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.00001)

    # Draw the plot
    draw()

    # Keep the window open
    input("Press Enter to close the plot...")

if __name__ == "__main__":
    test_simple_with_draw() 