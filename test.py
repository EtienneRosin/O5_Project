import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Generate the grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Set up the figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Initial surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

def update(frame):
    global surf
    # Remove old surface
    surf.remove()
    # Generate new Z values
    Z = np.sin(np.sqrt(X**2 + Y**2) + frame / 10)
    # Plot new surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    return surf,

# ax.set_axis_off()
# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# Save the animation with specified DPI
Writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'))
ani.save("high_quality_animation.mp4", writer=Writer, dpi=600)

plt.show()
