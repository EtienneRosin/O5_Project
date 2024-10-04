import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from schrodinger_engine.fourier import FourierBasis
from schrodinger_engine.wave_packets import gaussian_wave_packet, gaussian_envelope

# Animation setup
fig, ax = plt.subplots()

# Domain parameters
L = 50
N_x = 1000  # Augmentation du nombre de points
dx = L/N_x
fourier_basis = FourierBasis(L=L, N_x=N_x)

# Time parameters
T = 1000
N_t = 2000
dt = T/N_t

# lst_t = np.linspace(start=0, stop=T, num=N_t, endpoint=True)
lst_t = np.arange(start=0, stop=T, step=dt)
N_t = len(lst_t)

# Initial conditions parameters
# center = L/6
center = 0
amplitude = 1
width = L/20
# width = 15
frequency = 4*dx
print(f"{width = }, {frequency = }, {dx = }")
phase = 0

# Define the initial wave packet


def psi_0(x):
    return gaussian_wave_packet(x, center=center, amplitude=amplitude, width=width, frequency=frequency, phase=phase)


# def psi_0(x):
#     return gaussian_envelope(x, center=center, amplitude=amplitude, width=width)


# Hamiltonian matrix

k_vals = fourier_basis.k_values()
K = np.diag(0.5 * k_vals**2)  # Hamiltonian for the kinetic energy

# K = 0.5 * sp.linalg.dft(n=N_x)**2

# Compute the time evolution operator
U = sp.sparse.linalg.expm(- 1j * K * dt)

# Initialize the solution array
solution = np.zeros((N_t, N_x), dtype=complex)
solution[0, :] = psi_0(fourier_basis.mesh) / \
    np.linalg.norm(psi_0(fourier_basis.mesh))



# Time evolution (pre-compute the full time evolution for animation)
for i in range(1, N_t):
    fft = fourier_basis.project_function(solution[i - 1, :])
    tmp = U @ fft
    solution[i, :] = sp.fft.ifft((np.sqrt(L)/dx) * tmp)
    # solution[i, :] = U @ solution[i - 1, :]

# Colormap for phase representation
color_map = "hsv"
norm = plt.Normalize(-np.pi, np.pi)
cmap = plt.get_cmap(color_map)

# Initial plot using LineCollection
x = fourier_basis.mesh
y = np.abs(solution[0, :])**2
phases = np.angle(solution[0, :])

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc_label = r"$|\left< x | \psi(t) \right> |^2$"
lc = LineCollection(segments, cmap=cmap, norm=norm, label=lc_label)
lc.set_array(phases)
ax.add_collection(lc)
real_plot_label = r"$\Re \left(\left< x | \psi(t) \right> \right)$"
real_plot, = ax.plot(x, solution[0, :].real, label=real_plot_label, c = "blue")

# Set axis labels, title, and grid
ax.set(xlabel = r"$x$", title = "Wave Packet Evolution with Phase", xlim = (x.min(), x.max()))
# ax.set_xlabel("$x$")
# ax.set_ylabel()
# ax.set_title("Wave Packet Evolution with Phase")
ax.grid()
# ax.set_xlim(x.min(), x.max())

norm_label = r"$\left\| \left| \psi(t) \right> \right\|_{L^2\left(\Omega\right)}$"
norm_display = ax.text(0.5, 0.5, s=f"{norm_label} = {np.linalg.norm(solution[0, :]):.3g}",
                       transform=ax.transAxes)
# ax.set_ylim(0, np.max(np.abs(solution)**2) * 1.1)
ax.legend()
# ax.set_aspect("equal")
fig.colorbar(lc, label="phase")
# Initialization function


def init():
    y = np.abs(solution[0, :])**2
    phases = np.angle(solution[0, :])

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_segments(segments)
    lc.set_array(phases)
    real_plot.set_ydata(solution[0, :].real)
    norm_display.set_text(
        f"{norm_label} = {np.linalg.norm(solution[0, :]):.3g}")
    return lc, real_plot, norm_display,

# Update function for animation


def update(frame):
    y = np.abs(solution[frame, :])**2
    phases = np.angle(solution[frame, :])

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_segments(segments)
    lc.set_array(phases)
    real_plot.set_ydata(solution[frame, :].real)
    ax.set(title=f"t = {lst_t[frame]:.3g}")

    norm_display.set_text(
        f"{norm_label} = {np.linalg.norm(solution[frame, :]):.3g}")
    return lc, real_plot, norm_display,


# Create the animation
ani = FuncAnimation(fig, update, frames=N_t,
                    init_func=init, blit=False, interval=50)

# norm_display the animation
plt.show()
