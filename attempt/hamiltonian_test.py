import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from project.schrodinger_engine.fourier import FourierBasis
from project.schrodinger_engine.wave_packets import gaussian_wave_packet, gaussian_envelope

# Animation setup
fig, ax = plt.subplots()

# Domain parameters
L = 10
N_x = 1000
fourier_basis = FourierBasis(L=L, N_x=N_x)

# Time parameters
T = 10
N_t = 2000  # Increase the time steps for smoother animation
lst_t = np.linspace(start=0, stop=T, num=N_t, endpoint=True)

# Initial conditions parameters
center = L/4
amplitude = 0.1
width = L/20
frequency = 2
phase = 0

# Define the initial wave packet


def psi_0(x):
    return gaussian_wave_packet(x, center=center, amplitude=amplitude, width=width, frequency=frequency, phase=phase)


# def psi_0(x):
#     return gaussian_envelope(x, center=center, amplitude=amplitude, width=width)

    # Define the Fourier space variables
# k_vals = fourier_basis.k_values()
# K = 0.5 * np.diag((1j * k_vals)**2)  # Hamiltonian for the kinetic energy


K = 0.5 * \
    np.diag(fourier_basis.project_derivative_of_function(
        f=psi_0, order=2, return_real_domain=True))

dx = L/N_x

main_diag = -2 * np.ones(N_x)
off_diag = np.ones(N_x - 1)

# Utilisation de scipy.sparse pour créer une matrice creuse tri-diagonale
K = -0.5/(dx**2)*sp.sparse.diags([off_diag, main_diag, off_diag],
                                 offsets=[-1, 0, 1], format='csc')

# Compute the time evolution operator (for one time step)
U = sp.sparse.linalg.expm(- 1j * K * T / N_t)

# Initialize the solution array
solution = np.zeros((N_t, N_x), dtype=complex)
solution[0, :] = psi_0(fourier_basis.mesh) / \
    np.linalg.norm(psi_0(fourier_basis.mesh))

# Time evolution (pre-compute the full time evolution for animation)
for i in range(1, N_t):
    # U = sp.linalg.expm(- 1j * K * lst_t[i])
    solution[i, :] = U @ solution[i - 1, :]
    # print(solution[i, :])

# Initial plot
line, = ax.plot(fourier_basis.mesh, np.abs(solution[0, :])**2, color='b')
# line, = ax.plot(fourier_basis.mesh, solution[0, :].real, color='b')
# Set axis labels, title and grid
ax.set_xlabel("Position")
ax.set_ylabel(r"$|\psi(x)|^2$")
ax.set_title("Wave Packet Evolution")
ax.grid()

# Set limits for x and y axes
# ax.set_xlim(-L/2, L/2)
# You might need to adjust this depending on your wave packet properties
# ax.set_ylim(0, 1.5)

# Initialization function


def init():
    line.set_ydata(np.abs(solution[0, :])**2)
    # line.set_ydata(solution[0, :].real)
    ax.set(title=f"t = {lst_t[0]}")
    return line,

# Update function for animation


def update(frame):
    line.set_ydata(np.abs(solution[frame, :])**2)
    # line.set_ydata(solution[frame, :].real)
    ax.set(title=f"t = {lst_t[frame]}")
    return line,


# Create the animation
ani = FuncAnimation(fig, update, frames=N_t,
                    init_func=init, blit=False, interval=50)

# Display the animation
# ani.show()
plt.show()
