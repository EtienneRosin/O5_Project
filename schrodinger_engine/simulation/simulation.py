from schrodinger_engine.domains import SpatialDomain, TemporalDomain
from schrodinger_engine.fourier import FourierBasis
from schrodinger_engine.initial_conditions import InitialCondition, PlaneWave, GaussianWavePacket
from schrodinger_engine.potentials import Potential, TimeDependentPotential, Wall, HarmonicOscillator

from schrodinger_engine.visualization_tools import MulticolorLine2d, HandlerMulticolorLine2d

from schrodinger_engine.simulation import configuration_is_consistent

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm

class Simulation:
    """
    Represents a 1d Schrödinger equation.
    """
    def __init__(self,
                 spatial_domain: SpatialDomain,
                 temporal_domain: TemporalDomain,
                 initial_condition: InitialCondition,
                 potential: Potential = None
                 ) -> None:
        """
        Initialize the simulation.

        Parameters
        ----------
        spatial_domain : SpatialDomain
            Spatial domain of the simulation.
        temporal_domain : TemporalDomain
            Temporal domain of the simulation.
        initial_condition : InitialCondition
            Initial condition of the simulation (must be an InitialCondition instance).
        potential : Potential
            Potential of the simulation (must be an Potential instance, default : None).
        """
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.initial_condition = initial_condition
        self.potential = potential
        
        # check if the widhts and times are consistents to see something
        configuration_is_consistent(
            spatial_domain = spatial_domain, 
            temporal_domain = temporal_domain, 
            initial_condition = initial_condition, 
            potential = potential, 
            spatial_factor = 4, 
            temporal_factor = 4)

        # Check if the spatial domain is centered around 0
        if not spatial_domain.is_centered():
            raise ValueError("The spatial domain should be centered around zero (this is how is designed the simulation engine yet).")
        
        self.fourier_basis = FourierBasis(L = spatial_domain.width, N_x = spatial_domain.N)
        
        # self.kinetic_energy_propagator = self.get_kinetic_energy_propagator()
        self.solution = None
        pass
    
    def get_kinetic_energy_operator(self):
        k_vals = self.fourier_basis.k_values()
        
        return np.diag(0.5 * k_vals**2)
        # return np.diag(0.5 * k_vals**2)  # Hamiltonian for the kinetic energy
    
    def get_kinetic_energy_propagator(self):
        # K = sp.sparse.linalg.expm(- 1j * K * dt)
        K = self.get_kinetic_energy_operator()
        dt = self.temporal_domain.step
        return sp.sparse.linalg.expm(- 1j * K * dt)
    
    def get_potential_energy_propagator(self, t):
        if self.potential is None:
            return np.eye(self.spatial_domain.N)
        elif isinstance(self.potential, TimeDependentPotential):
            V = self.potential(self.spatial_domain.mesh, t)
        else :
            V = self.potential(self.spatial_domain.mesh)    # V is time-independant
        dt = self.temporal_domain.step
        return sp.sparse.linalg.expm(- 1j * np.diag(V) * dt)
    
    def solve(self) -> None:
        solution = np.zeros((self.temporal_domain.N, self.spatial_domain.N), dtype=complex)
        solution[0, :] = self.initial_condition(self.spatial_domain.mesh)  # initial state
        
        U = self.get_kinetic_energy_propagator()
        L = self.spatial_domain.width
        dx = self.spatial_domain.step
        lst_t = self.temporal_domain.mesh
        normalization_factor = np.sqrt(L) / dx
        # normalization_factor = np.sqrt(L) / self.spatial_domain.N
        
        if self.potential is None:
            for i in tqdm(range(1, self.temporal_domain.N), desc="Simulation Progress", unit="time step"):
                psi = solution[i - 1, :]
                psi = self.fourier_basis.project_function(psi)              # go to the momentum space
                psi = U @ psi                                               # propagate the kinetic energy
                solution[i, :] = sp.fft.ifft(normalization_factor * psi)    # go back to the position space
        
        elif isinstance(self.potential, TimeDependentPotential):
            for i in tqdm(range(1, self.temporal_domain.N), desc="Simulation Progress", unit="time step"):
                psi = solution[i - 1, :]
                psi = self.get_potential_energy_propagator(lst_t[i]) @ psi  # propagate potential in position space
                psi = self.fourier_basis.project_function(psi)              # go to the momentum space
                psi = U @ psi                                               # propagate the kinetic energy
                solution[i, :] = sp.fft.ifft(normalization_factor * psi)    # go back to the position space
        
        else : 
            potential_energy_progator = self.get_potential_energy_propagator(0)
            for i in tqdm(range(1, self.temporal_domain.N), desc="Simulation Progress", unit="time step"):
                psi = solution[i - 1, :]
                psi = potential_energy_progator @ psi                       # propagate potential in position space
                psi = self.fourier_basis.project_function(psi)              # go to the momentum space
                psi = U @ psi                                               # propagate the kinetic energy
                solution[i, :] = sp.fft.ifft(normalization_factor * psi)    # go back to the position space
        
        self.solution = solution
        
    
    def display(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        
        lst_x = self.spatial_domain.mesh
        lst_t = self.temporal_domain.mesh
        
        ylim = np.array([-1, 1])*np.max(np.abs(self.solution))/2
        ax.set(xlabel = r"$x$", title = f"t = {lst_t[0]:.3g}", xlim = (lst_x.min(), lst_x.max()), ylim = ylim)
        
        if not self.potential is None:
            if not isinstance(self.potential, TimeDependentPotential):
                # ax2 = ax.twinx()
                V = self.potential(lst_x)
                # V /= np.linalg.norm(V)
                ax.plot(lst_x, self.potential(lst_x), label = r"$V/\|V\|$")
        
        values = self.solution[0, :]
        mag = np.abs(values)**2
        phase = np.angle(values)
            
        cmap = "cmr.lavender"
        norm = plt.Normalize(-np.pi, np.pi)
        line = MulticolorLine2d(x = lst_x, y = mag, z = phase, cmap=cmap, norm=norm, label = r"$\left|\psi(x)\right|^2$")
        ax.add_collection(line)
        
        real_line, = ax.plot(lst_x, values.real, label = r"$\Re\left(\psi(x)\right)$", c = "blue", alpha = 0.5)
        
        fig.colorbar(line, label= r"$arg\left(\psi(x)\right)$")
        ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
        ax.grid()
        
        def update(frame):
            values = self.solution[frame, :]
            mag = np.abs(values)**2
            phase = np.angle(values)
            
            real_line.set_ydata(values.real)
            
            segments = line.create_segments(lst_x, mag)
            line.set_segments(segments)
            line.set_array(phase)
            ax.set(title=f"t = {lst_t[frame]:.3g}")

            return line,
        
        ani = FuncAnimation(fig, update, frames=self.temporal_domain.N,blit=False, interval=50)
        plt.show()
    

if __name__ == "__main__":
    T = 100
    N_t = 2000
    L = 200
    N_x = 2*500
    
    simu = Simulation(
        spatial_domain = SpatialDomain(boundaries = [-L/2, L/2], N = N_x),
        temporal_domain = TemporalDomain(t_end = T, N = N_t),
        initial_condition = GaussianWavePacket(x_0=0.0, sigma=2.0, lamb = 4),
        potential = Wall(x_0 = 30, V_0 = 1e2, b = 5)
        # potential = Wall(x_0 = 30, V_0 = 1e1, b = 5)
        # potential=HarmonicOscillator(x_0=0, omega=1e0)
        )
    simu.solve()
    simu.display()
    pass