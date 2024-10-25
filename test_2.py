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
        
        # Utilisation de sparse.diags pour créer une matrice diagonale creuse pour K
        return sp.sparse.diags(0.5 * k_vals**2)
    
    def get_kinetic_energy_propagator(self):
        K = self.get_kinetic_energy_operator()
        dt = self.temporal_domain.step

        # Utilisation de l'opérateur exponentiel sparse directement en produit avec la fonction
        def kinetic_propagate(psi):
            return sp.sparse.linalg.expm_multiply(-1j * K * dt, psi)

        return kinetic_propagate

    def get_potential_energy_propagator(self, t):
        if self.potential is None:
            return sp.sparse.eye(self.spatial_domain.N)
        
        elif isinstance(self.potential, TimeDependentPotential):
            V = self.potential(self.spatial_domain.mesh, t)
        else:
            V = self.potential(self.spatial_domain.mesh)
        
        dt = self.temporal_domain.step

        # Retourne une fonction pour appliquer l'exponentielle en forme creuse
        return lambda psi: sp.sparse.linalg.expm_multiply(-1j * sp.sparse.diags(V) * dt, psi)

    def solve(self) -> None:
        solution = np.zeros((self.temporal_domain.N, self.spatial_domain.N), dtype=complex)
        solution[0, :] = self.initial_condition(self.spatial_domain.mesh)

        # Processeurs d'évolution d'énergie cinétique et potentielle
        kinetic_propagator = self.get_kinetic_energy_propagator()
        potential_propagator = (self.get_potential_energy_propagator(t=0) if not isinstance(self.potential, TimeDependentPotential) 
                                else None)
        
        normalization_factor = np.sqrt(self.spatial_domain.width) / self.spatial_domain.step
        lst_t = self.temporal_domain.mesh

        # Propagation de la solution
        for i in tqdm(range(1, self.temporal_domain.N), desc="Simulation Progress", unit="time step"):
            psi = solution[i - 1, :]
            
            if self.potential:
                if isinstance(self.potential, TimeDependentPotential):
                    psi = self.get_potential_energy_propagator(lst_t[i])(psi)
                else:
                    psi = potential_propagator(psi)
            
            psi = self.fourier_basis.project_function(psi)  # Passage dans l'espace des impulsions
            psi = kinetic_propagator(psi)                   # Propagation d'énergie cinétique
            solution[i, :] = sp.fft.ifft(normalization_factor * psi)

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
                
            cmap = "cmr.iceburn"
            # cmap = "cmr.guppy"
            cmap = "hsv"
            norm = plt.Normalize(-np.pi, np.pi)
            line = MulticolorLine2d(x = lst_x, y = mag, z = phase, cmap=cmap, norm=norm, label = r"$\left|\psi(x)\right|^2$")
            ax.add_collection(line)
            
            # real_line, = ax.plot(lst_x, values.real, label = r"$\Re\left(\psi(x)\right)$", c = "blue", alpha = 0.25)
            
            fig.colorbar(line, label= r"$arg\left(\psi(x)\right)$")
            ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
            ax.grid()
            
            def update(frame):
                values = self.solution[frame, :]
                mag = np.abs(values)**2
                phase = np.angle(values)
                
                # real_line.set_ydata(values.real)
                
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
    N_x = 1*500
    
    simu = Simulation(
        spatial_domain = SpatialDomain(boundaries = [-L/2, L/2], N = N_x),
        temporal_domain = TemporalDomain(t_end = T, N = N_t),
        initial_condition = GaussianWavePacket(x_0=0.0, sigma=1.0, lamb = 4),
        # potential = Wall(x_0 = 30, V_0 = 1e2, b = 5)
        # potential = Wall(x_0 = 30, V_0 = 1e1, b = 5)
        # potential=HarmonicOscillator(x_0=0, omega=1e0)
        )
    simu.solve()
    simu.display()
    pass