
from schrodinger_engine.domains.test_domain import Domain
from schrodinger_engine.fourier.fourier_test import FourierBasis
from schrodinger_engine.initial_conditions.gaussian_wave_packet_test import GaussianWavePacket
from schrodinger_engine.potentials import Potential, TimeDependentPotential
from schrodinger_engine.potentials import HarmonicOscillator, Wall
from schrodinger_engine.utils.graphics.complex_fields.multicolored_line_2D import MulticolorLine2d, HandlerMulticolorLine2d

import scipy as sp
import numpy as np
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Simulation:
    def __init__(self,
        spatial_domain: Domain,
        temporal_domain: Domain,
        initial_condition: GaussianWavePacket,
        potential: Potential = None,
        save_folder: str = "results") -> None:
        
        self.save_folder = save_folder
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.spatial_domain = spatial_domain
        if temporal_domain.dim != 1:
            raise ValueError("Temporal domain dimension should be 1.")
        self.temporal_domain = temporal_domain
        if initial_condition.dim != spatial_domain.dim:
            raise ValueError("Initial condition dimension should matcht the spatial domain dimension")
        self.initial_condition = initial_condition
        
        self.potential = potential
        if not potential is None:
            # self.type = 
            if potential.dim != spatial_domain.dim:
                raise ValueError("Potential dimension should matcht the spatial domain dimension")
            
            if isinstance(self.potential, TimeDependentPotential):
                self.type = "time_dependant_potential"
            else:
                self.type = "time_independant_potential"
            
        else:
            self.type = "free_particle"
        self.potential = potential
        self.fourier_basis = FourierBasis(domain=spatial_domain)
    
    def save_mesh(self) -> None:
        np.save(os.path.join(self.save_folder, f"spatial_mesh.npy"), self.spatial_domain.mesh)
    
    def save_state(self, psi, iteration, time):
        np.save(os.path.join(self.save_folder, f"wave_function_{iteration}.npy"), psi)
        np.save(os.path.join(self.save_folder, f"time_{iteration}.npy"), time)
     
    def get_kinetic_energy_propagator(self):
        k_vals = self.fourier_basis.get_frequencies()
    
        if self.spatial_domain.dim == 1:
            K_values = 0.5 * k_vals**2
        else:
            K_values = 0.5 * (k_vals[0].flatten()**2 + k_vals[1].flatten()**2)
        
        dt = self.temporal_domain.step[0]
        exp_K = np.exp(-1j * K_values * dt) 
        return exp_K
    
    def get_potential_energy_propagator(self, t):
        if isinstance(self.potential, TimeDependentPotential):
            V = self.potential(self.spatial_domain.get_mesh(), t)
        else:
            V = self.potential(self.spatial_domain.get_mesh())
        dt = self.temporal_domain.step[0]

        return np.exp(-1j * V * dt)
        
   
    def solve(self) -> None:
        mesh = self.spatial_domain.get_mesh()
        solution = np.zeros((self.temporal_domain.N[0], np.prod(self.spatial_domain.N)), dtype=complex)
        solution[0, :] = self.initial_condition(mesh)
        lst_t = self.temporal_domain.get_mesh()
        self.save_mesh()
        self.save_state(solution[0, :], 0, lst_t[0])

        kinetic_propagator = self.get_kinetic_energy_propagator()
        

        if self.potential:
            potential_propagator = (self.get_potential_energy_propagator(t=0) if not isinstance(self.potential, TimeDependentPotential)
                                    else None)

        for i in tqdm(range(1, self.temporal_domain.N[0])):
            psi = solution[i - 1, :]
            if self.potential:
                if isinstance(self.potential, TimeDependentPotential):
                    psi = self.get_potential_energy_propagator(lst_t[i]) * psi
                    # psi = self.get_potential_energy_propagator(lst_t[i])(psi)
                else:
                    psi = potential_propagator * psi
                    # psi = potential_propagator(psi)

            psi = self.fourier_basis.fft(psi)
            psi = kinetic_propagator * psi
            solution[i, :] = self.fourier_basis.ifft(psi)
            # self.save_state(solution[i, :], i)
            self.save_state(solution[i, :], i, lst_t[i])

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
        
        ani = FuncAnimation(fig, update, frames=self.temporal_domain.N[0],blit=False, interval=50)
        plt.show()

if __name__ == '__main__':
    L = 50
    N = 1000
    
    T = 100
    N_t = 5000
    
    dim = 2
    
    # wall_2d = 
    
    boundaries = [[-L/2, L/2] for _ in range(dim)]
    
    x_0 = np.zeros(dim)
    # print(f"{10 + x_0 = }")
    p_0 = np.ones(dim)
    w = 5*np.ones(dim)
    # print(f"{w = }")
    sigma = 1
    spatial_domain = Domain(boundaries = boundaries, N = N)
    temporal_domain = Domain(boundaries=[[0, T]], N = N_t)
    gauss_wave_packet = GaussianWavePacket(x0=x_0, p0=p_0, sigma=sigma, dim=dim)
    
    
    sim = Simulation(
        spatial_domain = spatial_domain, 
        temporal_domain = temporal_domain, 
        initial_condition = gauss_wave_packet,
        potential=Wall(x_0=10 + x_0, V_0=1e10, width=w, dim=dim)
        )
    sim.solve()
    # sim.display()