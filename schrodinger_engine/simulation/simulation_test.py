import h5py
import os
import numpy as np
from tqdm import tqdm
from schrodinger_engine.utils.domain import Domain
from schrodinger_engine.utils.fourier_basis import FourierBasis
from schrodinger_engine.initial_conditions.gaussian_wave_packet import InitialCondition, GaussianWavePacket
from schrodinger_engine.potentials import Potential, TimeDependentPotential, Wall
from schrodinger_engine.simulation import configuration_is_consistent

class Simulation:
    def __init__(self,
                 spatial_domain: Domain,
                 temporal_domain: Domain,
                 initial_condition: InitialCondition,
                 potential: Potential = None,
                 save_folder: str ="results",
                 save_name: str = 'simulation',
                 spatial_factor: int = 4,
                 temporal_factor: int = 4):
        self.save_folder = save_folder
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.initial_condition = initial_condition
        self.potential = potential
        self.fourier_basis = FourierBasis(domain=spatial_domain)
        configuration_is_consistent(
            spatial_domain=spatial_domain, 
            temporal_domain=temporal_domain, 
            initial_condition=initial_condition, 
            potential=potential, 
            spatial_factor=spatial_factor,
            temporal_factor= temporal_factor)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        elif os.listdir(save_folder):
            choice = input(f"Folder {save_folder} is not empty. Overwrite? (y/n): ")
            if choice.lower() != 'y':
                print("Simulation aborted.")
                exit()
            else:
                for filename in os.listdir(save_folder):
                    file_path = os.path.join(save_folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

    def save_mesh(self, hdf5_file):
        """Sauvegarde du maillage spatial et temporel."""
        hdf5_file.create_dataset("spatial_mesh", data=self.spatial_domain.mesh)
        hdf5_file.create_dataset("temporal_mesh", data=self.temporal_domain.get_mesh())
    
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

    def solve(self):
        mesh = self.spatial_domain.get_mesh()
        psi = self.initial_condition(mesh)
        lst_t = self.temporal_domain.get_mesh()
        
        hdf5_path = os.path.join(self.save_folder, "simulation_data.h5")
        with h5py.File(hdf5_path, "w") as hdf5_file:
            self.save_mesh(hdf5_file)
            psi_shape = psi.shape
            dset = hdf5_file.create_dataset("wave_function", 
                                            shape=(1, *psi_shape), 
                                            maxshape=(None, *psi_shape), 
                                            dtype=psi.dtype)
            
            kinetic_propagator = self.get_kinetic_energy_propagator()
            if self.potential:
                potential_propagator = (self.get_potential_energy_propagator(t=0) if not isinstance(self.potential, TimeDependentPotential)
                                        else None)
            
            for i in tqdm(range(1, self.temporal_domain.N[0])):
                if self.potential:
                    if isinstance(self.potential, TimeDependentPotential):
                        psi *= self.get_potential_energy_propagator(lst_t[i])
                    else:
                        psi *= potential_propagator

                psi = self.fourier_basis.fft(psi)
                psi = kinetic_propagator * psi
                psi = self.fourier_basis.ifft(psi)
                
                
                dset.resize((i + 1, *psi_shape))
                dset[i, :] = psi

        print(f"Simulation data saved to {hdf5_path}")
    def __del__(self):
        print('Destructor called, Employee deleted.')
if __name__ == '__main__':
    L = 100
    dx = 0.1
    
    T = 40
    dt = 0.05
    dim = 1
    boundaries = [[-L/2, L/2] for _ in range(dim)]
    
    x_0 = np.zeros(dim)
    wave_number = 4
    k_0 = np.ones(dim)
    k_0 = wave_number * k_0 / np.linalg.norm(k_0)
    w = 5 * np.ones(dim)
    sigma = 2
    spatial_domain = Domain(boundaries=boundaries, step=dx)
    temporal_domain = Domain(boundaries=[[0, T]], step=dt)
    gauss_wave_packet = GaussianWavePacket(x_0=x_0, k_0=k_0, sigma=sigma, dim=dim)
    
    sim = Simulation(
        spatial_domain=spatial_domain, 
        temporal_domain=temporal_domain, 
        initial_condition=gauss_wave_packet,
        potential=Wall(x_0=30 + x_0, V_0=1e5, width=w, dim=dim)
    )
    sim.solve()