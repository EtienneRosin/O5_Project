import h5py
import os
import numpy as np
from tqdm import tqdm
from schrodinger_engine.utils.domain import Domain
from schrodinger_engine.utils.fourier_basis import FourierBasis
from schrodinger_engine.initial_conditions.gaussian_wave_packet import InitialCondition, GaussianWavePacket
from schrodinger_engine.potentials import Potential, TimeDependentPotential, Wall
from schrodinger_engine.simulation import configuration_is_consistent




class PeriodicSchrodingerSimulation:
    def __init__(
        self,
        spatial_domain: Domain,
        temporal_domain: Domain,
        initial_condition: InitialCondition,
        potential: Potential = None,
        save_folder: str ="results",
        name: str = 'simulation',
        spatial_factor: int = 4,
        temporal_factor: int = 4,
        overwrite: bool = False
        ) -> None:
        
        # Simulation properties
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.initial_condition = initial_condition
        self.potential = potential
        self.simulation_type = self._get_simulation_type()
        
        # Fourier basis initialization
        self.fourier_basis = FourierBasis(domain=spatial_domain)
        
        # File properties
        self._overwrite = overwrite
        self._save_folder = save_folder
        self.name = name
        self._save_path = f"{save_folder}/{name}.h5"
        self.save_file = self._setup_save_file()
        
        self._save_domains()
        self._save_potential()
        
    def _get_simulation_type(self) -> str:
        if self.potential:
            if isinstance(self.potential, TimeDependentPotential):
                return 'time_dependant'
            else:
                return 'stationnary'
        else:
            return 'free_particle'
    
    def _setup_save_file(self):
        """Setup the save file."""
        if not os.path.exists(self._save_folder):
            os.makedirs(self._save_folder)
        if os.path.isfile(self._save_path) and not self._overwrite:
            raise FileExistsError(f"File {self._save_path} already exists. Set `overwrite=True` to overwrite.")
        return h5py.File(name=self._save_path, mode='w')
    
    def _save_domains(self):
        """Sauvegarde du maillage spatial et temporel."""
        self.save_file.create_dataset("spatial_mesh", data=self.spatial_domain.get_mesh())
        self.save_file.create_dataset("temporal_mesh", data=self.temporal_domain.get_mesh())
    
    def _save_potential(self):
        """Sauvegarde le potentiel pour chaque instant s'il dépend du temps, sinon une seule fois."""
        mesh = self.spatial_domain.get_mesh()
        if isinstance(self.potential, TimeDependentPotential):
            V_dataset = self.save_file.create_dataset("potential", shape=(self.temporal_domain.N[0], *mesh.shape))
            for i, t in enumerate(self.temporal_domain.get_mesh()):
                V_dataset[i, :] = self.potential(mesh, t)
        else:
            V = self.potential(mesh)
            self.save_file.create_dataset("potential", data=V)
            
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
    
    def run(self):
        mesh = self.spatial_domain.get_mesh()
        psi = self.initial_condition(mesh)
        lst_t = self.temporal_domain.get_mesh()
        
        wave_function_data_set = self.save_file.create_dataset(
            "wave_function",
            shape=(self.temporal_domain.N[0], *psi.shape),
            dtype=psi.dtype)
        
        wave_function_data_set[0, :] = psi  # Initial condition
        
        kinetic_propagator = self.get_kinetic_energy_propagator()
        if self.potential and not isinstance(self.potential, TimeDependentPotential):
            potential_propagator = self.get_potential_energy_propagator(t=0)
        else:
            potential_propagator = None
            
        for i in tqdm(range(1, self.temporal_domain.N[0])):
            if self.potential:
                if isinstance(self.potential, TimeDependentPotential):
                    psi *= self.get_potential_energy_propagator(lst_t[i])
                else:
                    psi *= potential_propagator

            psi = self.fourier_basis.fft(psi)
            psi = kinetic_propagator * psi
            psi = self.fourier_basis.ifft(psi)
                
            wave_function_data_set[i, :] = psi
                
        print(f"Simulation data saved to {self._save_path}.")
    
    def close(self):
        """Close the save file."""
        self.save_file.close()
    
    def __del__(self):
        """Ensure the file is closed when ending the script."""
        self.close()
        
if __name__ == '__main__':
    # fname = "results/simulation_data.h5"
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
    
    # print("chubzoucboub")
    sim = PeriodicSchrodingerSimulation(
        spatial_domain=spatial_domain, 
        temporal_domain=temporal_domain, 
        initial_condition=gauss_wave_packet,
        potential=Wall(x_0=30 + x_0, V_0=1e5, width=w, dim=dim),
        overwrite=True
    )
    sim.run()
    