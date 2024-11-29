"""Core simulation implementation."""

import os
import h5py
import numpy as np
from tqdm import tqdm
from contextlib import closing
from typing import Optional

from pss.utils import Domain, FourierBasis
from pss.waves.base import InitialCondition
from pss.potentials.base import Potential, TimeDependentPotential
from pss.core.validators import validate_configuration

class PeriodicSchrodingerSimulation:
    """Main simulation class using split-step Fourier method."""
    
    def __init__(
        self,
        spatial_domain: Domain,
        temporal_domain: Domain,
        initial_condition: InitialCondition,
        potential: Optional[Potential] = None,
        save_folder: str = "results",
        name: str = "simulation",
        overwrite: bool = False
    ):
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain
        self.initial_condition = initial_condition
        self.potential = potential
        
        # Setup saving
        self._save_folder = save_folder
        self._save_path = f"{save_folder}/{name}.h5"
        os.makedirs(save_folder, exist_ok=True)
        
        if os.path.isfile(self._save_path) and not overwrite:
            raise FileExistsError(f"File {self._save_path} already exists")
            
        # Initialize components
        self.fourier_basis = FourierBasis(spatial_domain)
        validate_configuration(
            spatial_domain=spatial_domain,
            temporal_domain=temporal_domain,
            initial_condition=initial_condition,
            potential=potential
        )
            
    @property
    def simulation_type(self) -> str:
        if not self.potential:
            return 'free_particle'
        return 'time_dependent' if isinstance(self.potential, TimeDependentPotential) else 'stationary'
    
    def _setup_propagators(self):
        """Initialize kinetic and potential propagators."""
        k_vals = self.fourier_basis.frequencies
        dt = self.temporal_domain.step[0]
        
        if self.spatial_domain.dim == 1:
            K = 0.5 * k_vals**2
        else:
            K = 0.5 * (k_vals[0].flatten()**2 + k_vals[1].flatten()**2)
            
        self.kinetic_propagator = np.exp(-1j * K * dt)
        
        if self.potential and not isinstance(self.potential, TimeDependentPotential):
            self.potential_propagator = np.exp(-1j * self.potential(self.spatial_domain.mesh) * dt)
    
    def run(self):
        """Execute the simulation."""
        with closing(h5py.File(self._save_path, 'w')) as f:
            # Save metadata
            f.attrs['simulation_type'] = self.simulation_type
            f.attrs['dt'] = self.temporal_domain.step[0]
            
            # Save domains
            f.create_dataset('spatial_mesh', data=self.spatial_domain.mesh)
            f.create_dataset('temporal_mesh', data=self.temporal_domain.mesh)
            
            # Save potential if present
            if self.potential:
                if isinstance(self.potential, TimeDependentPotential):
                    self._save_time_dependent_potential(f)
                else:
                    f.create_dataset('potential', data=self.potential(self.spatial_domain.mesh))
            
            # Initialize wave function
            psi = self.initial_condition(self.spatial_domain.mesh)
            wave_function = f.create_dataset(
                'wave_function',
                shape=(self.temporal_domain.N[0], *psi.shape),
                dtype=psi.dtype
            )
            wave_function[0] = psi
            
            # Setup propagators
            self._setup_propagators()
            
            # Evolution loop
            for i in tqdm(range(1, self.temporal_domain.N[0])):
                psi = self._evolve_step(psi, i)
                wave_function[i] = psi
                
    def _evolve_step(self, psi: np.ndarray, step: int) -> np.ndarray:
        """Single time evolution step using split-step method."""
        if self.potential:
            if isinstance(self.potential, TimeDependentPotential):
                t = self.temporal_domain.mesh[step]
                V = self.potential(self.spatial_domain.mesh, t)
                psi *= np.exp(-1j * V * self.temporal_domain.step[0])
            else:
                psi *= self.potential_propagator
                
        psi = self.fourier_basis.fft(psi)
        psi = self.kinetic_propagator * psi
        return self.fourier_basis.ifft(psi)

