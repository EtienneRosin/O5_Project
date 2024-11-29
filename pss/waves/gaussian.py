"""Gaussian wave packet implementation."""

import numpy as np
from .base import InitialCondition
from ..config.wave_config import GaussianConfig

class GaussianWavePacket(InitialCondition):
    def __init__(self, config: GaussianConfig):
        self.dim = config.dim
        self.x_0 = np.asarray(config.x_0)
        self.k_0 = np.asarray(config.k_0)
        self.sigma = config.sigma
        
        self.wave_number = np.linalg.norm(self.k_0)
        self.omega = self.wave_number**2/2
        self.phase_velocity = self.wave_number/2
        self.group_velocity = 2 * self.phase_velocity
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self._standardize_input(x)
        envelope = np.exp(-np.sum((x - self.x_0[:, None])**2, axis=0)/(4*self.sigma**2))
        plane_wave = np.exp(1j * np.dot(self.k_0, x))
        psi = envelope * plane_wave
        return psi / (np.linalg.norm(psi) if x.size > 1 else 1)