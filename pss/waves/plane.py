"""Plane wave implementation."""
import numpy as np
from .base import InitialCondition
from ..config.wave_config import GaussianConfig

class PlaneWave(InitialCondition):
    def __init__(self, config: PlaneWaveConfig):
        self.dim = config.dim
        self.k_0 = np.asarray(config.k_0)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self._standardize_input(x)
        return np.exp(1j * np.dot(self.k_0, x))