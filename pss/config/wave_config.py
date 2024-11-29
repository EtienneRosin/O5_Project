"""Wave configuration classes for PSS simulations."""

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
from pss.config.base_config import SimConfig

@dataclass
class WaveConfig:
    """Base wave packet configuration."""
    x_0: Union[float, np.ndarray]  # Initial position
    k_0: Union[float, np.ndarray]  # Wave vector
    dim: int                       # Spatial dimension

@dataclass
class GaussianConfig(WaveConfig):
    """Gaussian wave packet configuration."""
    sigma: float  # Width of the packet

@dataclass
class PlaneWaveConfig(WaveConfig):
    """Plane wave configuration."""
    pass