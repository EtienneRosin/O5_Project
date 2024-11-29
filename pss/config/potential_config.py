"""Potential configuration classes for PSS simulations."""
from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
from pss.config.base_config import SimConfig

@dataclass
class PotentialConfig:
    """Base potential configuration."""
    x_0: Union[float, np.ndarray]  # Center position
    V_0: float                     # Potential strength
    dim: int                       # Spatial dimension

@dataclass
class WallConfig(PotentialConfig):
    """Wall potential configuration."""
    width: Union[float, Tuple[float, float]]  # Wall width(s)

@dataclass
class SlitConfig(PotentialConfig):
    """Slit potential configuration."""
    width: float   # Slit width
    depth: float   # Slit depth

@dataclass
class HarmonicConfig(PotentialConfig):
    """Harmonic oscillator configuration."""
    omega: float   # Angular frequency