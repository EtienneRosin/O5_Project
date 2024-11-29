"""
Base configuration classes for PSS simulations.
Provides dataclasses for storing and validating simulation parameters.
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np

@dataclass
class SimConfig:
    """Base simulation configuration parameters."""
    
    # Domain parameters
    L: float                # Spatial domain length
    T: float                # Total simulation time
    dx: float              # Spatial step
    dt: float              # Time step
    dim: int               # Dimension (1 or 2)
    
    # Physical parameters
    x_0: np.ndarray        # Initial position
    k_0: np.ndarray        # Initial wave vector
    sigma: float           # Wave packet width
    
    # Numerical parameters
    spatial_factor: int = 4    # Spatial resolution factor
    temporal_factor: int = 4   # Temporal resolution factor
    
    # Output parameters
    save_folder: str = "results"
    name: str = "simulation"
    overwrite: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dim not in [1, 2]:
            raise ValueError("Dimension must be 1 or 2")
        if not isinstance(self.x_0, np.ndarray):
            self.x_0 = np.array(self.x_0)
        if not isinstance(self.k_0, np.ndarray):
            self.k_0 = np.array(self.k_0)
        if self.x_0.shape != (self.dim,) or self.k_0.shape != (self.dim,):
            raise ValueError(f"x_0 and k_0 must have shape ({self.dim},)")