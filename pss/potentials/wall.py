import numpy as np
from .base import Potential
from ..config.potential_config import WallConfig

class Wall(Potential):
    def __init__(self, config: WallConfig):
        self.x_0 = np.asarray(config.x_0)
        self.V_0 = config.V_0
        self.dim = self._validate_dim(config.dim)
        self.width = np.array(config.width) if isinstance(config.width, (tuple, list, np.ndarray)) else np.array([config.width] * self.dim)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        match self.dim:
            case 1:
                inside_wall = np.abs(x - self.x_0) <= self.width/2
            case 2:
                X = (x.T - self.x_0).T
                    
                inside_wall = np.logical_and(
                    np.abs(X[0]) <= self.width[0]/2,
                    np.abs(X[1]) <= self.width[1]/2
                )
        return np.where(inside_wall, self.V_0, 0.0).flatten()