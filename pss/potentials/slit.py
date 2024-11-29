import numpy as np
from .base import Potential
from ..config.potential_config import SlitConfig

class Slit(Potential):
    def __init__(self, config: SlitConfig):
        self.x_0 = np.asarray(config.x_0)
        self.V_0 = config.V_0
        self.width = config.width
        self.depth = config.depth
        self.dim = 2
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        X = (x.T - self.x_0).T
        inside_wall = np.logical_and(
            np.abs(X[0]) <= self.depth/2,
            np.abs(X[1]) >= self.width/2
        )
        return np.where(inside_wall, self.V_0, 0.0)