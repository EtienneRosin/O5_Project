import numpy as np
from pss.potentials.base import Potential
from pss.config.potential_config import HarmonicConfig

class HarmonicOscillator(Potential):
    def __init__(self, config: HarmonicConfig):
        self.x_0 = np.asarray(config.x_0)
        self.omega = config.omega
        self.dim = self._validate_dim(config.dim)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        X = (x.T - self.x_0).T
        return 0.5 * (self.omega ** 2) * np.sum(X**2, axis=0).flatten()
    
if __name__ == '__main__':
    pass