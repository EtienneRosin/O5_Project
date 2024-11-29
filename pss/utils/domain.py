"""Domain and mesh handling for PSS."""

import numpy as np

class Domain:
    """Represents physical domain for quantum simulations."""
    def __init__(self, boundaries: list, step: float = None, N: int = None):
        self.dim = len(boundaries)
        self.boundaries = np.asarray(boundaries)
        self.length = np.array([(bound[1] - bound[0]) for bound in boundaries])
        
        if step and N:
            raise ValueError("Specify either `step` or `N`, not both")
        
        if N is not None:
            self.N = np.array(N) if isinstance(N, (list, tuple)) else np.array([N] * self.dim)
            self.step = [(bound[1] - bound[0]) / self.N[i] for i, bound in enumerate(boundaries)]
        elif step is not None:
            self.step = np.array(step) if isinstance(step, (list, tuple)) else np.array([step] * self.dim)
            self.N = np.array([np.ceil(L/self.step[i]).astype(int) for i, L in enumerate(self.length)])
        else:
            raise ValueError("Must specify either `step` or `N`")
        
        self.mesh = self.generate_mesh()
    
    def generate_mesh(self) -> np.ndarray:
        """Generate mesh grid for the domain."""
        if self.dim == 1:
            return np.linspace(self.boundaries[0][0], self.boundaries[0][1], self.N[0], endpoint=False)
        else:
            x = np.linspace(self.boundaries[0][0], self.boundaries[0][1], self.N[0], endpoint=False)
            y = np.linspace(self.boundaries[1][0], self.boundaries[1][1], self.N[1], endpoint=False)
            return np.array(np.meshgrid(x, y, indexing='ij'))

    def get_mesh(self) -> np.ndarray:
        return self.mesh