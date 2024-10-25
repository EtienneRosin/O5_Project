import numpy as np

class Domain:
    def __init__(self, boundaries, step=None, N=None):
        """
        Initialize a domain in 1D or 2D.
        
        Parameters:
        - boundaries : list of tuples or tuple of length 2 (for each dimension).
        - step : float or tuple of floats specifying the step size in each dimension (optional).
        - N : int or tuple of ints specifying the number of points in each dimension (optional).
        """
        self.dim = len(boundaries)
        self.boundaries = np.asarray(boundaries)
        self.length =  [(bound[1] - bound[0]) for bound in boundaries]
        if step and N:
            raise ValueError("Specify either `step` or `N`, not both.")
        
        if N is not None:
            self.N = np.array(N) if isinstance(N, (list, tuple)) else np.array([N] * self.dim)
            self.step = [(bound[1] - bound[0]) / self.N[i] for i, bound in enumerate(boundaries)]
        elif step is not None:
            self.step = np.array(step) if isinstance(step, (list, tuple)) else np.array([step] * self.dim)
            self.N = np.array([np.ceil(L/ self.step[i]).astype(int) for i, L in enumerate(self.length)])
        else:
            raise ValueError("You must specify either `step` or `N`.")

        # Génère la grille de points
        self.mesh = self.generate_mesh()
    
    def generate_mesh(self):
        """
        Generate the mesh grid based on the boundaries, step size, or number of points.
        """
        if self.dim == 1:
            x = np.linspace(self.boundaries[0][0], self.boundaries[0][1], self.N[0], endpoint=False)
            # self.unique_points = x
            return x
        elif self.dim == 2:
            x = np.linspace(self.boundaries[0][0], self.boundaries[0][1], self.N[0], endpoint=False)
            y = np.linspace(self.boundaries[1][0], self.boundaries[1][1], self.N[1], endpoint=False)
            # self.unique_points = [x,y]
            X, Y = np.meshgrid(x, y, indexing='ij')
            return np.array([X, Y])
    
    def is_centered(self):
        """
        Check if the domain is centered around zero.
        """
        return all((bound[0] == -bound[1]) for bound in self.boundaries)
    
    def get_step_size(self):
        """
        Return the step size for each dimension.
        """
        return self.step
    
    def get_mesh(self):
        """
        Return the mesh grid of the domain.
        """
        return self.mesh
    
    def get_boundaries(self):
        """
        Return the boundaries of the domain.
        """
        return self.boundaries
    
if __name__ == '__main__':
    # Domaine en 1D avec pas spécifié
    domain_1d = Domain(boundaries=[(-10, 10)], step=0.1)
    print("1D Domain mesh:", domain_1d.get_mesh())

    # # Domaine en 1D avec nombre de points spécifié
    # domain_1d_N = Domain(boundaries=[(-10, 10)], N=200)
    # print("1D Domain mesh (with N):", domain_1d_N.get_mesh())

    # # Domaine en 2D avec pas spécifié
    # domain_2d = Domain(boundaries=[(-10, 10), (-5, 5)], step=[0.1, 0.2])
    # print("2D Domain mesh:", domain_2d.get_mesh())

    # # Domaine en 2D avec nombre de points spécifié
    domain_2d_N = Domain(boundaries=[(-10, 10), (-5, 5)], N=[200, 100])
    print(f"{domain_2d_N.step}")
    # print("2D Domain mesh (with N):", domain_2d_N.get_mesh())