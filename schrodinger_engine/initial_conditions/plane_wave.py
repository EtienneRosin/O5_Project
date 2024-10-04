# from initial_conditions import InitialCondition
from schrodinger_engine.initial_conditions import InitialCondition 
import numpy as np
import matplotlib.pyplot as plt

class PlaneWave(InitialCondition):
    def __init__(self, lamb: float):
        self.lamb = lamb        # wavenumber
    
    def __call__(self, x):
        sol = np.exp(1j * 2 * np.pi * x / self.lamb)
        if isinstance(x, np.ndarray):
            sol /= np.linalg.norm(sol)
        return sol
        # return np.cos(self.k * x)
    
    def display(self, ax: plt.axes = None):
        """Affichage spécifique de l'onde plane, en utilisant la méthode générique"""
        # , label=f"Plane Wave (k={self.k})"
        super().display(ax)

if __name__ == "__main__":
    plane_wave = PlaneWave(lamb = 20)
    
    plane_wave.display()
    plt.show()