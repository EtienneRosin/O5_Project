# from initial_conditions import InitialCondition
from schrodinger_engine.initial_conditions import InitialCondition, PlaneWave
import numpy as np
import matplotlib.pyplot as plt

class GaussianWavePacket(InitialCondition):
    def __init__(self, x_0: float, sigma: float, lamb: float):
        self.x_0 = x_0      # center
        self.sigma = sigma  # width
        self.lamb = lamb    # wavelength
        self.plane_wave = PlaneWave(lamb = lamb)
    
    def __call__(self, x):
        """Évaluation du paquet d'onde gaussien en un point x"""
        envelope = np.exp(- (x - self.x_0)**2 / (2 * self.sigma**2))
        # oscillation = np.cos(self.k * (x - self.x0))
        sol = envelope * self.plane_wave(x)
        if isinstance(x, np.ndarray):
            sol /= np.linalg.norm(sol)
        return sol
    
    def display(self, ax: plt.axes = None):
        """Affichage spécifique du paquet d'onde gaussien, en utilisant la méthode générique"""
        # , label=f"Gaussian Wave Packet (x0={self.x0}, sigma={self.sigma}, k={self.k})"
        super().display(ax = ax)


# Exemple d'utilisation

if __name__ == "__main__":
    gaussian_packet = GaussianWavePacket(x_0=0.0, sigma = 2.0, lamb = 1.9)

    # Évaluation en un point x
    x_value = 1.0
    # print("Plane wave evaluated at x=1:", plane_wave(x_value))
    # print("Gaussian wave packet evaluated at x=1:", gaussian_packet(x_value))

    # Affichage des deux ondes
    # fig, ax = plt.subplots()

    gaussian_packet.display()
    plt.show()