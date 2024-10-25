"""
@file harmonic_oscillator.py
@brief Implementation a HarmonicOscillator potential class
@author Etienne Rosin 
@version 0.1
@date 28/09/2024
"""
from schrodinger_engine.potentials.potentials_2D import Potential2D

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class HarmonicOscillator(Potential2D):
    """
    @class HarmonicOscillator
    @brief Represent an harmonic oscillator
    """
    def __init__(self, x_0 : float, omega : float) -> None:
        """
        @brief Initialize an harmonic oscillator potential
        @param x_0: center of the harmonic oscillator
        @param omega: angular frequency
        """
        self.x_0 = x_0
        self.omega = omega
        
    def __call__(self, x: float|np.ndarray) -> float | np.ndarray:
        # (1/2)*(m*omega**2)*(x - x_0)**2)
        r = np.linalg
        return 0.5 * (self.omega**2) * (x - self.x_0)**2
    


# class TimeDependentHarmonicOscillator(Potential):
#     def __init__(self, k: float, omega: float):
#         self.k = k
#         self.omega = omega

#     def __call__(self, x: float | np.ndarray, t: float = 0.0) -> float | np.ndarray:
#         return 0.5 * self.k * x**2 * np.cos(self.omega * t)
    
    
if __name__ == "__main__":
    omega = 1/10
    ho = HarmonicOscillator(x_0 = 0, omega = omega)
    
    # ho.__call__()
    ho.display()
    plt.show()