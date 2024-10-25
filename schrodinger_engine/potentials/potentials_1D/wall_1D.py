"""
@file wall.py
@brief Implementation a Wall potential class
@author Etienne Rosin 
@version 0.1
@date 28/09/2024
"""

from schrodinger_engine.potentials.potentials_1D import Potential1D

import numpy as np
import matplotlib.pyplot as plt
from typing import Union

class Wall1D(Potential1D):
    """
    @class Wall
    @brief Represent a wall potential
    """
    def __init__(self, x_0 : float, V_0: float, b : float) -> None:
        """
        @brief Initialize a wall potential
        @param x_0: center of the wall
        @param V_0: height of the wall
        @param b: width of the wall
        """
        if b <= 0:
            raise ValueError(f"b should be strictly positive (here b = {b})")
        self.x_0 = x_0
        self.V_0 = V_0
        self.b = b
        
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r = np.abs(x - self.x_0)
        return self.V_0 * (r <= self.b).astype(float)
    

if __name__ == "__main__":
    barriere = Wall1D(x_0 = 0, V_0 = 10, b = 0.5)
    barriere.display()
    plt.show()