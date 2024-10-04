"""
@file kronig_penney.py
@brief Implementation a KronigPenney potential class
@author Etienne Rosin 
@version 0.1
@date 28/09/2024
"""
from schrodinger_engine.potentials import Potential, Wall

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class KronigPenney(Potential):
    """
    @class KronigPenney
    @brief Represent a Kronig-Penney potential
    """
    def __init__(self, x_0 : float, V_0: float, b : float, a: float, N: float) -> None :
        """
        @brief Initialize a Kronig-Penney potential
        @param x_0: center of the wall
        @param V_0: height of the wall
        @param b: width of the wall
        @param a: distance between 2 well
        @param N: number of well
        """
        self.ghost_wall = Wall(x_0 = x_0, V_0 = V_0, b = b)
        if a <= 0:
            raise ValueError(f"a should be strictly positive (here a = {a})")
        
        if N <= 0:
            raise ValueError(f"N should be strictly positive (here N = {N})")
        
        self.x_0 = x_0
        self.V_0 = V_0
        self.b = b
        self.a = a
        self.N = N
        
        
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        
        value = np.zeros_like(x)
        for n in range(-self.N//2, self.N//2):
            self.ghost_wall.x_0 = self.x_0 + (n + 0.5 + 0.5*(self.N%2)) * (self.a + self.b)
            value = np.logical_or(value, self.ghost_wall(x)).astype(float)
        return self.V_0 * value.astype(float)
    


if __name__ == "__main__":
    N = 3
    kp = KronigPenney(x_0 = 0, V_0 = 10, b = 0.5, a = 10, N = N)
    kp.display()
    plt.show()