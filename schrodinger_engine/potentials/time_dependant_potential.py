

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Union
from schrodinger_engine.utils.domain import Domain
from schrodinger_engine.potentials.potential import Potential

class TimeDependentPotential(Potential):
    """
    Base class for time-dependent potentials.
    """
    @abstractmethod
    def __call__(self, x: float | np.ndarray, t: float) -> float | np.ndarray :
        """
        Evaluate the potential on point(s) x at time t.
        
        Parameters
        ----------
        x : array-like
            Point(s) onto which evaluate the potential.
        t : float
            Time at which to evaluate the potential.
        
        Returns
        -------
        array-like
            V(x, t): potential at x and time t.
        """
        pass
    