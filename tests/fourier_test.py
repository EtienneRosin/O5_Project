from schrodinger_engine.fourier import FourierBasis

from typing import Union

import numpy as np


L = 10
N_x = 100
fourier_basis = FourierBasis(L=L, N_x=N_x)

def f(x: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
    return np.cos(2 * np.pi * x / L)

def f_prime(x: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
    return -(2 * np.pi / L)*np.sin(2 * np.pi * x / L)

def f_double_prime(x: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
    return -(2 * np.pi / L)**2 * np.cos(2 * np.pi * x / L)

def g(x: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
    return np.exp(np.cos(2 * np.pi * x / L))

def h(x: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
    return (np.abs(x) <= L/4)

# print(fourier_basis.k_values())
# fourier_basis.display_e_n(n_vals=[1, 2, 6])
fourier_basis.display_decomposition(f)
fourier_basis.compare_decomposition(f)

# fourier_basis.display_derivative_decomposition(f, order=10)
fourier_basis.compare_derivative_decomposition(f, f_prime, order=1)
fourier_basis.compare_derivative_decomposition(f, f_double_prime, order=2)

# x_vals = np.linspace(0, L, 100)  # 100 points sur l'intervalle [0, L]
# n_vals = np.arange(-5, 6)  # n allant de -5 à 5

# # Obtenir les fonctions de Fourier pour ces x et n
# # functions = fourier_basis.evaluate_basis(x_vals, n_vals)
