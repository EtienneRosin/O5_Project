# Exemple d'intégration de FourierBasis avec SpatialDomain

from schrodinger_engine.fourier import FourierBasis
from schrodinger_engine.domains import SpatialDomain

# Importation des classes nécessaires
import numpy as np
import matplotlib.pyplot as plt

# Instancier le domaine spatial
spatial_domain = SpatialDomain(boundaries=[-5, 5], N=100)

# Extraire la longueur et le nombre de points de discrétisation
L = spatial_domain.boundaries[1] - spatial_domain.boundaries[0]  # Longueur du domaine
N_x = spatial_domain.N  # Nombre de points de discrétisation

# Instancier la base de Fourier en utilisant la longueur et le nombre de points
fourier_basis = FourierBasis(L=L, N_x=N_x)

# Exemple d'utilisation de la base de Fourier pour projeter une fonction
def example_function(x):
    return np.sin(np.pi * x / L)

# Projeter la fonction sur la base de Fourier
coefficients = fourier_basis.project_function(example_function)

# Afficher la décomposition de Fourier de la fonction
fourier_basis.display_decomposition(example_function)

# Affichage d'une fonction particulière de la base (par exemple la première et la deuxième)
fourier_basis.display_e_n([1, 2])

plt.show()