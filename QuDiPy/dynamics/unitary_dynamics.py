""" Calculates the Unitary Dynamics of a System
Written by: Amro Dodin (Willard Group - MIT)

"""

import QuDiPy.util.linalg as la
from QuDiPy.util.constants import hbar
import numpy as np


def unitary_derivative(rho, ham):
    return -1j/hbar * la.commutator(ham, rho)


array_unitary_derivative = np.vectorize(unitary_derivative, excluded={'ham'})


def grid_unitary_derivative(rho_grid, ham):
    data = array_unitary_derivative(rho_grid.data, ham=ham)
    return rho_grid.like(data)
