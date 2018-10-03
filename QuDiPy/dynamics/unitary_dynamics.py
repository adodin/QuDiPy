""" Calculates the Unitary Dynamics of a System
Written by: Amro Dodin (Willard Group - MIT)

"""

import QuDiPy.util.linalg as la
from QuDiPy.util.constants import hbar


def unitary_derivative(rho, ham):
    return -1j/hbar * la.commutator(ham, rho)
