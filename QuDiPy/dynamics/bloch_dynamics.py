""" Bloch Dynamics for 2 Level Systems
Written by: Amro Dodin (Willard Group - MIT)

This is a generalization of the Bloch type equations including both the NMR and Optical Bloch Equations
"""

from QuDiPy.dynamics.unitary_dynamics import unitary_derivative
import QuDiPy.util.linalg as la


def bloch_derivative(rho, ham, gdiss, gdeph, rpump=0.):
    ud = unitary_derivative(rho, ham)
    i, x, y, z = rho.convert_cartesian()
    xdot = -(gdeph + 0.5 * gdiss + 0.5 * rpump) * x
    ydot = -(gdeph + 0.5 * gdiss + 0.5 * rpump) * y
    zdot = rpump * (1 - z) - gdiss * (1 + z)
    bd = la.CartesianSpinOperator((0, xdot, ydot, zdot))
    return ud + bd