""" Linear Algebra Utilities
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
from numpy import pi
from QuDiPy.util.spherical import spherical_to_cartesian


class Observable:
    def dot(self, other):
        assert len(self.vector) == len(other.vector)
        self_mat = self.calculate_matrix()
        assert np.shape(self_mat)[0] == np.shape(self_mat)[1]
        op_mat = other.calculate_matrix()
        assert np.shape(op_mat)[0] == np.shape(op_mat)[1]
        assert np.shape(self_mat) == np.shape(op_mat)
        return np.trace(np.conj(self_mat.T)@op_mat)/(np.shape(self_mat)[0])

    def __eq__(self, other):
        return np.array_equal(self.calculate_matrix(), other.calculate_matrix())

    def __mul__(self, other):
        return self.dot(other)

    def __abs__(self):
        norm = np.sqrt(self * self)
        assert norm >= 0.
        return norm

    def __init__(self, vector):
        self.vector = vector


def calculate_spin_matrix(cartesian_vector):
    i, x, y, z = cartesian_vector
    return np.array([[i + z, x - 1j * y], [x + 1j * y, i - z]])


class SpinObservable(Observable):
    def calculate_matrix(self):
        cart = self.convert_cartesian()
        return calculate_spin_matrix(cart)

    def __init__(self, vector):
        assert len(vector) == 4
        super().__init__(vector)


class CartesianSpinObservable(SpinObservable):
    def convert_cartesian(self):
        return self.vector

    def __init__(self, vector):
        assert len(vector) == 4
        # Check Inputs are Real
        i, x, y, z = vector
        assert i.imag == 0.0
        assert x.imag == 0.0
        assert y.imag == 0.0
        assert z.imag == 0.0

        super().__init__(vector)


class SphericalSpinObservable(SpinObservable):
    def convert_cartesian(self):
        return spherical_to_cartesian(self.vector)

    def __init__(self, vector):
        assert len(vector) == 4
        # Read and Check Spherical Coordinate Inputs
        i, r, theta, phi = vector
        assert i.imag == 0.0
        assert r.imag == 0.0 and 0. <= r
        assert theta.imag == 0.0 and 0 <= theta <= pi
        assert phi.imag == 0.0 and 0 <= phi <= 2 * pi

        # Call SpinObservable Constructor with Spherical Conversion
        super().__init__(vector)
