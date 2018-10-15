""" Linear Algebra Utilities
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
from QuDiPy.coordinates.spherical import spherical_to_cartesian
import QuDiPy.coordinates.spherical as sp


def calculate_spin_matrix(cartesian_vector):
    i, x, y, z = cartesian_vector
    return np.array([[i + z, x - 1j * y], [x + 1j * y, i - z]])


def calculate_cartesian_spin_vector_from_matrix(matrix):
    i = 0.5 * (matrix[0, 0] + matrix[1, 1])
    x = 0.5 * (matrix[0, 1] + matrix[1, 0])
    y = 0.5j * (matrix[0, 1] - matrix[1, 0])
    z = 0.5 * (matrix[0, 0] - matrix[1, 1])
    return i, x, y, z


def commutator(a, b):
    return a @ b - b @ a


class Operator:

    def __str__(self):
        vec_str = 'Vector: ' + str(self.vector) + '\n'
        mat_str = 'Matrix: ' + str(self.matrix)
        return vec_str + mat_str

    def __repr__(self):
        class_str = str(type(self)) + '\n'
        id_str = 'ID: ' + str(id(self)) + '\n'
        op_str = str(self)
        return class_str + id_str + op_str

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def __mul__(self, other):
        return self.dot(other)

    def __rmul__(self, other):
        prod_mat = other * self.matrix
        prod_vec = self.matrix_to_vector(prod_mat)
        return self.__class__(prod_vec)

    def __add__(self, other):
        sum_mat = self.matrix + other.matrix
        sum_vec = self.matrix_to_vector(sum_mat)
        return self.__class__(sum_vec)

    def __sub__(self, other):
        diff_mat = self.matrix - other.matrix
        diff_vec = self.matrix_to_vector(diff_mat)
        return self.__class__(diff_vec)

    def __matmul__(self, other):
        prod_mat = self.matrix @ other.matrix
        prod_vec = self.matrix_to_vector(prod_mat)
        return self.__class__(prod_vec)

    def __abs__(self):
        norm = np.sqrt(self * self)
        assert norm >= 0.
        return norm

    def __round__(self, n=None):
        if n is None:
            n = 0
        rounded_vec = np.round(self.vector, decimals=n)
        return type(self)(type(self.vector)(rounded_vec))

    def __init__(self, vector):
        self.vector = vector
        self.matrix = self.calculate_matrix()

    def dot(self, other):
        assert len(self.vector) == len(other.vector)
        assert np.shape(self.matrix)[0] == np.shape(self.matrix)[1]
        assert np.shape(other.matrix)[0] == np.shape(other.matrix)[1]
        assert np.shape(self.matrix) == np.shape(other.matrix)
        return np.trace(np.conj(self.matrix.T) @ other.matrix) / (np.shape(self.matrix)[0])


class SpinOperator(Operator):
    def calculate_matrix(self):
        cart = self.convert_cartesian()
        return calculate_spin_matrix(cart)

    def __init__(self, vector):
        assert len(vector) == 4
        super().__init__(vector)


class CartesianSpinOperator(SpinOperator):
    def convert_cartesian(self):
        return self.vector

    @staticmethod
    def matrix_to_vector(matrix):
        return calculate_cartesian_spin_vector_from_matrix(matrix)

    def __init__(self, vector):
        assert len(vector) == 4
        super().__init__(vector)


class SphericalSpinOperator(SpinOperator):
    def convert_cartesian(self):
        return spherical_to_cartesian(self.vector)

    @staticmethod
    def matrix_to_vector(matrix):
        cart_vec = calculate_cartesian_spin_vector_from_matrix(matrix)
        return sp.cartesian_to_spherical(cart_vec)

    def __init__(self, vector):
        assert len(vector) == 4
        # Call SpinObservable Constructor with Spherical Conversion
        super().__init__(vector)


def get_cartesian_vector(operator):
    vec = operator.convert_cartesian()
    return vec


get_cartesian_vectors = np.vectorize(get_cartesian_vector)
