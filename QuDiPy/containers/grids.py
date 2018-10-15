""" A set of utilities for defining generic grids
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.coordinates.spherical as sp
import QuDiPy.math.linear_algebra as la
from QuDiPy.coordinates.cartesian import calculate_cartesian_mesh, calculate_differentials, \
    calculate_cartesian_volume_element, calculate_cartesian_gradient, calculate_cartesian_divergence


class Grid:
    def calculate_grid(self, other=None):
        if other is None:
            return self._calculate_grid(self.coordinates)
        else:
            return self._calculate_grid(other)

    def __eq__(self, other):
        # Float equality of Grids
        return np.array_equal(np.round(self.grid, 7), np.round(other.grid, 7))

    def __getitem__(self, item):
        coord = tuple(c[item] for c in self.grid)
        return coord

    def __iter__(self):
        flat_grids = tuple(c.flatten() for c in self.grid)
        return zip(*flat_grids)

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.grid = self.calculate_grid()
        self.volume = self.calculate_volume()

    @property
    def shape(self):
        return np.shape(self.grid[0])


class CartesianGrid(Grid):
    @staticmethod
    def _calculate_grid(coords):
        return calculate_cartesian_mesh(coords)

    def calculate_volume(self):
        return calculate_cartesian_volume_element(self.coordinates)

    def gradient(self, funct):
        return tuple(calculate_cartesian_gradient(funct, self.coordinates))

    def divergence(self, vector_funct):
        return calculate_cartesian_divergence(vector_funct, self.coordinates)

    def __init__(self, coordinates):
        super().__init__(coordinates)


class SphericalGrid(Grid):
    @staticmethod
    def _calculate_grid(coords):
        return sp.spherical_to_cartesian_mesh(coords)

    def gradient(self, funct):
        return tuple(sp.calculate_spherical_gradient(funct, self.coordinates, self.grid))

    def divergence(self, vector_funct):
        return sp.calculate_spherical_divergence(vector_funct, self.coordinates, self.grid)

    def calculate_volume(self):
        differentials = calculate_differentials(self.coordinates)
        diff_grid = np.meshgrid(*differentials, indexing='ij')
        return sp.calculate_spherical_volume_element(self.grid[0], self.grid[1], diff_grid)


class GridData:
    def __getitem__(self, item):
        if type(self.data) == tuple:
            data = tuple(d[item] for d in self.data)
        else:
            data = self.data[item]
        coord = self.grid[item]
        return data, coord

    def __iter__(self):
        if type(self.data) == tuple:
            flat_data = tuple(d.flatten() for d in self.data)
            data_iter = zip(*flat_data)
        else:
            data_iter = self.data.flatten()
        grid_iter = self.grid.__iter__()
        return zip(data_iter, grid_iter)

    def __add__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        if type(self.data) == tuple:
            dat_sum = []
            for s, o in zip(self.data, other.data):
                dat_sum.append(s + o)
            dat_sum = tuple(dat_sum)
        else:
            dat_sum = self.data + other.data
        return self.like(dat_sum)

    def __mul__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        if type(self.data) == tuple:
            prod_sum = []
            for s, o in zip(self.data, other.data):
                prod_sum.append(s * o)
            prod_sum = tuple(prod_sum)
        else:
            prod_sum = self.data * other.data
        return self.like(prod_sum)

    def __rmul__(self, other):
        assert np.shape(self.data)
        if type(self.data) == tuple:
            prod_sum = []
            for s in self.data:
                prod_sum.append(other * s)
            prod_sum = tuple(prod_sum)
        else:
            prod_sum = other * self.data
        return self.like(prod_sum)

    def __matmul__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        if type(self.data) == tuple:
            prod_sum = []
            for s, o in zip(self.data, other.data):
                prod_sum.append(s @ o)
            prod_sum = tuple(prod_sum)
        else:
            prod_sum = self.data @ other.data
        return self.like(prod_sum)

    def __sub__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        if type(self.data) == tuple:
            dat_sum = []
            for s, o in zip(self.data, other.data):
                dat_sum.append(s - o)
            dat_sum = tuple(dat_sum)
        else:
            dat_sum = self.data - other.data
        return self.like(dat_sum)

    def __abs__(self):
        return abs(np.sum(self.data))

    def __eq__(self, other):
        grid_eq = self.grid == other.grid
        if type(self.data) == tuple:
            dat_eq = True
            for s, o in zip(self.data, other.data):
                dat_eq = dat_eq and np.array_equal(s, o)
        else:
            dat_eq = np.array_equal(self.data, other.data)
        return grid_eq and dat_eq

    def __init__(self, data, grid):
        for g in grid.grid:
            if type(data) == tuple:
                for d in data:
                    assert np.shape(g) == np.shape(d)
            else:
                assert np.shape(g) == np.shape(data)
        self.data = data
        self.grid = grid

    def like(self, data):
        return type(self)(data, self.grid)


def vectorize_operator_grid(operator_grid):
    vec_data = la.get_cartesian_vectors(operator_grid.data)
    return operator_grid.like(vec_data)


def unvectorize_operator_grid(vector_grid, operator_type, i_coord=None):
    ops = []
    for vec, coords in vector_grid:
        if i_coord is not None:
            vec = (i_coord, *vec)
        ops.append(operator_type(vec))
    ops = np.reshape(ops, vector_grid.grid.shape)
    return vector_grid.like(ops)


def initialize_operator_grid(grid, operator_type, i_coord=None):
    ops = []
    for coords in grid:
        if i_coord is not None:
            coords = (i_coord, *coords)
        ops.append(operator_type(coords))
    ops = np.reshape(ops, grid.shape)
    return GridData(ops, grid)
