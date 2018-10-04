""" A set of utilities for defining generic grids
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.util.spherical as sp
import QuDiPy.util.linalg as la


class Grid:
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


def calculate_cartesian_grid(coordinates):
    return np.meshgrid(*coordinates, indexing='ij')


def calculate_differentials(coordinates):
    differentials = []
    for coord in coordinates:
        diff = []
        diff.append(0.5 * abs(coord[1] - coord[0]))
        for ii in range(1, len(coord)-1):
            left = 0.5 * (coord[ii - 1] + coord[ii])
            right = 0.5 * (coord[ii] + coord[ii + 1])
            diff.append(abs(right - left))
        diff.append(0.5 * abs(coord[-1] - coord[-2]))
        differentials.append(diff)
    return tuple(differentials)


def calculate_cartesian_volume_element(coordinates):
    differentials = calculate_differentials(coordinates)
    diff_grid = np.meshgrid(*differentials, indexing='ij')
    return np.product(diff_grid, axis=0)


def calculate_cartesian_gradient(funct, coordinates):
    return np.gradient(funct, *coordinates, edge_order=2)


def calculate_cartesian_divergence(vector_funct, coordinates):
    assert len(vector_funct) == len(coordinates)
    divergence = np.zeros_like(vector_funct[0])
    for i in range(len(coordinates)):
        fi = vector_funct[i]
        ci = coordinates[i]
        divergence += np.gradient(fi, ci, axis=i, edge_order=2)
    return divergence


class CartesianGrid(Grid):
    def calculate_grid(self):
        return calculate_cartesian_grid(self.coordinates)

    def calculate_volume(self):
        return calculate_cartesian_volume_element(self.coordinates)

    def gradient(self, funct):
        return calculate_cartesian_gradient(funct, self.coordinates)

    def divergence(self, vector_funct):
        return calculate_cartesian_divergence(vector_funct, self.coordinates)

    def __init__(self, coordinates):
        super().__init__(coordinates)


class SphericalGrid(Grid):
    def calculate_grid(self):
        return sp.spherical_to_cartesian_grid(self.coordinates)

    def gradient(self, funct):
        return sp.calculate_spherical_gradient(funct, self.coordinates, self.grid)

    def divergence(self, vector_funct):
        return sp.calculate_spherical_divergence(vector_funct, self.coordinates, self.grif)

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

    def __eq__(self, other):
        grid_eq = self.grid == other.grid
        dat_eq = np.array_equal(np.round(self.data, 7), np.round(other.data, 7))
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


def vectorize_operator_grid(operator_grid):
    vec_data = la.get_cartesian_vectors(operator_grid.data)
    grid = operator_grid.grid
    return type(operator_grid)(vec_data, grid)