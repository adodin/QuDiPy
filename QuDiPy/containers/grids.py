""" A set of utilities for defining generic grids
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.coordinates.spherical as sp
import QuDiPy.math.linear_algebra as la
from QuDiPy.coordinates.cartesian import calculate_cartesian_mesh, calculate_differentials, \
    calculate_cartesian_volume_element, calculate_cartesian_gradient, calculate_cartesian_divergence


class Grid:
    def __eq__(self, other):
        return np.array_equal(self.cartesian, other.cartesian)

    def __getitem__(self, item):
        coord = self._container_type(c[item] for c in self.cartesian)
        return coord

    def mesh_item(self, *item):
        coord = self._container_type(c[item] for c in self.mesh)
        return coord

    def __iter__(self):
        flat_grids = [c.flatten() for c in self.cartesian]
        return (self._container_type(a) for a in zip(*flat_grids))

    def mesh_iter(self):
        flat_grids = [c.flatten() for c in self.mesh]
        return (self._container_type(a) for a in zip(*flat_grids))

    def __init__(self, coordinates):
        self._container_type = type(coordinates)
        self.coordinates = coordinates
        self.mesh = self._container_type(np.meshgrid(*coordinates, indexing='ij'))
        self.cartesian = self._calculate_cartesian_mesh(self.mesh)
        self.volume = self.calculate_volume()

    @property
    def shape(self):
        return np.shape(self.mesh[0])


class CartesianGrid(Grid):
    @staticmethod
    def _calculate_cartesian_mesh(mesh):
        return mesh

    def calculate_volume(self):
        return calculate_cartesian_volume_element(self.coordinates)

    def gradient(self, funct):
        return self._container_type(calculate_cartesian_gradient(funct, self.coordinates))

    def divergence(self, vector_funct):
        return calculate_cartesian_divergence(vector_funct, self.coordinates)

    def __init__(self, coordinates):
        super().__init__(coordinates)


class SphericalGrid(Grid):
    @staticmethod
    def _calculate_cartesian_mesh(mesh):
        return sp.spherical_to_cartesian(mesh)

    def gradient(self, funct):
        return self._container_type(sp.calculate_spherical_gradient(funct, self.coordinates, self.mesh))

    def divergence(self, vector_funct):
        return sp.calculate_spherical_divergence(vector_funct, self.coordinates, self.mesh)

    def calculate_volume(self):
        differentials = calculate_differentials(self.coordinates)
        diff_grid = np.meshgrid(*differentials, indexing='ij')
        return sp.calculate_spherical_volume_element(self.mesh[0], self.mesh[1], diff_grid)

    def __init__(self, coordinates):
        super().__init__(coordinates)


class DataGrid:
    def __init__(self, data, grid):
        assert grid.shape == np.shape(data)
        self.data = data
        self.grid = grid

    def __eq__(self, other):
        grid_eq = self.grid == other.grid
        dat_eq = np.array_equal(self.data, other.data)
        return grid_eq and dat_eq

    def __getitem__(self, item):
        data = self.data[item]
        coord = self.grid[item]
        return data, coord

    def mesh_item(self, *item):
        data = self.data[item]
        coord = self.grid.mesh_item(*item)
        return data, coord

    def __iter__(self):
        data_iter = self.data.flatten()
        grid_iter = self.grid.__iter__()
        return zip(data_iter, grid_iter)

    def mesh_iter(self):
        data_iter = self.data.flatten()
        grid_iter = self.grid.mesh_iter()
        return zip(data_iter, grid_iter)

    def __add__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        sum = self.data + other.data
        return self.like(sum)

    def __mul__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        prod = self.data * other.data
        self_type = type(np.take(self.data, 0))
        prod_type = type(np.take(prod, 0))
        if self_type is prod_type:
            return self.like(prod)
        else:
            return DataGrid(prod, self.grid)

    def __rmul__(self, other):
        prod = other * self.data
        return self.like(prod)

    def __matmul__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        prod = self.data @ other.data
        return self.like(prod)

    def __sub__(self, other):
        assert self.grid == other.grid
        assert np.shape(self.data) == np.shape(other.data)
        diff = self.data - other.data
        return self.like(diff)

    def __abs__(self):
        return np.sum(abs(self.data)*self.grid.volume)

    def like(self, data):
        return type(self)(data, self.grid)

    def gradient(self, operator_type, container_type=list):
        assert not issubclass(type(self), VectorGrid)
        grad = container_type(self.grid.gradient(self.data))
        return VectorGrid(grad, self.grid, operator_type)

    @property
    def shape(self):
        return self.grid.shape


class VectorGrid(DataGrid):
    def __init__(self, operators, grid, operator_type=None):
        # If already given an operator mesh just construct normally
        if operator_type is None:
            super().__init__(np.array(operators), grid)
        # If given an operator_type and vectorized coordinate mesh construct operators
        else:
            # Flatten input arrays using same technique as Grid.__iter__
            flat_grids = [c.flatten() for c in operators]
            container_type = type(operators)
            ops = []
            for a in zip(*flat_grids):
                ops.append(operator_type(container_type(a)))
            # Reshape Operators to match grid & construct
            ops = np.reshape(ops, grid.shape)
            super().__init__(ops, grid)

        # Pull Necessary Information from first operator
        op = np.take(self.data, 0)
        self._operator_type = type(op)
        self._container_type = op._container_type
        self._operator_dim = op._dim

    def divergence(self):
        assert hasattr(self, 'vectors')
        div = self.grid.divergence(self.vectors)
        # Note: cannot use self.like() since don't want to output VectorGrid
        return DataGrid(div, self.grid)

    @property
    def vectors(self, container_type=None):
        if container_type is None:
            container_type = self._container_type
        vecs = container_type(la.get_vectors(self.data))
        return vecs

    @property
    def cartesian_vectors(self, container_type=None):
        if container_type is None:
            container_type = self._container_type
        vecs = container_type(la.get_cartesian_vectors(self.data))
        return vecs
