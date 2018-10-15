""" Useful Functions for working with Cartesian Coordinates
Written by: Amro Dodin (Willard Group - MIT

"""
import numpy as np


# Setup Cartesian Meshes
def calculate_cartesian_mesh(coordinates):
    return np.meshgrid(*coordinates, indexing='ij')


# Cartesian Differential Functions
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


# Cartesian Integral Functions
def calculate_cartesian_volume_element(coordinates):
    differentials = calculate_differentials(coordinates)
    diff_grid = np.meshgrid(*differentials, indexing='ij')
    return np.product(diff_grid, axis=0)
