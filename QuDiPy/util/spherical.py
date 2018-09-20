""" Useful Functions for converting between 3D spherical and cartesian coordinates
Written by: Amro Dodin (Willard Group - MIT)

"""

from numpy import pi, sin, cos
import numpy as np


def spherical_to_cartesian(spherical_vector):
    # Read and Check Spherical Coordinate Inputs
    if len(spherical_vector) == 4:
        i, r, theta, phi = spherical_vector
    else:
        r, theta, phi = spherical_vector
    if hasattr(r, '__iter__'):
        assert np.shape(r) == np.shape(theta) == np.shape(phi)
        for rad, t, p in zip(np.nditer(r), np.nditer(theta), np.nditer(phi)):
            assert rad.imag == 0 and 0 <= rad
            assert t.imag == 0 and 0 <= t <= pi
            assert p.imag == 0 and 0 <= p <= 2 * pi
    else:
        assert r.imag == 0 and 0 <= r
        assert theta.imag == 0 and 0 <= theta <= pi
        assert phi.imag == 0 and 0 <= phi <= 2*pi

    # Calculate Cartesian Coordinates
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    if len(spherical_vector) == 4:
        return i, x, y, z
    else:
        return x, y, z


def spherical_to_cartesian_grid(coordinates):
    rs, thetas, phis = coordinates
    polar_grid = np.meshgrid(rs, thetas, phis, indexing='ij')
    return spherical_to_cartesian(polar_grid)


def calculate_spherical_volume_element(r, theta, diffs):
    assert len(diffs) == 3
    dr, dtheta, dphi = diffs
    return r ** 2 * sin(theta) * dr * dtheta * dphi


def calculate_spherical_gradient(funct, coordinates, grid):
    partial_derivatives = np.gradient(funct, *coordinates)
    r, theta, phi = grid
    partial_derivatives[1] = partial_derivatives[1]/r
    partial_derivatives[2] = partial_derivatives[2]/(r*sin(theta))
    return partial_derivatives

def calculate_spherical_divergence(vector_funct, coordinates, grid):
    r, theta, phi = grid
    v_funct = []
    v_funct.append(vector_funct[0] * r ** 2)
    v_funct.append(vector_funct[1] * sin(theta))
    v_funct.append(vector_funct[2])

    divergence = np.zeros_like(v_funct[0])
    r_div = np.gradient(v_funct[0], coordinates[0], axis=0, edge_order=2)/(r ** 2)
    divergence += r_div
    t_div = np.gradient(v_funct[1], coordinates[1], axis=1, edge_order=2)/(r * sin(theta))
    divergence += t_div
    p_div = np.gradient(v_funct[2], coordinates[2], axis=2, edge_order=2)/(r * sin(theta))
    divergence += p_div
    return divergence
