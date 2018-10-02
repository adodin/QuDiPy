""" Useful Functions for converting between 3D spherical and cartesian coordinates
Written by: Amro Dodin (Willard Group - MIT)

"""

from numpy import pi, sin, cos, sqrt, arccos, arctan
import numpy as np


def spherical_to_cartesian(spherical_vector):
    # Read and Check Spherical Coordinate Inputs
    if len(spherical_vector) == 4:
        i, r, theta, phi = spherical_vector
    else:
        r, theta, phi = spherical_vector

    # Calculate Cartesian Coordinates
    x_r = r.real * sin(theta.real) * cos(phi.real)
    y_r = r.real * sin(theta.real) * sin(phi.real)
    z_r = r.real * cos(theta.real)
    x_i = r.imag * sin(theta.imag) * cos(phi.imag)
    y_i = r.imag * sin(theta.imag) * sin(phi.imag)
    z_i = r.imag * cos(theta.imag)
    x = x_r + 1.j * x_i
    y = y_r + 1.j * y_i
    z = z_r + 1.j * z_i
    if len(spherical_vector) == 4:
        return i, x, y, z
    else:
        return x, y, z


def cartesian_to_spherical(cartesian_vector):
    # Parse 3 and 4-vectors
    if len(cartesian_vector) == 4:
        i, x, y, z = cartesian_vector
    else:
        x, y, z = cartesian_vector

    # Convert Real Part To Spherical
    r_r = sqrt(x.real**2 + y.real**2 + z.real**2)
    if r_r != 0:
        theta_r = arccos(z.real / (r_r+1E-50))
        phi_r = np.arctan2(y.real, x.real)
    else:
        theta_r = 0
        phi_r = 0

    # Convert Imaginary Part to Spherical
    r_i = sqrt(x.imag ** 2 + y.imag ** 2 + z.imag ** 2)
    if r_i != 0:
        theta_i = arccos(z.imag / (r_i + 1E-50))
        phi_i = np.arctan2(y.imag, x.imag)
    else:
        theta_i = 0
        phi_i = 0

    # Combine Real and Imaginary Parts
    r = r_r + 1.j * r_i
    theta = theta_r + 1.j * theta_i
    phi = phi_r + 1.j * phi_i

    # Output Results
    if len(cartesian_vector) == 4:
        return i, r, theta, phi
    else:
        return r, theta, phi


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
