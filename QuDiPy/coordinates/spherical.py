""" Useful Functions for Working with Spherical Coordinates
Written by: Amro Dodin (Willard Group - MIT)

"""

from numpy import pi, sin, cos, sqrt, arccos, arctan
import numpy as np


# Spherical <-> Cartesian Conversion (Vectorized)
def spherical_to_cartesian(spher_vec):
    """ Converts a vector from spherical to cartesian coordinates.

    :param spher_vec: A container of length 3 (r, theta, phi) or length 4 (i, r, theta, phi). Maybe Complex.
    :return: Cartesian container of the same length and type as the input (x, y, z) or (i, x, y, z)

    Note: If a complex vector is given, converts the real and imaginary parts separately then adds them up.
    """
    # Parse 3 and 4-Vectors
    container_type = type(spher_vec)
    if len(spher_vec) == 4:
        i, r, theta, phi = spher_vec
    else:
        r, theta, phi = spher_vec

    # Split Real and Complex Parts
    r_r, r_i = r.real, r.imag
    t_r, t_i = theta.real, theta.imag
    p_r, p_i = phi.real, phi.imag

    # Convert Real Part to Cartesian
    x_r = r_r * sin(t_r) * cos(p_r)
    y_r = r_r * sin(t_r) * sin(p_r)
    z_r = r_r * cos(t_r)

    # Convert Imaginary Part to Cartesian
    x_i = r_i * sin(t_i) * cos(p_i)
    y_i = r_i * sin(t_i) * sin(p_i)
    z_i = r_i * cos(t_i)

    # Combine Real and Imaginary Parts
    x = x_r + 1.j * x_i
    y = y_r + 1.j * y_i
    z = z_r + 1.j * z_i

    # Output the Converted Coordinates
    if len(spher_vec) == 4:
        return container_type((i, x, y, z))
    else:
        return container_type((x, y, z))


def cartesian_to_spherical(cart_vec):
    """ Converts a vector from cartesian to spherical coordinates.

    :param cart_vec: A container of length 3 (x, y, z) or length 4 (i, x, y, z). Maybe Complex.
    :return: Spherical container of the same length and type as the input (r, theta, phi) or (i, r, theta, phi)

    Note: If a complex vector is given, converts the real and imaginary parts separately then adds them up.
    """
    # Parse 3 and 4-vectors
    container_type = type(cart_vec)
    if len(cart_vec) == 4:
        i, x, y, z = cart_vec
    else:
        x, y, z = cart_vec

    # Split Real and Complex Parts
    x_r, x_i = x.real, x.imag
    y_r, y_i = y.real, y.imag
    z_r, z_i = z.real, z.imag

    # Convert Real Part To Spherical
    r_r = sqrt(x_r ** 2 + y_r ** 2 + z_r ** 2)
    if r_r != 0:
        t_r = arccos(z_r / (r_r+1E-50))
        p_r = np.arctan2(y_r, x_r)
    else:
        t_r = 0
        p_r = 0

    # Convert Imaginary Part to Spherical
    r_i = sqrt(x_i ** 2 + y_i ** 2 + z_i ** 2)
    if r_i != 0:
        t_i = arccos(z_i / (r_i + 1E-50))
        p_i = np.arctan2(y_i, x_i)
    else:
        t_i = 0
        p_i = 0

    # Combine Real and Imaginary Parts
    r = r_r + 1.j * r_i
    theta = t_r + 1.j * t_i
    phi = p_r + 1.j * p_i

    # Output Results
    if len(cart_vec) == 4:
        return container_type((i, r, theta, phi))
    else:
        return container_type((r, theta, phi))


# Spherical <-> Cartesian Mesh Conversion
def spherical_to_cartesian_mesh(spher_coords):
    """ Generates a spherical mesh in cartesian coordinates.

    :param spher_coords: A length 3 set of spherical coordinates used to generate a mesh
    :return: (x, y, z) Cartesian mesh
    """
    rs, thetas, phis = spher_coords
    polar_grid = np.meshgrid(rs, thetas, phis, indexing='ij')
    return spherical_to_cartesian(polar_grid)


def cartesian_to_spherical_mesh(cart_coords):
    """ Generates a cartesian mesh in spherical coordinates.

        :param cart_coords: A length 3 set of cartesian coordinates used to generate a mesh
        :return: (r, theta, phi) Spherical mesh
        """
    xs, ys, zs = cart_coords
    cart_mesh = np.meshgrid(xs, ys, zs, indexing='ij')
    return cartesian_to_spherical(cart_mesh)


# Differential Calculus Functions
def calculate_spherical_gradient(data, coords, grid):
    partial_derivatives = np.gradient(data, *coords)
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


# Integral Calculus Functions
def calculate_spherical_volume_element(r, theta, diffs):
    assert len(diffs) == 3
    dr, dtheta, dphi = diffs
    return r ** 2 * sin(theta) * dr * dtheta * dphi
