from numpy import pi, sin, cos


def spherical_to_cartesian(spherical_vector):
    # Read and Check Spherical Coordinate Inputs
    i, r, theta, phi = spherical_vector
    assert 0 <= r
    assert 0 <= theta <= pi
    assert 0 <= phi <= 2*pi

    # Calculate Cartesian Coordinates
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return i, x, y, z
