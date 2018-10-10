""" An Example for Generating and Plotting Bloch Sphere Flow Fields
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.util.linalg as la
import QuDiPy.util.grids as gr
import  QuDiPy.visualization.bloch_quiver as bq

# Make Cartesian Grid
res_x = 7
res_y = 7
res_z = 7

x = np.linspace(-1., 1., res_x)
y = np.linspace(-1., 1., res_y)
z = np.linspace(-1., 1., res_z)
cart_grid = gr.CartesianGrid((x, y, z))

# Make Spherical Grid
res_r = 7
res_theta = 7
res_phi = 7

r = np.linspace(0.1, 1., res_r)
theta = np.linspace(0., np.pi, res_theta)
phi = np.linspace(0., 2 * np.pi, res_phi)
spher_grid = gr.SphericalGrid((r, theta, phi))

# Calculate Initial Operators
cart_ops = []
