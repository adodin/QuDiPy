""" Example Script so demonstrate Isosurface Plotting
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.containers.grids as gr
import QuDiPy.visualization.isosurfaces as iso
from QuDiPy.visualization.formatting import red

# Make Cartesian Grid
res_x = 50
res_y = 50
res_z = 50

x = np.linspace(-0.5, 0.5, res_x)
y = np.linspace(-0.5, 0.5, res_y)
z = np.linspace(-0.5, 0.5, res_z)
cart_grid = gr.CartesianGrid((x, y, z))

xg, yg, zg = cart_grid.grid
data = np.exp(-(xg-0.25)**2/0.25 - (yg)**2/0.25 - (zg)**2/0.25)
funct_grid = gr.GridData(data, cart_grid)

# Make Spherical Grid
res_r = 50
res_theta = 50
res_phi = 50

r = np.linspace(0.0, 0.5, res_r)
theta = np.linspace(0., np.pi, res_theta)
phi = np.linspace(0., 2 * np.pi, res_phi)
spher_grid = gr.SphericalGrid((r, theta, phi))
xg, yg, zg = spher_grid.grid
data = np.exp(-(xg-0.25)**2/0.25 - yg**2/0.25 - zg**2/0.25)
spher_funct = gr.GridData(data, spher_grid)

iso.plot_cartesian_isosurface([funct_grid], [0.9], [1.5, 1.0], cont_kwargs=[{'color': red, 'alpha': 0.5}])
iso.plot_spherical_isosurface([spher_funct], [0.9])
