""" Example Script so demonstrate Isosurface Plotting
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.containers.grids as gr
import QuDiPy.visualization.isosurfaces as iso
from QuDiPy.visualization.formatting import red
import QuDiPy.containers.timeseries as ts


# Make Cartesian Grid
res_x = 50
res_y = 50
res_z = 50

x = np.linspace(-0.5, 0.5, res_x)
y = np.linspace(-0.5, 0.5, res_y)
z = np.linspace(-0.5, 0.5, res_z)
cart_grid = gr.CartesianGrid((x, y, z))

xg, yg, zg = cart_grid.mesh
grid_list = []
time_list = []
for t in range(50):
    data = np.exp(-(xg-0.25*np.cos(t*np.pi/25.))**2/0.25 - (yg-0.25*np.sin(t*np.pi/25.))**2/0.25 - (zg)**2/0.25)
    funct_grid = gr.DataGrid(data, cart_grid)
    grid_list.append(funct_grid)
    time_list.append(t)

cartesian_series = ts.TimeSeries(grid_list, time_list)

# Make Spherical Grid
res_r = 50
res_theta = 50
res_phi = 50

r = np.linspace(0.0, 0.5, res_r)
theta = np.linspace(0., np.pi, res_theta)
phi = np.linspace(0., 2 * np.pi, res_phi)
spher_grid = gr.SphericalGrid((r, theta, phi))
xg, yg, zg = spher_grid.cartesian
grid_list = []
time_list = []
for t in range(50):
    data = np.exp(-(xg-0.25*np.cos(t*np.pi/25.))**2/0.25 - (yg-0.25*np.sin(t*np.pi/25.))**2/0.25 - (zg)**2/0.25)
    funct_grid = gr.DataGrid(data, spher_grid)
    grid_list.append(funct_grid)
    time_list.append(t)

spherical_series = ts.TimeSeries(grid_list, time_list)

'''fig1, ani1 = iso.animate_cartesian_isosurface([cartesian_series], [[0.9]], [1.5, 1.0],
                                              cont_kwargs=[{'color': red, 'alpha': 0.5}])'''

fig2, ani2 = iso.animate_spherical_isosurface([spherical_series], [[0.9]], [1.5, 1.0],
                                              cont_kwargs=[{'color': red, 'alpha': 0.5}])

