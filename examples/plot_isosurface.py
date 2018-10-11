""" Example Script so demonstrate Isosurface Plotting
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.util.grids as gr
import QuDiPy.visualization.isosurfaces as iso
from QuDiPy.visualization.formatting import red, green


# Make Cartesian Grid
res_x = 50
res_y = 50
res_z = 50

x = np.linspace(-0.5, 0.5, res_x)
y = np.linspace(-0.5, 0.5, res_y)
z = np.linspace(-0.5, 0.5, res_z)
cart_grid = gr.CartesianGrid((x, y, z))

xg, yg, zg = cart_grid.grid
data = np.exp(-xg**2/0.25 - yg**2/0.25 - zg**2/0.25)
funct_grid = gr.GridData(data, cart_grid)

iso.plot_cartesian_isosurface([funct_grid], [0.5], [1.5, 1.0], cont_kwargs=[{'color': red, 'alpha': 0.5}])
