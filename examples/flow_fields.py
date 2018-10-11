""" An Example for Generating and Plotting Bloch Sphere Flow Fields
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.util.linalg as la
import QuDiPy.util.grids as gr
from QuDiPy.dynamics.unitary_dynamics import grid_unitary_derivative
from QuDiPy.dynamics.bloch_dynamics import grid_bloch_derivative
import  QuDiPy.visualization.bloch_quiver as bq
from QuDiPy.visualization.formatting import red, green
from QuDiPy.util.constants import hbar

# Dynamics Properties
delta = 1.0
vv = 0.0
g_diss = 1.0
g_deph = 2.0
r_pump = 0.5
ham = la.CartesianSpinOperator((0., vv.real, vv.imag, delta))

# Make Cartesian Grid
res_x = 5
res_y = 5
res_z = 5

x = np.linspace(-0.5, 0.5, res_x)
y = np.linspace(-0.5, 0.5, res_y)
z = np.linspace(-0.5, 0.5, res_z)
cart_grid = gr.CartesianGrid((x, y, z))

# Make Spherical Grid
res_r = 5
res_theta = 5
res_phi = 5

r = np.linspace(0.1, 1., res_r)
theta = np.linspace(0., np.pi, res_theta)
phi = np.linspace(0., 2 * np.pi, res_phi)
spher_grid = gr.SphericalGrid((r, theta, phi))

# Initialize Operators
cart_ops = gr.initialize_operator_grid(cart_grid, la.CartesianSpinOperator, i_coord=0.5)
spher_ops = gr.initialize_operator_grid(spher_grid, la.CartesianSpinOperator, i_coord=0.5)

# Calculate Unitary Derivatives for Both Grids
cart_rho_dot = hbar * grid_unitary_derivative(cart_ops, ham)
spher_rho_dot = hbar * grid_bloch_derivative(spher_ops, ham, g_diss, g_diss, r_pump)

# Set Formatting for Plots
quiver_kwargs=[{'linewidth': 2., 'colors': red}, {'linewidth': 2., 'colors': green}]
proj_quiver_kwargs=[{'linewidth': 2., 'colors': red, 'alpha': 0.5}, {'linewidth': 2., 'colors': green, 'alpha': 0.0}]

bq.plot_flow([cart_ops, spher_ops], quiver_kwargs=quiver_kwargs,
             proj_quiver_kwargs=proj_quiver_kwargs, fig_num='Operators')
bq.plot_flow([cart_rho_dot, spher_rho_dot], quiver_kwargs=quiver_kwargs,
             proj_quiver_kwargs=proj_quiver_kwargs, fig_num='Flow')

