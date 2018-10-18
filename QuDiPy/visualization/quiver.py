""" Visualizing Operator Fields as Bloch Sphere Quiver Plots
Written by: Amro Dodin (Willard Group - MIT)

"""

# Import Packages
import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
from QuDiPy.visualization.formatting import format_3d_axes, red
from mpl_toolkits.mplot3d import axes3d


def plot_flow(flow_grids, quiver_kwargs=[{'linewidth': 2., 'colors': red}],
              proj_quiver_kwargs=[{'linewidth': 2., 'colors': red, 'alpha': 0.5}],
              fig_kwargs={'figsize': [12., 15.]}, ax_kwargs={}, font_kwargs={'name': 'serif', 'size': 40},
              show_plot=True, fname=None, fig_num=None):

    # Repeat axis and line formats if only one value is given
    if len(quiver_kwargs) == 1:
        quiver_kwargs = quiver_kwargs * len(flow_grids)
    if len(proj_quiver_kwargs) == 1:
        proj_quiver_kwargs = proj_quiver_kwargs * len(flow_grids)

    # Checks that all inputs are now the same length
    assert len(flow_grids) == len(quiver_kwargs) == len(proj_quiver_kwargs)

    # Initialize Figure and Axes
    fig = plt.figure(fig_num, **fig_kwargs)
    ax = fig.gca(projection='3d')

    # Get Axis Parameters
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    z_mins = []
    z_maxs = []
    for fg in flow_grids:
        fg = fg.grid
        x_mins.append(2 * np.min(fg.cartesian[0]).real)
        x_maxs.append(2 * np.max(fg.cartesian[0]).real)
        y_mins.append(2 * np.min(fg.cartesian[1]).real)
        y_maxs.append(2 * np.max(fg.cartesian[1]).real)
        z_mins.append(2 * np.min(fg.cartesian[2]).real)
        z_maxs.append(2 * np.max(fg.cartesian[2]).real)
    x_lim = np.min(x_mins), np.max(x_maxs)
    y_lim = np.min(y_mins), np.max(y_maxs)
    z_lim = np.min(z_mins), np.max(z_maxs)

    # Format Axes
    format_3d_axes(ax, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, font_kwargs=font_kwargs, **ax_kwargs)

    # Generate Quiver Plot
    for flow_grid, line_format, proj_format in zip(flow_grids, quiver_kwargs, proj_quiver_kwargs):
        # Grab Coordinates
        xc, yc, zc = tuple(flow_grid.grid.cartesian)           # Factor of 2 for Consistency with Bloch Sphere Convention
        x = 2 * xc.real
        y = 2 * yc.real
        z = 2 * zc.real
        mid_x = int(np.floor(np.shape(x)[0] / 2))
        mid_y = int(np.floor(np.shape(y)[1] / 2))
        mid_z = int(np.floor(np.shape(z)[2] / 2))

        # Vectorize Operators
        flow = flow_grid.cartesian_vectors

        quiv = ax.quiver(x, y, z, flow[0], flow[1], flow[2], length=0.25, **line_format)
        quiv = ax.quiver(x[:, mid_x, :], x_lim[1] * np.ones_like(y[:, mid_x, :]), z[:, mid_x, :],
                         flow[0][:, mid_x, :], np.zeros_like(flow[1][:, mid_x, :]), flow[2][:, mid_x, :],
                         length=0.25, zorder=-1., **proj_format)
        quiv = ax.quiver(y_lim[0] * np.ones_like(x[mid_y, :, :]), y[mid_y, :, :], z[mid_y,:, :],
                         np.zeros_like(flow[0][mid_y, :, :]), flow[1][mid_y, :, :], flow[2][mid_y, :, :],
                         length=0.25, zorder=-1., **proj_format)
        quiv = ax.quiver(x[:, :, mid_z], y[:, :, mid_z], z_lim[0] * np.ones_like(z[:, :, mid_z]),
                         flow[0][:, :, mid_z], flow[1][:, :, mid_z], np.zeros_like(flow[2][:, :, mid_z]),
                         length=0.25,  **proj_format)

    # Plot and Save Figure
    if show_plot:
        plt.show()
    if fname is not None:
        fig.savefig(fname)

    return fig, quiv