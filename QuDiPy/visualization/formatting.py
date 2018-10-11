""" Useful Formatting Standards for MatPlotLib Visualisations
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np

red = [0.988, 0.078, 0.145]
blue = [0.008, 0.475, 1.]
green = [0.439, 0.678, 0.278]


def format_3d_axes(ax, x_label='x', y_label='y', z_label='z', x_lim=(-1., 1.), y_lim=(-1., 1.), z_lim=(-1., 1.),
                   x_ticks=5, y_ticks=5, z_ticks=5, pad=30, labelpad=60, font_kwargs={'name': 'serif', 'size': 40}):
    ax.set_xlim(x_lim)
    ax.tick_params(axis='x', pad=pad)
    ax.set_xticks(np.linspace(x_lim[0], x_lim[1], x_ticks))
    ax.set_xticklabels(np.linspace(x_lim[0], x_lim[1], x_ticks), **font_kwargs)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='y', pad=pad)
    ax.set_yticks(np.linspace(y_lim[0], y_lim[1], y_ticks))
    ax.set_yticklabels(np.linspace(y_lim[0], y_lim[1], y_ticks), **font_kwargs)
    ax.set_zlim(z_lim)
    ax.tick_params(axis='z', pad=pad)
    ax.set_zticks(np.linspace(z_lim[0], z_lim[1], z_ticks))
    ax.set_zticklabels(np.linspace(z_lim[0], z_lim[1], z_ticks), **font_kwargs)
    ax.set_xlabel(x_label, labelpad=labelpad, **font_kwargs)
    ax.set_ylabel(y_label, labelpad=labelpad, **font_kwargs)
    ax.set_zlabel(z_label, labelpad=labelpad, **font_kwargs)
    return ax
