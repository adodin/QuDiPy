""" Functions for Plotting and Animating Isosurfaces on the Bloch Sphere
Written by: Amro Dodin (Willard Group - MIT)

"""
import numpy as np
from skimage import measure
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from QuDiPy.visualization.formatting import format_3d_axes
import QuDiPy.coordinates.spherical as sp
import matplotlib.animation as an
from mpl_toolkits.mplot3d import axes3d


def plot_cartesian_isosurface(function_grids, levels, contours=[], cont_kwargs=[{}], tri_kwargs=[{'cmap': 'Spectral', 'lw': 1}],
                    fig_kwargs={'figsize': [12., 15.]}, ax_kwargs={}, font_kwargs={'name': 'serif', 'size': 30},
                    show_plot=True, fname=None, fig_num=None):

    # Repeat axis and line formats if only one value is given
    if len(tri_kwargs) == 1:
        tri_kwargs = tri_kwargs * len(function_grids)
    if len(cont_kwargs) == 1:
        cont_kwargs = cont_kwargs * len(function_grids)
    if len(levels) == 1:
        levels = levels * len(function_grids)

    # Checks that all inputs are now the same length
    assert len(function_grids) == len(tri_kwargs) == len(cont_kwargs) == len(levels)

    # Format Figure and Axes
    fig = plt.figure(fig_num, **fig_kwargs)
    ax = fig.add_axes([0.025, 0.15, 0.8, 0.85], projection='3d')

    # Format Axes
    format_3d_axes(ax, font_kwargs=font_kwargs, **ax_kwargs)

    for fg, level, tri_kwarg, cont_kwarg in zip(function_grids, levels, tri_kwargs, cont_kwargs):
        func = fg.data
        x, y, z = fg.grid.coordinates

        # Calculate Projected Functions (Normalized to the same max as original Function)
        yz_func = np.sum(func, axis=0)
        yz_func *= np.max(func)/np.max(yz_func)
        xz_func = np.sum(func, axis=1)
        xz_func *= np.max(func)/np.max(xz_func)
        xy_func = np.sum(func, axis=2)
        xy_func *= np.max(func)/np.max(xy_func)

        # Calculate and Plot Isosurface
        # Factor of 2 for consistency with bloch sphere convention
        del_x, del_y, del_z = 2 * (x[1] - x[0]), 2 * (y[1] - y[0]), 2 * (z[1] - z[0])
        x_min, y_min, z_min = 2 * np.min(x), 2 * np.min(y), 2 * np.min(z)
        x_max, y_max, z_max = 2 * np.max(x), 2 * np.max(y), 2 * np.max(z)
        verts, faces = measure.marching_cubes_classic(volume=func, level=level)
        verts[:, 0] = verts[:, 0] * del_x + x_min
        verts[:, 1] = verts[:, 1] * del_y + y_min
        verts[:, 2] = verts[:, 2] * del_z + z_min
        tri = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], zorder=100., **tri_kwarg)
        for contour in contours:
            lvl = contour * level
            if np.min(yz_func) < lvl < np.max(yz_func):
                cont = np.array(measure.find_contours(yz_func, lvl))
                cont = cont[0]
                cont[:, 0] = cont[:, 0] * del_y + y_min
                cont[:, 1] = cont[:, 1] * del_z + z_min
                tri = ax.plot(xs=x_min * np.ones_like(cont[:, 0]), ys=cont[:, 0], zs=cont[:, 1], zorder=-1., **cont_kwarg)
            if np.min(xz_func) < lvl < np.max(xz_func):
                cont = np.array(measure.find_contours(xz_func, lvl))
                cont = cont[0]
                cont[:, 0] = cont[:, 0] * del_x + x_min
                cont[:, 1] = cont[:, 1] * del_z + z_min
                tri = ax.plot(xs=cont[:, 0], ys=y_max * np.ones_like(cont[:, 0]), zs=cont[:, 1], zorder=-1., **cont_kwarg)
            if np.min(xy_func) < lvl < np.max(xy_func):
                cont = np.array(measure.find_contours(xy_func, lvl))
                cont = cont[0]
                cont[:, 0] = cont[:, 0] * del_x + x_min
                cont[:, 1] = cont[:, 1] * del_y + y_min
                tri = ax.plot(xs=cont[:, 0], ys=cont[:, 1], zs=z_min * np.ones_like(cont[:, 0]), zorder=-1., **cont_kwarg)

    # Draw and Save Figure
    if show_plot:
        plt.show()
    if fname is not None:
        fig.savefig(fname)

    return fig, tri


def animate_cartesian_isosurface(function_grid_series, levels, contours=[], cont_kwargs=[{}],
                       tri_kwargs=[{'cmap': 'Spectral', 'lw': 1}],
                       writer_kwargs={'fps': 30}, ani_kwargs={'blit': False, 'repeat': False, 'interval':33},
                       fig_kwargs={'figsize': [12., 15.]}, ax_kwargs={},
                       font_kwargs={'name': 'serif', 'size': 30}, normalize_level=False,
                       show_plot=True, fname=None, fig_num=None):

    # Repeat axis and line formats if only one value is given
    if len(tri_kwargs) == 1:
        tri_kwargs = tri_kwargs * len(function_grid_series)
    if len(cont_kwargs) == 1:
        cont_kwargs = cont_kwargs * len(function_grid_series)
    if len(levels) == 1:
        levels = levels * len(function_grid_series)

    # Checks that all inputs are now the same length
    assert len(function_grid_series) == len(tri_kwargs) == len(cont_kwargs) == len(levels)

    # Format Figure and Axes
    fig = plt.figure(fig_num, **fig_kwargs)
    ax = fig.add_axes([0.025, 0.15, 0.8, 0.85], projection='3d')

    # Format Axes
    format_3d_axes(ax, font_kwargs=font_kwargs, **ax_kwargs)

    # Set Animation Parameters
    nframes = len(function_grid_series[0])

    # Define Update Function
    def update(ff):
        ax.clear()
        format_3d_axes(ax, font_kwargs=font_kwargs, **ax_kwargs)

        for fgs, level, tri_kwarg, cont_kwarg in zip(function_grid_series, levels, tri_kwargs, cont_kwargs):
            func = fgs[ff][0].data
            x, y, z = fgs[ff][0].grid.coordinates

            # Calculate Projected Functions (Normalized to the same max as original Function)
            yz_func = np.sum(func, axis=0)
            yz_func *= np.max(func) / np.max(yz_func)
            xz_func = np.sum(func, axis=1)
            xz_func *= np.max(func) / np.max(xz_func)
            xy_func = np.sum(func, axis=2)
            xy_func *= np.max(func) / np.max(xy_func)

            # Calculate and Plot Isosurface
            # Factor of 2 for consistency with bloch sphere convention
            del_x, del_y, del_z = 2 * (x[1] - x[0]), 2 * (y[1] - y[0]), 2 * (z[1] - z[0])
            x_min, y_min, z_min = 2 * np.min(x), 2 * np.min(y), 2 * np.min(z)
            x_max, y_max, z_max = 2 * np.max(x), 2 * np.max(y), 2 * np.max(z)
            if len(level) == 1:
                lvl = level[0]
            else:
                lvl = level[ff]
            if normalize_level:
                lvl = lvl * np.max(func)
            verts, faces = measure.marching_cubes_classic(volume=func, level=lvl)
            verts[:, 0] = verts[:, 0] * del_x + x_min
            verts[:, 1] = verts[:, 1] * del_y + y_min
            verts[:, 2] = verts[:, 2] * del_z + z_min
            tri = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], zorder=100., **tri_kwarg)
            for contour in contours:
                clvl = contour * lvl
                if np.min(yz_func) < clvl < np.max(yz_func):
                    cont = np.array(measure.find_contours(yz_func, clvl))
                    cont = cont[0]
                    cont[:, 0] = cont[:, 0] * del_y + y_min
                    cont[:, 1] = cont[:, 1] * del_z + z_min
                    tri = ax.plot(xs=x_min * np.ones_like(cont[:, 0]), ys=cont[:, 0], zs=cont[:, 1], zorder=-1.,
                                  **cont_kwarg)
                if np.min(xz_func) < clvl < np.max(xz_func):
                    cont = np.array(measure.find_contours(xz_func, clvl))
                    cont = cont[0]
                    cont[:, 0] = cont[:, 0] * del_x + x_min
                    cont[:, 1] = cont[:, 1] * del_z + z_min
                    tri = ax.plot(xs=cont[:, 0], ys=y_max * np.ones_like(cont[:, 0]), zs=cont[:, 1], zorder=-1.,
                                  **cont_kwarg)
                if np.min(xy_func) < clvl < np.max(xy_func):
                    cont = np.array(measure.find_contours(xy_func, clvl))
                    cont = cont[0]
                    cont[:, 0] = cont[:, 0] * del_x + x_min
                    cont[:, 1] = cont[:, 1] * del_y + y_min
                    tri = ax.plot(xs=cont[:, 0], ys=cont[:, 1], zs=z_min * np.ones_like(cont[:, 0]), zorder=-1.,
                                  **cont_kwarg)

        return tri,

    # Draw and Save Animation
    ani = an.FuncAnimation(fig, update, frames=range(nframes), **ani_kwargs)
    if show_plot:
        plt.show()
    if fname is not None:
        print('saving '+ fname)
        FFwriter = an.FFMpegWriter(**writer_kwargs)
        ani.save(fname, writer=FFwriter)

    return fig, ani


def plot_spherical_isosurface(function_grids, levels, tri_kwargs=[{'cmap': 'Spectral', 'lw': 1}],
                    fig_kwargs={'figsize': [12., 15.]}, ax_kwargs={}, font_kwargs={'name': 'serif', 'size': 30},
                    show_plot=True, fname=None, fig_num=None):

    # Repeat axis and line formats if only one value is given
    if len(tri_kwargs) == 1:
        tri_kwargs = tri_kwargs * len(function_grids)
    if len(levels) == 1:
        levels = levels * len(function_grids)

    # Checks that all inputs are now the same length
    assert len(function_grids) == len(tri_kwargs) == len(levels)

    # Format Figure and Axes
    fig = plt.figure(fig_num, **fig_kwargs)
    ax = fig.add_axes([0.025, 0.15, 0.8, 0.85], projection='3d')

    # Format Axes
    format_3d_axes(ax, font_kwargs=font_kwargs, **ax_kwargs)

    for fg, level, tri_kwarg in zip(function_grids, levels, tri_kwargs):
        func = fg.data
        r, theta, phi = fg.grid.coordinates

        # Calculate and Plot Isosurface
        # Factor of 2 for consistency with bloch sphere convention
        del_r, del_theta, del_phi = 2 * (r[1] - r[0]),  (theta[1] - theta[0]), (phi[1] - phi[0])
        r_min, theta_min, phi_min = 2 * np.min(r), np.min(theta), np.min(phi)
        verts, faces = measure.marching_cubes_classic(volume=func, level=level)
        v = []
        v.append(verts[:, 0] * del_r + r_min)
        v.append(verts[:, 1] * del_theta + theta_min)
        v.append(verts[:, 2] * del_phi + phi_min)
        v = sp.spherical_to_cartesian(tuple(v))
        verts[:, 0] = v[0]
        verts[:, 1] = v[1]
        verts[:, 2] = v[2]
        tri = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], zorder=100., **tri_kwarg)

    # Draw and Save Figure
    if show_plot:
        plt.show()
    if fname is not None:
        fig.savefig(fname)

    return fig, tri


def animate_spherical_isosurface(function_grid_series, levels, contours=[], cont_kwargs=[{}],
                                 tri_kwargs=[{'cmap': 'Spectral', 'lw': 1}],
                                 writer_kwargs={'fps': 30}, ani_kwargs={'blit': False, 'repeat': False, 'interval':33},
                                 fig_kwargs={'figsize': [12., 15.]}, ax_kwargs={},
                                 font_kwargs={'name': 'serif', 'size': 30}, normalize_level=False,
                                 show_plot=True, fname=None, fig_num=None):

    # Repeat axis and line formats if only one value is given
    if len(tri_kwargs) == 1:
        tri_kwargs = tri_kwargs * len(function_grid_series)
    if len(cont_kwargs) == 1:
        cont_kwargs = cont_kwargs * len(function_grid_series)
    if len(levels) == 1:
        levels = levels * len(function_grid_series)

    # Checks that all inputs are now the same length
    assert len(function_grid_series) == len(tri_kwargs) == len(cont_kwargs) == len(levels)

    # Format Figure and Axes
    fig = plt.figure(fig_num, **fig_kwargs)
    ax = fig.add_axes([0.025, 0.15, 0.8, 0.85], projection='3d')

    # Format Axes
    format_3d_axes(ax, font_kwargs=font_kwargs, **ax_kwargs)

    # Set Animation Parameters
    nframes = len(function_grid_series[0])

    # Define Update Function
    def update(ff):
        ax.clear()
        format_3d_axes(ax, font_kwargs=font_kwargs, **ax_kwargs)

        for fgs, level, tri_kwarg, cont_kwarg in zip(function_grid_series, levels, tri_kwargs, cont_kwargs):
            func = fgs[ff][0].data
            r, theta, phi = fgs[ff][0].grid.coordinates

            # Calculate and Plot Isosurface
            # Factor of 2 for consistency with bloch sphere convention
            del_r, del_theta, del_phi = 2 * (r[1] - r[0]), (theta[1] - theta[0]), (phi[1] - phi[0])
            r_min, theta_min, phi_min = 2 * np.min(r), np.min(theta), np.min(phi)
            if len(level) == 1:
                lvl = level[0]
            else:
                lvl = level[ff]
            if normalize_level:
                lvl = lvl * np.max(func)
            verts, faces = measure.marching_cubes_classic(volume=func, level=lvl)
            v = []
            v.append(verts[:, 0] * del_r + r_min)
            v.append(verts[:, 1] * del_theta + theta_min)
            v.append(verts[:, 2] * del_phi + phi_min)
            v = sp.spherical_to_cartesian(tuple(v))
            verts[:, 0] = v[0]
            verts[:, 1] = v[1]
            verts[:, 2] = v[2]
            tri = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], zorder=100., **tri_kwarg)

        return tri,

    # Draw and Save Animation
    ani = an.FuncAnimation(fig, update, frames=range(nframes), **ani_kwargs)
    if show_plot:
        plt.show()
    if fname is not None:
        print('saving '+ fname)
        FFwriter = an.FFMpegWriter(**writer_kwargs)
        ani.save(fname, writer=FFwriter)

    return fig, ani
