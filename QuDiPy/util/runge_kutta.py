""" Set of Functions for Explicit Runge-Kutta Style Integrators
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np
import QuDiPy.util.timeseries as ts

# Default Butcher tableau using the RK4 method
rk4_a = [[0., 0., 0., 0.],
         [0.5, 0., 0., 0.],
         [0., 0.5, 0., 0.],
         [0., 0., 1., 0.]]
rk4_b = [1./6., 1./3., 1./3., 1./6.]
rk4_c = [0., 0.5, 0.5, 1.]


def rk_step(y0, t0, dt, deriv, a=rk4_a, b=rk4_b, c=rk4_c, deriv_kwargs={}):
    assert len(b) == len(c) == len(a[0])
    assert np.round(np.sum(b), 7) == 1.
    k = []
    for ai, ci in zip(a, c):
        assert 0. <= ci <= 1.
        assert np.round(ci, 7) == np.round(np.sum(ai), 7)
        ti = t0 + ci * dt
        yi = y0
        for i, ki in enumerate(k):
            yi += dt * ai[i] * ki
        k.append(deriv(yi, ti, **deriv_kwargs))
    for bi, ki in zip(b, k):
        y0 += bi * ki * dt
    t0 += dt
    return y0, t0


def rk_integrate(y0, t0, dt, tf, deriv, interv=lambda x, t: (x, t), rk_kwargs={}, deriv_kwargs={}, interv_kwargs={}):
    trajectory = ts.TimeSeries()
    trajectory.append(y0, t0)
    t = t0
    y = y0
    while t < tf:
        y, t = rk_step(y, t, dt, deriv, **rk_kwargs, **deriv_kwargs)
        y, t = interv(y, t, **interv_kwargs)
        trajectory.append(y, t)

    return trajectory
