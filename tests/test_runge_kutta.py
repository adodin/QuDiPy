import unittest
from unittest import TestCase
import QuDiPy.util.timeseries as ts
import QuDiPy.util.runge_kutta as rk
import numpy as np



class Testrk_integrate(TestCase):
    def test_rk_linear(self):
        def deriv(y, t):
            return 1
        y0 = 0.
        t0 = 0.
        tf = 1.
        num_time = 50
        dt = (tf - t0)/(num_time - 1)
        t_correct = np.linspace(t0, tf, num_time)
        y_correct = t_correct
        series_correct = ts.TimeSeries()

        for y, t in zip(y_correct, t_correct):
            series_correct.append(y, t)
        series_calculated = rk.rk_integrate(y0, t0, dt, tf, deriv)
        self.assertEqual(series_correct, series_calculated)

    def test_rk_quadratic(self):
        def deriv(y, t):
            return 2*t

        y0 = 0.
        t0 = 0.
        tf = 1.
        num_time = 50
        dt = (tf - t0) / (num_time - 1)
        t_correct = np.linspace(t0, tf, num_time)
        y_correct = t_correct ** 2
        series_correct = ts.TimeSeries()

        for y, t in zip(y_correct, t_correct):
            series_correct.append(y, t)
        series_calculated = rk.rk_integrate(y0, t0, dt, tf, deriv)
        self.assertEqual(series_correct, series_calculated)

    def test_rk_exponential(self):
        def deriv(y, t):
            return - y

        y0 = 1.
        t0 = 0.
        tf = 1.
        num_time = 50
        dt = (tf - t0) / (num_time - 1)
        t_correct = np.linspace(t0, tf, num_time)
        y_correct = np.exp(-t_correct)
        series_correct = ts.TimeSeries()

        for y, t in zip(y_correct, t_correct):
            series_correct.append(y, t)
        series_calculated = rk.rk_integrate(y0, t0, dt, tf, deriv)
        self.assertEqual(series_correct, series_calculated)


if __name__ == '__main__':
    unittest.main()
