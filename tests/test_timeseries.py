from unittest import TestCase
import unittest
import QuDiPy.util.timeseries as ts
import numpy as np


class TestTimeSeries(TestCase):
    def test_timeseries_init(self):
        series = ts.TimeSeries()
        self.assertEqual([], series.data)
        self.assertEqual([], series.time)
        self.assertEqual(0, series.length)

    def test_timeseries_append(self):
        series = ts.TimeSeries()
        series.append('foo', 0.)
        series.append('bar', 1.)
        expected_data = ['foo', 'bar']
        expected_time = [0., 1.]
        self.assertTrue(np.array_equal(series.data, expected_data))
        self.assertTrue(np.array_equal(series.time, expected_time))

    def test_timeseries_iteration(self):
        series = ts.TimeSeries()
        series.append('foo', 0.)
        series.append('bar', 1.)
        for de, te, (d, t) in zip(['foo', 'bar'], [0., 1.], series):
            self.assertEqual(d, de)
            self.assertEqual(t, te)

    def test_timeseries_equality(self):
        series1 = ts.TimeSeries()
        series2 = ts.TimeSeries()
        series3 = ts.TimeSeries()
        series4 = ts.TimeSeries()
        series1.append('foo', 0.)
        series1.append('bar', 1.)
        series2.append('foo', 0.)
        series2.append('bar', 1.)
        series3.append('foo', 0.)
        series3.append('baz', 1.)
        series4.append('foo', 0.)
        series4.append('bar', 2.)
        self.assertEqual(series1, series2)
        self.assertTrue(series1 == series2)
        self.assertFalse(series1 != series2)
        self.assertNotEqual(series1, series3)
        self.assertNotEqual(series1, series4)
        self.assertTrue(series1 != series3)
        self.assertFalse(series1 == series3)
