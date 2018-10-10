from unittest import TestCase
import unittest
import QuDiPy.util.timeseries as ts
import numpy as np


class TestTimeSeries(TestCase):
    def test_timeseries_null_init(self):
        series = ts.TimeSeries()
        self.assertEqual([], series.data)
        self.assertEqual([], series.time)
        self.assertEqual(0, len(series))

    def test_timeseries_array_init(self):
        data = [1, 2, 3]
        time = [0., 0.5, 1.5]
        series = ts.TimeSeries(data, time)
        self.assertEqual(data, series.data)
        self.assertEqual(time, series.time)
        self.assertEqual(3, len(series))

    def test_timeseries_append_null_init(self):
        series = ts.TimeSeries()
        series.append('foo', 0.)
        series.append('bar', 1.)
        expected_data = ['foo', 'bar']
        expected_time = [0., 1.]
        self.assertTrue(np.array_equal(series.data, expected_data))
        self.assertTrue(np.array_equal(series.time, expected_time))

    def test_timeseries_append_array_init(self):
        data = [1, 2, 3]
        time = [0., 0.5, 1.5]
        series = ts.TimeSeries(data, time)
        series.append(4, 3.5)
        expected_data = data + [4]
        expected_time = time + [3.5]
        self.assertTrue(np.array_equal(series.data, expected_data))
        self.assertTrue(np.array_equal(series.time, expected_time))

    def test_timeseries_iteration(self):
        series = ts.TimeSeries()
        series.append('foo', 0.)
        series.append('bar', 1.)
        for de, te, (d, t) in zip(['foo', 'bar'], [0., 1.], series):
            self.assertEqual(d, de)
            self.assertEqual(t, te)

    def test_timeseries_getitem(self):
        series = ts.TimeSeries()
        series.append('foo', 0.)
        series.append('bar', 1.)
        self.assertEqual(series[0], ('foo', 0.))
        self.assertEqual(series[1], ('bar', 1.))

    def test_timeseries_len(self):
        series = ts.TimeSeries()
        series.append('foo', 0.)
        self.assertEqual(1, len(series))
        series.append('bar', 1.)
        self.assertEqual(2, len(series))

    def test_timeseries_round(self):
        data = [1.1, 2.2, 1.01]
        r_data = [1., 2., 1.]
        time = [0., 1., 2.]
        series = ts.TimeSeries(data, time)
        expected = ts.TimeSeries(r_data, time)
        calculated = round(series)
        self.assertEqual(expected, calculated)

    def test_timeseries_add(self):
        d1 = [1., 2.]
        d2 = [10., 20.]
        d_sum = [11., 22.]
        time = [0., 1.]
        s1 = ts.TimeSeries(d1, time)
        s2 = ts.TimeSeries(d2, time)
        expected = ts.TimeSeries(d_sum, time)
        actual = s1 + s2
        self.assertEqual(actual, expected)

    def test_timeseries_sub(self):
        d1 = [1., 2.]
        d2 = [10., 20.]
        d_sum = [9., 18.]
        time = [0., 1.]
        s1 = ts.TimeSeries(d1, time)
        s2 = ts.TimeSeries(d2, time)
        expected = ts.TimeSeries(d_sum, time)
        actual = s2 - s1
        self.assertEqual(actual, expected)

    def test_timeseries_mul(self):
        d1 = [1., 2.]
        d2 = [10., 20.]
        d_sum = [10., 40.]
        time = [0., 1.]
        s1 = ts.TimeSeries(d1, time)
        s2 = ts.TimeSeries(d2, time)
        expected = ts.TimeSeries(d_sum, time)
        actual = s2 * s1
        self.assertEqual(actual, expected)

    def test_timeseries_rmul(self):
        d1 = [1., 2.]
        d_sum = [2., 4.]
        time = [0., 1.]
        s1 = ts.TimeSeries(d1, time)
        expected = ts.TimeSeries(d_sum, time)
        actual = 2 * s1
        self.assertEqual(actual, expected)

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
