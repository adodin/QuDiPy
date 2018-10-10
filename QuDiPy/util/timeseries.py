""" Custom Iterable Container to Hold Time Series Data
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np


class TimeSeries:
    def __getitem__(self, item):
        return self.data[item], self.time[item]

    def __iter__(self):
        return zip(self.data, self.time)

    def __len__(self):
        assert len(self.data) == len(self.time)
        return len(self.time)

    def __add__(self, other):
        assert len(self.data) == len(other.data)
        sum_data = []
        for s, o in zip(self.data, other.data):
            sum_data.append(s + o)
        return self.like(sum_data)

    def __rmul__(self, other):
        product_data = []
        for s in self.data:
            product_data.append(other * s)
        return self.like(product_data)

    def __mul__(self, other):
        assert len(self.data) == len(other.data)
        product_data = []
        for s, o in zip(self.data, other.data):
            product_data.append(s * o)
        return self.like(product_data)

    def __matmul__(self, other):
        assert len(self.data) == len(other.data)
        product_data = []
        for s, o in zip(self.data, other.data):
            product_data.append(s @ o)
        return self.like(product_data)

    def __sub__(self, other):
        assert len(self.data) == len(other.data)
        diff_data = []
        for s, o in zip(self.data, other.data):
            diff_data.append(s - o)
        return self.like(diff_data)

    def __abs__(self):
        return abs(sum(self.data))

    def __round__(self, n=None):
        rounded_data = []
        rounded_time = []
        assert len(self.data) == len(self.time)
        for d, t in zip(self.data, self.time):
            rounded_data.append(round(d))
            rounded_time.append(round(t))
        return TimeSeries(rounded_data, rounded_time)

    def __eq__(self, other):
        data_equality = self.data == other.data
        time_equality = self.time == other.time
        return data_equality and time_equality

    def __init__(self, data=None, time=None):
        if data is None:
            self.data = []
        else:
            self.data = data[:]
        if time is None:
            self.time = []
        else:
            self.time = time[:]

    def append(self, data, time):
        self.data.append(data)
        self.time.append(time)

    def like(self, data):
        return TimeSeries(data, self.time)
