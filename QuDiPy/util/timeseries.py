""" Custom Iterable Container to Hold Time Series Data
Written by: Amro Dodin (Willard Group - MIT)

"""

import numpy as np

class TimeSeries:
    def append(self, data, time):
        self.data.append(data)
        self.time.append(time)
        self.length += 1

    def __iter__(self):
        return self

    def __eq__(self, other):
        data_equality = np.array_equal(self.data, other.data)
        time_equality = np.array_equal(self.time, other.time)
        return data_equality and time_equality

    def __next__(self):
        if self.__index == self.length:
            raise StopIteration
        d, t = (self.data[self.__index], self.time[self.__index])
        self.__index += 1
        return d, t

    def __init__(self):
        self.data = []
        self.time = []
        self.length = 0

        self.__index = 0
